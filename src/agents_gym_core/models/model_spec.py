import importlib
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from uipath_langchain_client.base_client import UiPathBaseLLMClient

from agents_gym_core.models.utils import (
    GLOBAL_DEFAULT_JUDGE_CONFIG,
    GLOBAL_DEFAULT_MODEL_CONFIG,
    GLOBAL_EMBEDDING_MODEL_CONFIG,
    ensure_embedding_dimensions,
    get_or_create_rate_limiter,
    patch_embeddings_with_rate_limiter,
    set_model_alias,
)
from agents_gym_core.utils.general_utils import deep_freeze, hashable_lru_cache

_DIAGNOSTIC_IDENTIFIERS = frozenset(
    {
        "model",
        "model_name",
        "model_id",
        "api_version",
        "openai_api_version",
        "deployment",
        "deployment_name",
        "deployment_id",
        "azure_deployment",
        "azure_deployment_name",
    }
)
_DIAGNOSTIC_ENDPOINTS = frozenset(
    {"azure_endpoint", "base_url", "openai_api_base", "api_base", "endpoint", "endpoint_url"}
)
_CREDENTIAL_KEYS = frozenset(
    {"api_key", "api_token", "access_token", "token", "password", "secret", "authorization"}
)


def _diagnostic_identifier(value: Any, credentials: set[str]) -> str | None:
    if (
        type(value) is str
        and len(value) <= 200
        and value not in credentials
        and "://" not in value
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]*", value)
    ):
        return value
    return None


def _diagnostic_host(value: Any) -> str | None:
    if type(value) is not str or not re.fullmatch(r"[\x21-\x7e]{1,4096}", value):
        return None
    try:
        endpoint = urlsplit(value)
        host = endpoint.hostname
    except ValueError:
        return None
    if (
        endpoint.scheme in {"http", "https"}
        and host is not None
        and re.fullmatch(r"[A-Za-z0-9.-]{1,253}|[A-Fa-f0-9:]{2,39}", host)
    ):
        return host
    return None


@dataclass
class ModelSpec:
    """Model specification for chat models or embeddings.

    model_class: a concrete type *or* a ``"module.path:ClassName"`` string that
        is resolved lazily on first use. Use the string form for classes behind
        optional dependencies so the registry module can be imported without them.
    kwarg_map: declarative kwarg transformation applied by _build_config (chat and embedding models).
        - None: strip the kwarg (e.g. Gemini, where batch_size is a per-call method param)
        - "key": rename to a top-level constructor key
        - "parent.child": inject into a nested dict, merging with existing values
                          (e.g. "encode_kwargs.batch_size" for HuggingFaceEmbeddings)
    model_param: constructor param name for the model identifier. Defaults to "model";
                 set to "model_name" for HuggingFaceEmbeddings.
    alias: registry key and the identity get_model_name() reports; empty means
        model_name. Set it when specs share a vendor model_name, or they collide
        onto one loop key and experiment directory.
    """

    model_name: str
    model_class: type[BaseChatModel] | type[Embeddings] | str
    default_config: dict[str, Any] = field(default_factory=dict)
    max_concurrency: int | None = 20
    kwarg_map: dict[str, str | None] = field(default_factory=dict)
    model_param: str = "model"
    factory_method: str | None = None
    dimensions: int | None = None
    alias: str = ""

    def __post_init__(self) -> None:
        if not self.alias:
            self.alias = self.model_name

    def _resolve_class(self) -> type:
        """Return the concrete class, importing from a string path if needed."""
        if isinstance(self.model_class, str):
            module_path, class_name = self.model_class.rsplit(":", 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        return self.model_class

    def _build_config(self, base: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Apply kwarg_map transformations to base and kwargs, then set model_param.

        Keys mapped to None are stripped. kwargs override base when keys collide.
        """
        config: dict[str, Any] = {}
        for key, value in {**base, **kwargs}.items():
            target = self.kwarg_map.get(key, key)
            if target is None:
                continue
            if "." in target:
                parent, child = target.split(".", 1)
                nested: dict[str, Any] = dict(config.get(parent, {}))
                nested[child] = value
                config[parent] = nested
            else:
                config[target] = value
        config[self.model_param] = self.model_name
        return config

    def _instantiate(self, config: dict[str, Any]) -> BaseChatModel | Embeddings:
        """Instantiate model_class using factory_method if set, else the constructor."""
        cls = self._resolve_class()
        try:
            if self.factory_method:
                return getattr(cls, self.factory_method)(**config)
            return cls(**config)
        except Exception as e:
            # Compare only plain credential strings, without unwrapping objects or recursing.
            credentials: set[str] = set()
            for key, value in config.items():
                if type(key) is not str:
                    continue
                if type(value) is str and (
                    key.lower() in _CREDENTIAL_KEYS
                    or key.lower().endswith(("_api_key", "_token", "_password", "_secret"))
                ):
                    credentials.add(value)
                elif key in {"headers", "default_headers", "extra_headers"} and isinstance(
                    value, dict
                ):
                    credentials.update(item for item in dict.values(value) if type(item) is str)

            settings: dict[str, str | int | float] = {}
            for key, value in config.items():
                if type(key) is not str:
                    continue
                if (
                    key in {"temperature", "max_tokens", "max_retries", "request_timeout"}
                    and type(value) in (int, float)
                    and -1_000_000_000 <= value <= 1_000_000_000
                ):
                    settings[key] = value
                elif key in _DIAGNOSTIC_IDENTIFIERS:
                    identifier = _diagnostic_identifier(value, credentials)
                    if identifier is not None:
                        settings[key] = identifier
                elif key in _DIAGNOSTIC_ENDPOINTS:
                    host = _diagnostic_host(value)
                    if host is not None and host not in credentials and value not in credentials:
                        settings[f"{key}_host"] = host

            class_name = _diagnostic_identifier(getattr(cls, "__name__", None), credentials)
            model_name = _diagnostic_identifier(self.model_name, credentials)
            raise ValueError(
                f"Failed to instantiate {class_name or '<redacted>'} "
                f"with model {model_name or '<redacted>'} and settings {settings}"
            ) from e

    @hashable_lru_cache
    def get_chat_model(self, is_judge: bool = False, **kwargs: Any) -> BaseChatModel:
        """Create model instance with config overrides and optional throttling."""
        rate_limiter = None
        if self.max_concurrency is not None:
            rate_limiter = get_or_create_rate_limiter(self.model_name, self.max_concurrency)
        config = self._build_config(
            {
                **(GLOBAL_DEFAULT_JUDGE_CONFIG if is_judge else GLOBAL_DEFAULT_MODEL_CONFIG),
                **self.default_config,
            },
            **kwargs,
        )
        config["rate_limiter"] = rate_limiter

        # Apply gateway-specific retry settings and headers only to UiPath clients.
        if not issubclass(self._resolve_class(), UiPathBaseLLMClient):
            config.pop("retry_config", None)
            config.pop("default_headers", None)

        model = self._instantiate(config)
        assert isinstance(model, BaseChatModel), (
            f"Model {self.model_name} is not a BaseChatModel instance"
        )
        set_model_alias(model, self.alias)

        return model

    @hashable_lru_cache
    def get_embedding_model(self, **kwargs: Any) -> Embeddings:
        """Create an embedding model with its dimensions set and, if configured, a rate limiter."""
        rate_limiter = None
        if self.max_concurrency is not None:
            # Doubled: aembed_documents may call aembed_query (or vice versa), so one request
            # can take two of the limiter's per-second tokens.
            rate_limiter = get_or_create_rate_limiter(self.model_name, self.max_concurrency * 2)

        config = self._build_config(
            {**GLOBAL_EMBEDDING_MODEL_CONFIG, **self.default_config}, **kwargs
        )

        # Same guard as get_chat_model: strip UiPath-only kwargs for non-UiPath classes.
        if not issubclass(self._resolve_class(), UiPathBaseLLMClient):
            config.pop("retry_config", None)
            config.pop("default_headers", None)

        model = self._instantiate(config)
        assert isinstance(model, Embeddings), (
            f"Model {self.model_name} is not an Embeddings instance"
        )

        target_dimensions = kwargs.get("dimensions", self.dimensions)
        model = ensure_embedding_dimensions(model, target_dimensions)
        if rate_limiter is not None:
            patch_embeddings_with_rate_limiter(model, rate_limiter)
        # ensure_embedding_dimensions may hand back a copy or a wrapper.
        set_model_alias(model, self.alias)

        return model

    @property
    def name(self) -> str:
        return self.model_name

    def __str__(self) -> str:
        return self.model_name

    def __format__(self, format_spec: str) -> str:
        return self.model_name

    def __hash__(self) -> int:
        return hash(
            (
                self.model_class,
                self.model_name,
                # The cached instance carries the alias; variants need their own.
                self.alias,
                deep_freeze(self.default_config),
                deep_freeze(self.kwarg_map),
                self.factory_method,
            )
        )


class ModelRegistryMeta(type):
    """Metaclass for model registries with [] and 'in' support, keyed by ModelSpec.alias."""

    _index: dict[str, ModelSpec]

    def __new__(mcs, name: str, bases: tuple, namespace: dict) -> "ModelRegistryMeta":
        cls = super().__new__(mcs, name, bases, namespace)
        cls._index = {v.alias: v for v in namespace.values() if isinstance(v, ModelSpec)}
        return cls

    def __setattr__(cls, key: str, value: Any) -> None:
        # Replacing a spec with one of a different alias must drop the old index entry.
        if isinstance(value, ModelSpec):
            old_spec = getattr(cls, key, None)
            if isinstance(old_spec, ModelSpec) and old_spec.alias != value.alias:
                cls._index.pop(old_spec.alias, None)
        super().__setattr__(key, value)
        if isinstance(value, ModelSpec):
            cls._index[value.alias] = value

    def register(cls, spec: ModelSpec) -> ModelSpec:
        """Add a spec that has no class attribute, keyed by its alias."""
        cls._index[spec.alias] = spec
        return spec

    def __contains__(cls, model_name: str) -> bool:
        return model_name in cls._index

    def __getitem__(cls, model_name: str) -> ModelSpec:
        return cls._index[model_name]
