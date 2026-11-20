from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from uipath_langchain_client.clients.google.embeddings import UiPathGoogleGenerativeAIEmbeddings
from uipath_langchain_client.clients.openai.embeddings import UiPathAzureOpenAIEmbeddings
from uipath_langchain_client.factory import get_embedding_model as get_embedding_model_factory

from agents_gym_core.models.model_spec import ModelRegistryMeta, ModelSpec
from agents_gym_core.models.utils import (
    UIPATH_RETRY_CONFIG,
    ModelBackend,
    default_backend,
    ensure_embedding_dimensions,
    get_or_create_rate_limiter,
    patch_embeddings_with_rate_limiter,
)
from agents_gym_core.utils.general_utils import hashable_lru_cache
from agents_gym_core.utils.gym_settings_core import gym_settings_core
from agents_gym_core.utils.trace_utils import warning


@hashable_lru_cache
def get_embedding_model(
    model_name: str,
    max_concurrency: int | None = None,
    from_factory: bool = False,
    backend: ModelBackend | str | None = None,
    **kwargs: Any,
) -> Embeddings:
    """Get embedding model instance by name.

    ``backend`` defaults to :func:`~agents_gym_core.models.utils.default_backend`.
    """
    backend = ModelBackend(backend) if backend else default_backend()
    registries = (
        (LangchainOpenAIEmbeddingModels, LangchainGeminiEmbeddingModels, HuggingFaceEmbeddingModels)
        if backend == "langchain"
        else (OpenAIEmbeddingModels, GeminiEmbeddingModels, HuggingFaceEmbeddingModels)
    )
    if not from_factory:
        for registry in registries:
            if model_name in registry:
                return registry[model_name].get_embedding_model(**kwargs)

    if max_concurrency is None:
        max_concurrency = gym_settings_core.UIPATH_AGENTS_GYM_MAX_LLM_CONCURRENCY

    if backend == ModelBackend.LANGCHAIN:
        # Unregistered names go to the direct OpenAI-compatible client, never to gateway discovery.
        return ModelSpec(
            model_name,
            "agents_gym_core.models.openai_compat:OpenAIOrAzureEmbeddings",
            factory_method="create",
            max_concurrency=max_concurrency,
            dimensions=0,
        ).get_embedding_model(**kwargs)
    rate_limiter = None
    if max_concurrency is not None:
        # Doubled for the same reason as in ModelSpec.get_embedding_model.
        rate_limiter = get_or_create_rate_limiter(model_name, max_concurrency * 2)

    try:
        if not from_factory:
            warning(
                f"Model {model_name} not found in any registry, attempting to get it directly from the client factory."
            )
        factory_kwargs = {**kwargs}
        if backend == ModelBackend.UIPATH:
            factory_kwargs.setdefault("retry_config", UIPATH_RETRY_CONFIG)
        model = get_embedding_model_factory(
            model_name,
            **factory_kwargs,
        )
    except ValueError as e:
        raise ValueError(f"Model {model_name} not found in any registry") from e

    model = ensure_embedding_dimensions(model)

    if rate_limiter is not None:
        patch_embeddings_with_rate_limiter(model, rate_limiter)

    return model


class OpenAIEmbeddingModels(metaclass=ModelRegistryMeta):
    """Azure OpenAI embedding models."""

    _default_config: dict[str, Any] = {"retry_config": UIPATH_RETRY_CONFIG}

    text_embedding_3_large = ModelSpec(
        "text-embedding-3-large",
        UiPathAzureOpenAIEmbeddings,
        default_config=_default_config,
        max_concurrency=20,
        dimensions=3072,
    )
    text_embedding_3_small = ModelSpec(
        "text-embedding-3-small",
        UiPathAzureOpenAIEmbeddings,
        default_config=_default_config,
        max_concurrency=20,
        dimensions=1536,
    )
    text_embedding_ada_002 = ModelSpec(
        "text-embedding-ada-002",
        UiPathAzureOpenAIEmbeddings,
        default_config=_default_config,
        max_concurrency=10,
        dimensions=0,
    )


class GeminiEmbeddingModels(metaclass=ModelRegistryMeta):
    """Google Vertex AI embedding models.

    Gemini takes batch_size only per embed_documents() call (default 100), so chunk_size is
    stripped at construction.
    """

    _default_config: dict[str, Any] = {"retry_config": UIPATH_RETRY_CONFIG}
    _kwarg_map: dict[str, str | None] = {"chunk_size": None}

    gemini_embedding_001 = ModelSpec(
        "gemini-embedding-001",
        UiPathGoogleGenerativeAIEmbeddings,
        default_config=_default_config,
        max_concurrency=20,
        dimensions=3072,
        kwarg_map=_kwarg_map,
    )


class HuggingFaceEmbeddingModels(metaclass=ModelRegistryMeta):
    """HuggingFace local embedding models.

    chunk_size is forwarded as encode_kwargs["batch_size"]. No rate limiter, since inference
    is local.
    """

    qwen3_embedding_0_6b = ModelSpec(
        "Qwen/Qwen3-Embedding-0.6B",
        "langchain_huggingface:HuggingFaceEmbeddings",
        default_config={
            "model_kwargs": {"device": "cuda", "model_kwargs": {"dtype": "float16"}},
            "encode_kwargs": {"normalize_embeddings": True},
        },
        max_concurrency=None,
        kwarg_map={
            "chunk_size": "encode_kwargs.batch_size",
            "max_retries": None,
            "request_timeout": None,
        },
        model_param="model_name",
        dimensions=1024,
    )


class LangchainOpenAIEmbeddingModels(metaclass=ModelRegistryMeta):
    """OpenAI embedding models via direct API (see :mod:`agents_gym_core.models.openai_compat`)."""

    text_embedding_3_large = ModelSpec(
        "text-embedding-3-large",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureEmbeddings",
        factory_method="create",
        max_concurrency=20,
        dimensions=3072,
    )
    text_embedding_3_small = ModelSpec(
        "text-embedding-3-small",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureEmbeddings",
        factory_method="create",
        max_concurrency=20,
        dimensions=1536,
    )
    text_embedding_ada_002 = ModelSpec(
        "text-embedding-ada-002",
        "agents_gym_core.models.openai_compat:OpenAIOrAzureEmbeddings",
        factory_method="create",
        max_concurrency=10,
        dimensions=0,  # ada-002 rejects the dimensions parameter
    )


class LangchainGeminiEmbeddingModels(metaclass=ModelRegistryMeta):
    """Google Gemini embedding models via direct API (requires GOOGLE_API_KEY)."""

    _kwarg_map: dict[str, str | None] = {"chunk_size": None}

    gemini_embedding_001 = ModelSpec(
        "gemini-embedding-001",
        GoogleGenerativeAIEmbeddings,
        max_concurrency=20,
        dimensions=3072,
        kwarg_map=_kwarg_map,
    )
