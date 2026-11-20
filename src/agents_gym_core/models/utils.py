import os
from enum import StrEnum
from functools import wraps
from typing import Any

import httpx
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.rate_limiters import BaseRateLimiter, InMemoryRateLimiter
from langchain_core.runnables import Runnable
from pydantic import BaseModel
from uipath.llm_client.utils.exceptions import (
    UiPathGatewayTimeoutError,
    UiPathRateLimitError,
    UiPathTooManyRequestsError,
)

from agents_gym_core.utils.gym_settings_core import gym_settings_core


class ModelBackend(StrEnum):
    UIPATH = "uipath"
    LANGCHAIN = "langchain"


LLM_BACKEND_ENV = "AGENTS_GYM_LLM_BACKEND"


def default_backend() -> ModelBackend:
    """Backend for callers that do not pass one: ``$AGENTS_GYM_LLM_BACKEND``, else ``langchain``.

    ``uipath`` resolves models through the UiPath LLM Gateway; ``langchain`` calls providers
    directly with their own API keys (see ``models/openai_compat.py``). Subprocesses inherit
    the variable, so the optimizer, its MCP server and the testbed runs agree.
    """
    return ModelBackend(os.environ.get(LLM_BACKEND_ENV) or ModelBackend.LANGCHAIN)


UIPATH_RETRY_CONFIG: dict[str, Any] = {
    "retry_on_exceptions": (
        UiPathRateLimitError,
        UiPathGatewayTimeoutError,
        UiPathTooManyRequestsError,
        httpx.TimeoutException,
    ),
}

# ``AllowFull4xxResponse`` makes the UiPath LLM Gateway return the provider's 4xx body
# instead of ``{"message": null}``. ``TimeoutSeconds`` lowers the gateway's server-side
# timeout (the client's default is 895 s) below the 300 s request timeout. These headers
# are merged over the client's own.
UIPATH_DEFAULT_HEADERS: dict[str, str] = {
    "X-UiPath-LLMGateway-TimeoutSeconds": "295",
    "X-UiPath-LLMGateway-AllowFull4xxResponse": "true",
}

GLOBAL_DEFAULT_MODEL_CONFIG: dict[str, Any] = dict(
    temperature=gym_settings_core.UIPATH_AGENTS_GYM_TEMPERATURE,
    max_tokens=gym_settings_core.UIPATH_AGENTS_GYM_MAX_TOKENS,
    max_retries=gym_settings_core.UIPATH_AGENTS_GYM_MAX_RETRIES,
    request_timeout=gym_settings_core.UIPATH_AGENTS_GYM_REQUEST_TIMEOUT,
    default_headers=UIPATH_DEFAULT_HEADERS,
)

GLOBAL_DEFAULT_JUDGE_CONFIG: dict[str, Any] = dict(
    temperature=gym_settings_core.UIPATH_AGENTS_GYM_LLM_JUDGE_TEMPERATURE,
    max_tokens=gym_settings_core.UIPATH_AGENTS_GYM_LLM_JUDGE_MAX_TOKENS,
    max_retries=gym_settings_core.UIPATH_AGENTS_GYM_LLM_JUDGE_MAX_RETRIES,
    request_timeout=gym_settings_core.UIPATH_AGENTS_GYM_LLM_JUDGE_REQUEST_TIMEOUT,
    default_headers=UIPATH_DEFAULT_HEADERS,
)

GLOBAL_EMBEDDING_MODEL_CONFIG: dict[str, Any] = dict(
    max_retries=gym_settings_core.UIPATH_AGENTS_GYM_MAX_RETRIES,
    request_timeout=gym_settings_core.UIPATH_AGENTS_GYM_REQUEST_TIMEOUT,
    default_headers=UIPATH_DEFAULT_HEADERS,
)

_RATE_LIMITERS: dict[str, BaseRateLimiter] = {}

_MODEL_ALIAS_ATTR = "_gym_model_alias"


def set_model_alias(model: BaseChatModel | Embeddings, alias: str) -> None:
    """Record the ModelSpec alias a model instance was built from."""
    # These are Pydantic models with no such field, so bypass field validation.
    object.__setattr__(model, _MODEL_ALIAS_ATTR, alias)


def get_model_name(model: BaseChatModel | Embeddings | Runnable) -> str:
    """Return the config identity of a model, used for loop keys and experiment paths.

    The ModelSpec alias wins over the vendor ``model_name``, which variants of
    one deployment share.
    """
    return getattr(model, _MODEL_ALIAS_ATTR, None) or (
        getattr(model, "model_name", "")
        or getattr(model, "model", "")
        or getattr(model, "model_id", "unknown")
    )


def get_or_create_rate_limiter(model_name: str, max_concurrency: int) -> BaseRateLimiter:
    """Return the shared limiter for ``model_name``.

    ``max_concurrency`` is a rate, not a cap on in-flight requests: the token bucket admits
    ``max_concurrency`` requests per second (bursts of up to that many), however long each
    request takes.
    """
    if model_name not in _RATE_LIMITERS:
        _RATE_LIMITERS[model_name] = InMemoryRateLimiter(
            requests_per_second=max_concurrency,
            check_every_n_seconds=0.1,
            max_bucket_size=max_concurrency,
        )
    return _RATE_LIMITERS[model_name]


def patch_embeddings_with_rate_limiter(model: Embeddings, rate_limiter: BaseRateLimiter) -> None:
    """Make aembed_query and aembed_documents acquire ``rate_limiter`` before each call.

    BaseChatModel takes a rate_limiter natively; Embeddings does not.
    """
    original_aembed_query = model.aembed_query
    original_aembed_documents = model.aembed_documents

    @wraps(original_aembed_query)
    async def rate_limited_aembed_query(text: str) -> list[float]:
        await rate_limiter.aacquire(blocking=True)
        return await original_aembed_query(text)

    @wraps(original_aembed_documents)
    async def rate_limited_aembed_documents(texts: list[str]) -> list[list[float]]:
        await rate_limiter.aacquire(blocking=True)
        return await original_aembed_documents(texts)

    # Bypass Pydantic's __setattr__ validation.
    object.__setattr__(model, "aembed_query", rate_limited_aembed_query)
    object.__setattr__(model, "aembed_documents", rate_limited_aembed_documents)


def ensure_embedding_dimensions(model: Embeddings, dimensions: int | None = None) -> Embeddings:
    """Return ``model`` with ``dimensions`` applied.

    ``0`` returns the model untouched (e.g. ada-002, whose API rejects the parameter). ``None``
    probes the native size with one ``embed_query`` call and records it on the model. A
    positive value is set on the model, which is wrapped in :class:`TruncatedEmbeddings` if
    it had no ``dimensions`` attribute.
    """
    if dimensions == 0:
        return model

    has_dimensions = hasattr(model, "dimensions")
    probed = False

    if not dimensions:  # None
        probed = True
        dimensions = len(model.embed_query("test"))

    if isinstance(model, BaseModel):
        model = model.model_copy(update={"dimensions": dimensions})
    else:
        setattr(model, "dimensions", dimensions)

    # A probed size, or a native ``dimensions`` attribute, needs no truncation.
    if probed or has_dimensions:
        return model

    return TruncatedEmbeddings(base_model=model, target_dims=dimensions)


class TruncatedEmbeddings(Embeddings):
    """Truncate embedding vectors to ``target_dims`` and re-normalize them to unit L2 norm."""

    def __init__(self, base_model: Embeddings, target_dims: int):
        self.base_model = base_model
        self.dimensions = target_dims

    def _truncate_and_normalize(self, vectors: list[list[float]]) -> list[list[float]]:
        arr = np.array(vectors)[:, : self.dimensions]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        arr = arr / norms
        return arr.tolist()  # pyright: ignore[reportReturnType]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raw = self.base_model.embed_documents(texts)
        return self._truncate_and_normalize(raw)

    def embed_query(self, text: str) -> list[float]:
        raw = self.base_model.embed_query(text)
        return self._truncate_and_normalize([raw])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        raw = await self.base_model.aembed_documents(texts)
        return self._truncate_and_normalize(raw)

    async def aembed_query(self, text: str) -> list[float]:
        raw = await self.base_model.aembed_query(text)
        return self._truncate_and_normalize([raw])[0]
