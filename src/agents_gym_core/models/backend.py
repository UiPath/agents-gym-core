"""Process-wide model backend switch used by the entry points (testbed, optimizer)."""

import os

from agents_gym_core.models.chat.clients import get_chat_model
from agents_gym_core.models.embeddings.clients import get_embedding_model
from agents_gym_core.models.utils import LLM_BACKEND_ENV, ModelBackend

__all__ = ["configure_model_resolution", "set_default_backend"]


def set_default_backend(backend: ModelBackend | str) -> ModelBackend:
    """Make ``backend`` the default for this process and its subprocesses; drop cached clients.

    Entry points call this with their ``--model_backend`` so every model role resolves on the
    same backend: agent, optimizer, LLM judges and tool-internal models alike.
    """
    resolved = ModelBackend(backend)
    os.environ[LLM_BACKEND_ENV] = resolved.value
    get_chat_model.cache_clear()  # type: ignore[attr-defined]
    get_embedding_model.cache_clear()  # type: ignore[attr-defined]
    return resolved


def configure_model_resolution() -> None:
    """Route LLM judges through the registry, then apply ``AGENTS_GYM_MODEL_OVERRIDES``.

    What every entry point that evaluates agents (testbed, GEPA runners) does before
    resolving models.
    """
    from agents_gym_core.models.judge_routing import route_llm_judges_through_registry
    from agents_gym_core.models.overrides import apply_configured_model_overrides

    route_llm_judges_through_registry()
    apply_configured_model_overrides()
