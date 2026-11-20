"""Build the DSPy LM that GEPA reflects with, on the configured model backend.

``langchain`` backend: a litellm ``dspy.LM``,
Azure OpenAI or OpenAI picked from the environment like
:mod:`agents_gym_core.models.openai_compat` (``OPENAI_BASE_URL`` is honoured).
``uipath`` backend: :class:`agents_gym_core.loops.dspy_lm.GymLM` wraps the registry model
and routes reflection calls through the UiPath LLM Gateway.
"""

from __future__ import annotations

import os

import dspy

from agents_gym_core.models.chat import get_chat_model
from agents_gym_core.models.chat.clients import LangchainOpenAIModels
from agents_gym_core.models.utils import ModelBackend, default_backend


def make_dspy_lm(
    model_name: str,
    *,
    max_tokens: int = 16000,
    temperature: float = 0.0,
) -> dspy.BaseLM:
    """Return the reflection LM for ``model_name`` on the configured backend.

    ``langchain`` resolution order:

    1. ``AZURE_OPENAI_API_KEY`` + ``AZURE_OPENAI_ENDPOINT`` → ``azure/<model>``
       (api_version from ``AZURE_OPENAI_API_VERSION`` or ``2025-03-01-preview``).
    2. ``OPENAI_API_KEY`` → ``openai/<model>`` (``OPENAI_BASE_URL`` when set).
    """
    if default_backend() is ModelBackend.UIPATH:
        from agents_gym_core.loops.dspy_lm import GymLM

        chat_model = get_chat_model(model_name, temperature=temperature, max_tokens=max_tokens)
        return GymLM(chat_model, model_name=model_name)

    # The reasoning effort the registry gives the model (ReAPO's, e.g. "low" for GPT-5).
    reasoning: dict = {}
    if model_name in LangchainOpenAIModels:
        effort = (LangchainOpenAIModels[model_name].default_config.get("reasoning") or {}).get(
            "effort"
        )
        reasoning = {"reasoning_effort": effort} if effort else {}

    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return dspy.LM(
            model=f"azure/{model_name}",
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview"),
            max_tokens=max_tokens,
            temperature=temperature,
            **reasoning,
        )

    return dspy.LM(
        model=f"openai/{model_name}",
        api_key=os.environ["OPENAI_API_KEY"],
        api_base=os.environ.get("OPENAI_BASE_URL"),
        max_tokens=max_tokens,
        temperature=temperature,
        **reasoning,
    )
