import os
from enum import StrEnum
from typing import Any

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings

SKIP_DOTENV_ENV = "AGENTS_GYM_SKIP_DOTENV"


def load_env_file() -> None:
    """Load the nearest ``.env`` unless ``$AGENTS_GYM_SKIP_DOTENV`` is set.

    Variables already in the environment win. Tests set the switch so a developer's
    provider keys never send stubbed runs to a real model; with it set, importing
    ``agents_gym_core`` also keeps litellm from loading the file.
    """
    if not os.environ.get(SKIP_DOTENV_ENV):
        load_dotenv(find_dotenv())


class LLMBackendType(StrEnum):
    AGENTHUB = "agenthub"
    LLMGATEWAY = "llmgateway"


class GymSettingsCore(BaseSettings):
    """Core settings for Agents Gym Core."""

    UIPATH_AGENTS_GYM_LOOP_RECURSION_LIMIT: int = 20
    UIPATH_AGENTS_GYM_NO_TOOL_MESSAGES_LIMIT: int = 5
    UIPATH_AGENTS_GYM_PARALLEL_TOOL_CALLS: bool = False
    UIPATH_AGENTS_GYM_THROW_GRAPH_RECURSION_ERROR_AFTER_N_SAME_MESSAGES: int = 5

    # Extra keyword arguments for the loop constructor; SummarizingLoop reads the langmem
    # summarization keys and defaults those left out. Example:
    # {"max_summary_tokens": 500, "max_tokens_before_summary": 600, "max_tokens": 2000,
    #  "initial_summary_prompt": "core::summary_initial_prompt.md",
    #  "existing_summary_prompt": "core::summary_existing_prompt.md"}
    UIPATH_AGENTS_GYM_SUMMARIZATION_SETTINGS: dict[str, Any] = {}
    # Attachment structured-output ``strict`` option, forwarded to the selected model
    # client's ``with_structured_output`` method.
    UIPATH_AGENTS_GYM_LLM_STRICT_MODE: bool = True

    # LLM default settings
    UIPATH_AGENTS_GYM_TEMPERATURE: float = 1.0
    UIPATH_AGENTS_GYM_MAX_TOKENS: int | None = None
    UIPATH_AGENTS_GYM_MAX_RETRIES: int = 5
    UIPATH_AGENTS_GYM_REQUEST_TIMEOUT: int = 300

    # LLM judge client defaults (GLOBAL_DEFAULT_JUDGE_CONFIG). Judges routed through
    # models/judge_routing.py always pass their evaluator JSON's temperature and max_tokens
    # (max_tokens unset: 8000 for Claude Haiku and Sonnet 4.5, else 4096), which override the
    # TEMPERATURE and MAX_TOKENS values here. Only MAX_RETRIES and REQUEST_TIMEOUT apply.
    UIPATH_AGENTS_GYM_LLM_JUDGE_TEMPERATURE: float = 0.0
    UIPATH_AGENTS_GYM_LLM_JUDGE_MAX_TOKENS: int | None = None
    UIPATH_AGENTS_GYM_LLM_JUDGE_MAX_RETRIES: int = 5
    UIPATH_AGENTS_GYM_LLM_JUDGE_REQUEST_TIMEOUT: int = 300

    UIPATH_AGENTS_GYM_MAX_LLM_CONCURRENCY: int = 20

    # LLM service settings
    UIPATH_LLM_SERVICE: str = os.getenv("UIPATH_LLM_SERVICE", LLMBackendType.AGENTHUB)

    # Skip every agent's ``init:`` checks and scripts (e.g. when eval-set files are present).
    UIPATH_GYM_SKIP_INIT: bool = False

    def __init__(self, **kwargs: Any):
        load_env_file()
        super().__init__(**kwargs)


gym_settings_core = GymSettingsCore()
