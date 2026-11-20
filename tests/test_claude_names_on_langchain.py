"""Bedrock / Vertex Claude names reach the Anthropic API on the langchain backend."""

import pytest
from langchain_anthropic import ChatAnthropic

from agents_gym_core.models.chat.clients import _direct_anthropic_name, get_chat_model
from agents_gym_core.models.utils import LLM_BACKEND_ENV


@pytest.mark.parametrize(
    ("name", "direct"),
    [
        ("anthropic.claude-sonnet-4-5-20250929-v1:0", "claude-sonnet-4-5-20250929"),
        ("anthropic.claude-opus-4-6-v1", "claude-opus-4-6"),  # the paper's Opus optimizer
        ("anthropic.claude-opus-5-5", "claude-opus-5-5"),
        ("claude-sonnet-4-5@20250929", "claude-sonnet-4-5-20250929"),
        ("anthropic.claude-sonnet-4-6", "claude-sonnet-4-6"),
        ("gpt-4o-2024-11-20", "gpt-4o-2024-11-20"),
        ("anthropic.claude-unknown-v1:0", "anthropic.claude-unknown-v1:0"),
    ],
)
def test_direct_anthropic_name(name: str, direct: str) -> None:
    assert _direct_anthropic_name(name) == direct


def test_a_bedrock_judge_name_builds_chat_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(LLM_BACKEND_ENV, "langchain")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    get_chat_model.cache_clear()  # type: ignore[attr-defined]

    model = get_chat_model("anthropic.claude-opus-4-6-v1", is_judge=True)

    assert isinstance(model, ChatAnthropic) and model.model == "claude-opus-4-6"
    get_chat_model.cache_clear()  # type: ignore[attr-defined]
