"""Trimming keeps tool calls paired, and the context budget follows the optimizer model."""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents_gym_core.prompt_optimization.mcp_tools import helpers
from agents_gym_core.prompt_optimization.react_optimizer import optimizer


@pytest.mark.parametrize("max_chars", range(8_000, 17_000, 500))  # all below the 17,409 total
def test_trimming_keeps_tool_calls_paired(max_chars: int) -> None:
    messages: list = [HumanMessage("protocol " + "p" * 3000)]
    for i in range(6):  # parallel calls; every other turn also carries text
        calls = [{"name": "get_full_results", "args": {}, "id": f"c{i}{j}"} for j in range(2)]
        messages.append(AIMessage("t" * 800 * (i % 2), tool_calls=calls))
        messages += [ToolMessage("r" * 1000, tool_call_id=c["id"]) for c in calls]
    sent = optimizer._build_pre_model_hook(max_chars)({"messages": messages})["llm_input_messages"]
    assert len(sent) < len(messages)  # this size trims
    assert not any(isinstance(m, SystemMessage) for m in sent)  # Claude rejects a mid-list one
    asked: set[str | None] = set()
    for m in sent:
        if isinstance(m, AIMessage):
            asked |= {c["id"] for c in m.tool_calls}
        elif isinstance(m, ToolMessage):
            assert m.tool_call_id in asked  # no result without its call
    assert asked == {m.tool_call_id for m in sent if isinstance(m, ToolMessage)}


@pytest.mark.parametrize(
    ("model", "budget"),
    [("gpt-5-2025-08-07", 600_000), ("Qwen/Qwen3.5-9B", 400_000), ("Qwen/Qwen3.5-2B", 400_000)],
)
async def test_context_budget_follows_the_optimizer_model(
    model: str, budget: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[int] = []

    def stop(**kwargs: object) -> None:
        raise RuntimeError("stop")

    @asynccontextmanager
    async def no_tools():
        yield []

    monkeypatch.setattr(optimizer, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(helpers, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(optimizer, "get_chat_model", lambda *a, **k: object())
    monkeypatch.setattr(
        optimizer, "_build_pre_model_hook", lambda max_chars: seen.append(max_chars)
    )
    monkeypatch.setattr(optimizer, "coach_light_session", no_tools)
    monkeypatch.setattr(optimizer, "create_react_agent", stop)
    with pytest.raises(RuntimeError, match="stop"):
        await optimizer.run_react_optimizer(
            "claim_history:train", optimizer_model=model, output_dir="tmp_b"
        )
    assert seen == [budget]
