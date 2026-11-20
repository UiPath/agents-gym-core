"""Graceful-termination valves in the custom LLM node.

When ``graceful_termination=True``, the node forces ``end_execution`` instead
of raising at the iteration limit. When ``detect_tool_repetition=True``, it
also forces termination when a single tool call repeats ``max(limit // 3, 2)``
times. Both produce a scorable output rather than an error.
"""

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from uipath.agent.react import END_EXECUTION_TOOL
from uipath_langchain.agent.react import AgentGraphState

from agents_gym_core.loops.nodes import create_custom_llm_node

# -- fixtures ----------------------------------------------------------------


@tool
def lookup(query: str) -> str:
    """Look up a value."""
    return f"result for {query}"


class _InnerState(BaseModel):
    tools_storage: dict[str, Any] = Field(default_factory=dict)


ALL_TOOLS: list[Any] = [lookup, END_EXECUTION_TOOL]


class StubModel:
    """Chat model stand-in that records bound tools and invocation messages."""

    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.last_tools: list[Any] = []
        self.last_kwargs: dict[str, Any] = {}
        self.last_messages: list[Any] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StubModel":
        self.last_tools = list(tools)
        self.last_kwargs = kwargs
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        self.last_messages = list(messages)
        return self.response


def _node(model: StubModel, **kwargs: Any) -> Any:
    return create_custom_llm_node(model, tools=ALL_TOOLS, **kwargs)  # type: ignore[arg-type]


def _state_at_step(
    step: int,
    *,
    tool_counts: dict[str, int] | None = None,
) -> AgentGraphState:
    """Build a state whose tools_storage records *step* prior LLM calls."""
    storage: dict[str, Any] = {"total_llm_calls": step}
    if tool_counts is not None:
        storage["tool_counts"] = tool_counts
    messages: list[Any] = [
        SystemMessage(content="You are a helpful agent."),
        HumanMessage(content="find x"),
    ]
    for i in range(min(step, 3)):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[{"name": "lookup", "args": {"query": f"q{i}"}, "id": f"call-{i}"}],
            )
        )
        messages.append(ToolMessage(content=f"result {i}", tool_call_id=f"call-{i}", name="lookup"))
    state = AgentGraphState(messages=messages)
    state.inner_state = _InnerState(tools_storage=storage)  # type: ignore[attr-defined]
    return state


def _end_execution_response(**kwargs: Any) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "end_execution", "args": kwargs, "id": "call-end"}],
    )


def _normal_tool_response() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "lookup", "args": {"query": "x"}, "id": "call-n"}],
    )


# -- Valve A: step-limit forced termination ----------------------------------


async def test_forces_end_at_step_limit() -> None:
    model = StubModel(_end_execution_response(answer="forced"))
    node = _node(model, llm_messages_limit=5, graceful_termination=True)
    state = _state_at_step(5)

    result = await node(state)

    bound_names = [t.name for t in model.last_tools]
    assert bound_names == ["end_execution"]
    assert model.last_kwargs.get("tool_choice") == "any"
    human_contents = [m.content for m in model.last_messages if isinstance(m, HumanMessage)]
    assert any("MUST call end_execution" in c for c in human_contents)
    assert result["messages"][0].tool_calls[0]["name"] == "end_execution"


async def test_normal_below_step_limit() -> None:
    model = StubModel(_normal_tool_response())
    node = _node(model, llm_messages_limit=5, graceful_termination=True)
    state = _state_at_step(3)

    result = await node(state)

    bound_names = [t.name for t in model.last_tools]
    assert "lookup" in bound_names
    assert "end_execution" in bound_names
    human_contents = [m.content for m in model.last_messages if isinstance(m, HumanMessage)]
    assert not any("MUST call end_execution" in c for c in human_contents)
    assert result["messages"][0].tool_calls[0]["name"] == "lookup"


# -- Valve B: repeated tool-call forced termination --------------------------


async def test_forces_end_on_repeated_tool() -> None:
    model = StubModel(_end_execution_response(answer="repeated"))
    node = _node(
        model,
        llm_messages_limit=6,
        graceful_termination=True,
        detect_tool_repetition=True,
    )
    # threshold = max(6 // 3, 2) = 2; set count to 2 to trigger
    repeated_key = json.dumps({"query": "same"}, sort_keys=True)
    state = _state_at_step(3, tool_counts={f"lookup:{repeated_key}": 2})

    result = await node(state)

    bound_names = [t.name for t in model.last_tools]
    assert bound_names == ["end_execution"]
    assert model.last_kwargs.get("tool_choice") == "any"
    human_contents = [m.content for m in model.last_messages if isinstance(m, HumanMessage)]
    assert any("MUST call end_execution" in c for c in human_contents)
    assert result["messages"][0].tool_calls[0]["name"] == "end_execution"


async def test_no_force_below_repetition_threshold() -> None:
    model = StubModel(_normal_tool_response())
    node = _node(
        model,
        llm_messages_limit=6,
        graceful_termination=True,
        detect_tool_repetition=True,
    )
    # threshold = max(6 // 3, 2) = 2; count 1 is below
    repeated_key = json.dumps({"query": "same"}, sort_keys=True)
    state = _state_at_step(2, tool_counts={f"lookup:{repeated_key}": 1})

    result = await node(state)

    bound_names = [t.name for t in model.last_tools]
    assert "lookup" in bound_names
    assert "end_execution" in bound_names
    human_contents = [m.content for m in model.last_messages if isinstance(m, HumanMessage)]
    assert not any("MUST call end_execution" in c for c in human_contents)
    assert result["messages"][0].tool_calls[0]["name"] == "lookup"


async def test_threshold_floor_is_two() -> None:
    """With llm_messages_limit=5, threshold = max(5 // 3, 2) = 2. One repeat must not fire."""
    model = StubModel(_normal_tool_response())
    node = _node(
        model,
        llm_messages_limit=5,
        graceful_termination=True,
        detect_tool_repetition=True,
    )
    repeated_key = json.dumps({"query": "same"}, sort_keys=True)
    state = _state_at_step(2, tool_counts={f"lookup:{repeated_key}": 1})

    result = await node(state)

    bound_names = [t.name for t in model.last_tools]
    assert "lookup" in bound_names
    assert result["messages"][0].tool_calls[0]["name"] == "lookup"


# -- Tool-count tracking ----------------------------------------------------


async def test_increments_tool_counts() -> None:
    model = StubModel(_normal_tool_response())
    node = _node(
        model,
        llm_messages_limit=10,
        graceful_termination=True,
        detect_tool_repetition=True,
    )
    state = _state_at_step(1)

    result = await node(state)

    storage = result["inner_state"]["tools_storage"]
    assert storage["total_llm_calls"] == 2
    assert "tool_counts" in storage
    assert any("lookup:" in k for k in storage["tool_counts"])


async def test_step_counter_without_repetition_tracking() -> None:
    model = StubModel(_normal_tool_response())
    node = _node(
        model,
        llm_messages_limit=10,
        graceful_termination=True,
        detect_tool_repetition=False,
    )
    state = _state_at_step(4)

    result = await node(state)

    storage = result["inner_state"]["tools_storage"]
    assert storage["total_llm_calls"] == 5
    assert "tool_counts" not in storage
