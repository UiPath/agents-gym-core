"""The custom LLM node must fail a run whose model stops calling tools.

The router treats a content-only turn as progress, so the node is the only
place a deployment that ignores forced ``tool_choice`` gets caught.
"""

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.react import AgentGraphState

from agents_gym_core.loops.nodes import create_custom_llm_node


@tool
def ping(value: str) -> str:
    """Echo back the value."""
    return value


class StubModel:
    """Chat model stand-in that records how its tools were bound."""

    def __init__(self, response: AIMessage) -> None:
        self.response = response
        self.bind_kwargs: dict[str, Any] = {}

    def bind_tools(self, tools: Any, **kwargs: Any) -> "StubModel":
        self.bind_kwargs = kwargs
        return self

    async def ainvoke(self, messages: Any) -> AIMessage:
        return self.response


def _tool_less_state(turns: int) -> AgentGraphState:
    messages: list[Any] = [HumanMessage(content="go")]
    messages += [AIMessage(content=f"thinking {index}") for index in range(turns)]
    return AgentGraphState(messages=messages)


def _stub_model() -> StubModel:
    return StubModel(
        AIMessage(
            content="",
            tool_calls=[{"name": "ping", "args": {"value": "x"}, "id": "call-1"}],
        )
    )


def _node(model: StubModel, **kwargs: Any) -> Any:
    return create_custom_llm_node(model, tools=[ping], **kwargs)  # type: ignore[arg-type]


async def test_the_limit_turn_forces_a_tool_call() -> None:
    model = _stub_model()
    node = _node(model, thinking_messages_limit=2)

    result = await node(_tool_less_state(2))

    assert model.bind_kwargs["tool_choice"] == "any"
    assert result["messages"][0].tool_calls


@pytest.mark.parametrize("graceful_termination", [False, True])
async def test_a_turn_still_tool_less_after_forcing_fails_the_run(
    graceful_termination: bool,
) -> None:
    node = _node(
        _stub_model(),
        thinking_messages_limit=2,
        graceful_termination=graceful_termination,
    )

    with pytest.raises(AgentRuntimeError) as error:
        await node(_tool_less_state(3))

    assert error.value.as_dict["code"] == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.THINKING_LIMIT_EXCEEDED
    )


async def test_a_conversational_agent_may_answer_without_tools() -> None:
    node = _node(_stub_model(), thinking_messages_limit=2, is_conversational=True)

    assert await node(_tool_less_state(5))


async def test_a_vllm_served_model_is_forced_with_the_directive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Self-hosted vLLM models get the in-context directive, not ``tool_choice="any"``."""
    import agents_gym_core.loops.nodes as nodes
    from agents_gym_core.models.chat.clients import VllmChatModels, get_chat_model
    from agents_gym_core.models.utils import get_model_name

    assert get_model_name(get_chat_model("Qwen/Qwen3.5-9B", backend="langchain")) in VllmChatModels
    seen: list[Any] = []

    class Recording(StubModel):
        async def ainvoke(self, messages: Any) -> AIMessage:
            seen.extend(messages)
            return self.response

    model = Recording(_stub_model().response)
    monkeypatch.setattr(nodes, "get_model_name", lambda _: "Qwen/Qwen3.5-9B")
    await _node(model, thinking_messages_limit=2)(_tool_less_state(2))

    assert model.bind_kwargs.get("tool_choice") != "any"
    assert seen[-1].content == "You must call a tool now."
