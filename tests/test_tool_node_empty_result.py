"""An empty list from a tool must reach the model as "[]", not as zero content blocks."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from uipath_langchain.agent.react import AgentGraphState

from agents_gym_core.loops.nodes import EMPTY_TOOL_RESULT, create_tool_node


@tool
async def lookup(found: bool) -> list[str]:
    """Return the matching records."""
    return ["0031"] if found else []


def _state(found: bool) -> AgentGraphState:
    call = {"name": "lookup", "args": {"found": found}, "id": "call_1"}
    return AgentGraphState(
        messages=[HumanMessage("find"), AIMessage(content="", tool_calls=[call])]
    )


@pytest.mark.parametrize("handle_tool_errors", [True, False])
async def test_empty_list_result_is_sent_as_text(handle_tool_errors: bool) -> None:
    node = create_tool_node([lookup], handle_tool_errors=handle_tool_errors)["lookup"]

    output = await node.ainvoke(_state(found=False))

    message = output["messages"][0]
    assert isinstance(message, ToolMessage)
    assert message.content == EMPTY_TOOL_RESULT


async def test_non_empty_result_is_unchanged() -> None:
    node = create_tool_node([lookup])["lookup"]

    output = await node.ainvoke(_state(found=True))

    assert output["messages"][0].content == "['0031']"
