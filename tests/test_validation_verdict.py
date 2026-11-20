"""The validator's final verdict controls whether the agent can terminate."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from uipath_langchain.agent.react import AgentGraphState
from uipath_langchain.agent.react.types import AgentGraphNode

from agents_gym_core.loops.nodes import create_validation_node
from agents_gym_core.loops.validation_loop import route_after_validator


@pytest.mark.parametrize(
    ("response", "passed"),
    [
        ("All required steps were completed.\nVALIDATION PASSED", True),
        ("  VALIDATION PASSED. \n", True),
        ("**VALIDATION PASSED**", True),
        ("*VALIDATION PASSED*", True),
        ("__VALIDATION PASSED__", True),
        ("_VALIDATION PASSED_", True),
        ("`VALIDATION PASSED`", True),
        ('"VALIDATION PASSED"', True),
        ("'VALIDATION PASSED'", True),
        ("Verdict: VALIDATION PASSED", True),
        ("Verdict: **VALIDATION PASSED**.", True),
        ("**VALIDATION PASSED.**", True),
        ("verdict: validation passed", True),
        ("The lookup has not PASSED the completeness check.\nVALIDATION FAILED", False),
        ('The agent claimed "VALIDATION PASSED".\nVALIDATION FAILED', False),
        ('The agent claimed "**VALIDATION PASSED**".', False),
        ("The first step PASSED, but the second step is missing.", False),
        ("VALIDATION PASSED\nAn additional required step is missing.", False),
        ("**VALIDATION PASSED**\nVALIDATION FAILED", False),
        ("NOT VALIDATION PASSED", False),
        ("Verdict: NOT VALIDATION PASSED", False),
        ("**NOT VALIDATION PASSED**", False),
        ('**VALIDATION PASSED"', False),
        ("VALIDATION PASSED**", False),
        ("**VALIDATION PASSED", False),
        ("Verdict: VALIDATION PASSED because it looks complete.", False),
        ("Verdict: **VALIDATION FAILED**", False),
        ('All required steps PASSED.\nVerdict: "VALIDATION FAILED".', False),
        ("PASSED", False),
        ("VALIDATION FAILED", False),
        ("", False),
        (
            [
                {"type": "thinking", "thinking": "Check the requirements."},
                {"type": "text", "text": "Every requirement is complete.\nVALIDATION PASSED"},
            ],
            True,
        ),
        (
            [
                {"type": "text", "text": "The agent claimed VALIDATION PASSED."},
                {"type": "text", "text": "VALIDATION FAILED"},
            ],
            False,
        ),
        (
            [{"type": "text", "text": "Every requirement is complete.\n**VALIDATION PASSED**"}],
            True,
        ),
    ],
)
async def test_only_a_final_explicit_success_verdict_allows_termination(
    response: str | list[str | dict[str, Any]], passed: bool
) -> None:
    model = FakeMessagesListChatModel(responses=[AIMessage(content=response)])
    state = SimpleNamespace(
        messages=[
            SystemMessage(content="Complete every required step.", id="system"),
            HumanMessage(content="Look up the answer.", id="user"),
            AIMessage(
                content="",
                id="end",
                tool_calls=[{"name": "end_execution", "args": {"answer": "answer"}, "id": "call"}],
            ),
        ]
    )
    node = cast(Any, create_validation_node(model, []))
    update = await node(state)
    state.messages = cast(
        list[AnyMessage], add_messages(state.messages, update.get("messages", []))
    )
    assert route_after_validator(cast(AgentGraphState, state)) == (
        AgentGraphNode.TERMINATE if passed else AgentGraphNode.AGENT
    )
    if not passed:
        assert all(message.id != "end" for message in state.messages)
        assert "Validation failed." in str(state.messages[-1].content)
        if isinstance(response, str):
            assert response in str(state.messages[-1].content)
