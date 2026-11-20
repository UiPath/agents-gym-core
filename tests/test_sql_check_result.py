"""The SQL agent's check_result_tool also checks the result against the agent's question.

The deterministic check adds a note only when a result is clearly wrong for the question;
expected results get none.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel
from uipath_langchain.agent.react.utils import create_state_with_input

from agents_gym_core.agents.sql.tools import QueryResultCheckerTool, result_problems
from agents_gym_core.loops.nodes import create_tool_node

SQL_DIR = Path(__file__).resolve().parents[1] / "src" / "agents_gym_core" / "agents" / "sql"
TRAIN_SET = SQL_DIR / "evaluations" / "eval-sets" / "evaluation-set-sql_train.json"
LLM_REPLY = "The result is not empty. No need to re-try!"


def _gold() -> list[tuple[str, str]]:
    pairs = []
    for item in json.loads(TRAIN_SET.read_text())["evaluations"]:
        for criteria in item["evaluationCriterias"].values():
            expected = (criteria or {}).get("expectedOutput") or {}
            if isinstance(expected, dict) and "result" in expected:
                pairs.append((item["inputs"]["query"], str(expected["result"])))
    return pairs


def test_expected_results_get_no_note() -> None:
    gold = _gold()
    assert len({q for q, _ in gold}) == 41
    assert [(q, r) for q, r in gold if result_problems(q, r)] == []


@pytest.mark.parametrize(
    ("question", "result", "expected"),
    [
        ("Get the name of the album with the most tracks in the database.", "[]", "empty"),
        ("Find the name of the artist with the ArtistId of 7.", "  ", "empty"),
        (
            "Find the name of the artist with the ArtistId of 7.",
            "Error: (sqlite3.OperationalError) no such column: Artist.Id",
            "failed",
        ),
        ("Find the invoice total for the invoice with InvoiceId 3.", "Leonie Köhler", "number"),
        ("Get the name and unit price of the least expensive track.", "'Make Believe'", "number"),
        ("How many tracks are in the 'Rock' genre?", "Many tracks.", "number"),
        (
            "Retrieve the full name (first and last) and email of the customer with CustomerId 9.",
            "Kara Nielsen",
            "email",
        ),
        (  # two rows for a single-item question
            "Find the name of the track with the lowest unit price.",
            "[('For Those About To Rock (We Salute You)',), ('Balls to the Wall',)]",
            "2 rows",
        ),
    ],
)
def test_clearly_wrong_results_are_flagged(question: str, result: str, expected: str) -> None:
    problems = result_problems(question, result)
    assert len(problems) == 1 and expected in problems[0]


@pytest.mark.parametrize(
    ("question", "result"),
    [
        # The asked-for item decides, not any word in the question.
        ("Retrieve the name of the genre with the lowest number of tracks.", "'Opera'"),
        ("Find the tables other than Customer that have the 'Email' column.", "Employee"),
        ("Get the full name of the employee with the longest phone number.", "Nancy Edwards"),
        # A track named "40" is a name, and plural questions may return several rows.
        ("Get the name and unit price of the least expensive track.", "[('40', 0.99)]"),
        ("List all the tables in the database.", "[('Album',), ('Artist',)]"),
        ("Get the columns for the 'Album' table.", "[('AlbumId',), ('Title',), ('ArtistId',)]"),
        (
            "List the first name and last name of the employees who report to EmployeeId 6.",
            "[('Robert', 'King'), ('Laura', 'Callahan')]",
        ),
        ("What is the average invoice total per country?", "[('USA', 5.75), ('Canada', 5.43)]"),
        ("Get the number of tracks in each genre.", "[('Rock', 1297), ('Jazz', 130)]"),
    ],
)
def test_correct_results_are_not_flagged(question: str, result: str) -> None:
    assert result_problems(question, result) == []


def test_what_the_model_sees_of_the_tool_is_unchanged() -> None:
    """The tool definition sent to the model, which the question check leaves unchanged."""
    tool = QueryResultCheckerTool(llm=FakeListChatModel(responses=[LLM_REPLY]))
    assert convert_to_openai_tool(tool) == {
        "type": "function",
        "function": {
            "name": "check_result_tool",
            "description": "Use this tool to check the result of your SQL query.",
            "parameters": {
                "properties": {"query_result": {"type": "string"}},
                "required": ["query_result"],
                "type": "object",
            },
        },
    }


class _SqlInput(BaseModel):
    query: str


async def _reply(handle_tool_errors: bool, question: str, result: str) -> ToolMessage:
    """Run the tool through the agent loop's own tool node and state class."""
    tool = QueryResultCheckerTool(llm=FakeListChatModel(responses=[LLM_REPLY]))
    node = create_tool_node([tool], handle_tool_errors=handle_tool_errors)[tool.name]
    call = {"name": tool.name, "args": {"query_result": result}, "id": "call-1"}
    state_class: Any = create_state_with_input(_SqlInput)  # the loop's state: messages + inputs
    state = state_class(
        query=question,
        messages=[HumanMessage(question), AIMessage("", tool_calls=[call])],
    )
    (message,) = (await node.ainvoke(state))["messages"]
    return message


@pytest.mark.parametrize("handle_tool_errors", [False, True])
async def test_a_correct_result_gets_the_checker_reply_unchanged(handle_tool_errors: bool) -> None:
    question = "Find the invoice total for the invoice with InvoiceId 2."
    message = await _reply(handle_tool_errors, question, "[(3.96,)]")
    assert message.content == LLM_REPLY
    assert message.tool_call_id == "call-1"

    # Byte-identical to the reply of the same tool without the wrapper.
    tool = QueryResultCheckerTool(llm=FakeListChatModel(responses=[LLM_REPLY]))
    call = {
        "name": tool.name,
        "args": {"query_result": "[(3.96,)]"},
        "id": "call-1",
        "type": "tool_call",
    }
    assert message.content == (await tool.ainvoke(call)).content


@pytest.mark.parametrize("handle_tool_errors", [False, True])
async def test_a_wrong_result_gets_the_note_after_the_checker_reply(
    handle_tool_errors: bool,
) -> None:
    question = "Find the invoice total for the invoice with InvoiceId 2."
    message = await _reply(handle_tool_errors, question, "Leonie Köhler")
    assert isinstance(message.content, str)
    first, note = message.content.split("\n", 1)
    assert first == LLM_REPLY
    assert note == "Check: the question asks for the invoice total, but the result has no number."


class _FailingChecker(FakeListChatModel):
    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("checker down")


async def test_handle_tool_errors_still_turns_a_checker_failure_into_an_error_reply() -> None:
    tool = QueryResultCheckerTool(llm=_FailingChecker(responses=["unused"]))
    call = {"name": tool.name, "args": {"query_result": "[(3.96,)]"}, "id": "call-1"}
    state_class: Any = create_state_with_input(_SqlInput)
    state = state_class(
        query="Find the invoice total.", messages=[AIMessage("", tool_calls=[call])]
    )

    handled = create_tool_node([tool], handle_tool_errors=True)[tool.name]
    (message,) = (await handled.ainvoke(state))["messages"]
    assert "checker down" in str(message.content)

    unhandled = create_tool_node([tool], handle_tool_errors=False)[tool.name]
    with pytest.raises(RuntimeError, match="checker down"):
        await unhandled.ainvoke(state)
