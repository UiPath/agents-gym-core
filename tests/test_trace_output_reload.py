"""Reloading a saved trace returns the recorded agent output, JSON-looking answers included."""

from types import SimpleNamespace
from typing import cast

from opentelemetry.sdk.trace import ReadableSpan

from agents_gym_core.utils.trace_utils import get_agent_output_from_trace


def _chain(name: str, value: object) -> ReadableSpan:
    attributes = {"openinference.span.kind": "CHAIN", "output.value": value}
    return cast(ReadableSpan, SimpleNamespace(name=name, attributes=attributes))


def test_a_json_object_is_returned_as_a_dict() -> None:
    trace = [_chain("LangGraph", '{"answer": "Paris", "reasoning": "capital"}')]
    assert get_agent_output_from_trace(trace) == {"answer": "Paris", "reasoning": "capital"}


def test_plain_text_is_returned_stripped() -> None:
    assert get_agent_output_from_trace([_chain("LangGraph", "  Paris \n")]) == "Paris"


def test_an_answer_that_parses_as_a_json_scalar_is_kept_as_text() -> None:
    """A single-string output such as an AIME or HotpotQA answer is recorded as the bare
    string; ``"17"``, ``"true"`` or ``"1998"`` must not vanish because they parse as JSON."""
    for recorded in ("17", "1998", "true", "3.5", "[1, 2]"):
        assert get_agent_output_from_trace([_chain("terminate", recorded)]) == recorded


def test_the_last_chain_span_with_a_value_wins() -> None:
    trace = [
        _chain("LangGraph", '{"answer": "old"}'),
        _chain("LangGraph", "17"),
        cast(ReadableSpan, SimpleNamespace(name="LangGraph", attributes={"other": 1})),
    ]
    assert get_agent_output_from_trace(trace) == "17"


def test_empty_values_are_skipped_and_no_output_is_an_empty_dict() -> None:
    assert get_agent_output_from_trace([_chain("LangGraph", "   ")]) == {}
    assert get_agent_output_from_trace([]) == {}
