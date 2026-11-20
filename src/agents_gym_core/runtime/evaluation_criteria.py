"""Prepare criteria passed to the SDK evaluators."""

from typing import Any, TypeGuard

from uipath.eval._helpers.evaluators_helpers import extract_tool_calls
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.tool_call_count_evaluator import ToolCallCountEvaluator
from uipath.eval.models import WorkloadExecution


def is_infrastructure_failure(output: Any) -> TypeGuard[dict[str, Any]]:
    """True for a run that failed outside the agent (GENERAL_EXECUTION_ERROR, e.g. a timeout).

    Such a run scores 0 without an evaluator call. Agent errors such as raise_error() or the
    iteration limit carry their own code and are scored as outputs.
    """
    return (
        isinstance(output, dict)
        and output.get("type") == "error"
        and output.get("code") == "GENERAL_EXECUTION_ERROR"
    )


def prepare_evaluation_criteria(
    evaluator: BaseEvaluator,
    execution: WorkloadExecution,
    criteria: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Use empty expected counts when no tools were called and every expected count is zero."""
    if type(evaluator) is not ToolCallCountEvaluator or not isinstance(criteria, dict):
        return criteria

    key = "toolCallsCount" if "toolCallsCount" in criteria else "tool_calls_count"
    expected = criteria.get(key)
    if not isinstance(expected, dict) or not expected:
        return criteria
    if not all(
        isinstance(count, (list, tuple))
        and len(count) == 2
        and count[0] in ("=", "==")
        and count[1] == 0
        for count in expected.values()
    ):
        return criteria
    if extract_tool_calls(execution.workload_trace, include_args=False):
        return criteria

    return {**criteria, key: {}}
