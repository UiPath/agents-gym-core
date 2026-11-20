"""Pass empty expected counts to the SDK only for satisfied zero-call criteria."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import dspy
import pytest
from langchain_core.language_models import BaseChatModel
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import NoOpTracerProvider
from uipath.eval.evaluators.tool_call_count_evaluator import ToolCallCountEvaluator
from uipath.eval.models import WorkloadExecution

from agents_gym_core.hydra_agent import CoreAgentRegistry, EvaluationLoader
from agents_gym_core.runtime.evaluation_criteria import prepare_evaluation_criteria
from agents_gym_core.runtime.run_utils import AgentDatapointExecution
from agents_gym_core.runtime.runtime_config import RuntimeConfig
from agents_gym_core.runtime.runtime_helpers import RuntimeHelpers

COUNT_ID = "meeting_management_ToolCallCountEvaluator"
SCHEDULE = "AFE_ScheduleZoomWrapper"


def _execution(tool: str | None = None) -> WorkloadExecution:
    spans = [ReadableSpan(name="model", attributes={"openinference.span.kind": "LLM"})]
    if tool:
        spans.append(ReadableSpan(name=tool, attributes={"tool.name": tool}))
    return WorkloadExecution(agent_input={}, workload_output={}, workload_trace=spans)


@pytest.fixture
def evaluator() -> ToolCallCountEvaluator:
    registry = CoreAgentRegistry()
    result = EvaluationLoader.get_evaluators(
        "meeting_management", registry.get_directory_provider()
    )[COUNT_ID]
    assert type(result) is ToolCallCountEvaluator
    return result


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    ("counts", "tool", "score", "becomes_empty"),
    [
        ({SCHEDULE: ["=", 0]}, None, 1.0, True),
        ({SCHEDULE: ["=", 0]}, "GetOutlookEvents", 1.0, False),
        ({SCHEDULE: ["=", 0]}, SCHEDULE, 0.0, False),
        ({SCHEDULE: ["=", 1]}, None, 0.0, False),
        ({SCHEDULE: ["=", 0], "lookup": ["=", 1]}, None, 0.0, False),
        ({}, None, 1.0, False),
        ({}, "GetOutlookEvents", 0.0, False),
        ({SCHEDULE: ["==", 0], "lookup": ["=", 0]}, None, 1.0, True),
    ],
)
async def test_expected_empty_substitution(
    evaluator: ToolCallCountEvaluator,
    strict: bool,
    counts: dict[str, Any],
    tool: str | None,
    score: float,
    becomes_empty: bool,
) -> None:
    evaluator.evaluator_config.strict = strict
    criteria = {"toolCallsCount": counts}
    original = copy.deepcopy(criteria)
    execution = _execution(tool)
    prepared = prepare_evaluation_criteria(evaluator, execution, criteria)

    assert prepared == {"toolCallsCount": {}} if becomes_empty else prepared == original
    assert criteria == original
    result = await evaluator.validate_and_evaluate_criteria(execution, prepared)
    assert result.score == score


def _weekend_case():
    agent = CoreAgentRegistry().get_agent("meeting_management", "train")
    item = next(dp for dp in agent.evaluation_set.evaluations if dp.name == "Weekend Meeting Test")
    return agent, item


@pytest.mark.parametrize("tool", [None, "GetOutlookEvents"])
async def test_runtime_saves_effective_expected_counts(
    evaluator: ToolCallCountEvaluator, tmp_path: Path, tool: str | None
) -> None:
    agent, item = _weekend_case()
    execution = AgentDatapointExecution(datapoint=item, workload_execution=_execution(tool))
    result = await RuntimeHelpers(RuntimeConfig()).run_and_save_evaluator(
        agent, evaluator, execution, tmp_path
    )
    assert result[3].score == 1.0
    saved = json.loads(next(tmp_path.rglob(f"{COUNT_ID}.json")).read_text())
    assert saved["evaluation_criteria"] == {
        "toolCallsCount": {} if tool is None else {SCHEDULE: ["=", 0]}
    }
    assert item.evaluation_criterias[COUNT_ID] == {"toolCallsCount": {SCHEDULE: ["=", 0]}}


@pytest.mark.parametrize("tool", [None, "GetOutlookEvents"])
async def test_native_gepa_uses_expected_empty(
    evaluator: ToolCallCountEvaluator, monkeypatch: pytest.MonkeyPatch, tool: str | None
) -> None:
    from agents_gym_core.prompt_optimization.gepa import rollout

    agent, item = _weekend_case()
    execution = _execution(tool)
    monkeypatch.setattr(
        rollout, "get_loop", lambda **_: SimpleNamespace(ainvoke=AsyncMock(return_value={}))
    )
    monkeypatch.setattr(
        rollout,
        "_ensure_tracing",
        lambda: {
            "span_exporter": SimpleNamespace(pop_trace=lambda _: execution.workload_trace),
            "tracer_provider": NoOpTracerProvider(),
        },
    )
    result = await rollout.rollout_one(
        {}, agent, item, cast(BaseChatModel, object()), {COUNT_ID: evaluator}, {}
    )
    assert result.aggregated_score == 1.0


@pytest.mark.parametrize("tool", [None, "GetOutlookEvents"])
def test_dspy_gepa_uses_expected_empty(evaluator: ToolCallCountEvaluator, tool: str | None) -> None:
    from agents_gym_core.prompt_optimization.gepa_dspy.metric import make_gym_metric

    _, item = _weekend_case()
    metric = make_gym_metric(evaluators={COUNT_ID: evaluator}, evaluator_weights={})
    result = metric(
        dspy.Example(_eval_item=item),
        dspy.Prediction(agent_output={}, agent_spans=_execution(tool).workload_trace),
    )
    assert result.score == 1.0
