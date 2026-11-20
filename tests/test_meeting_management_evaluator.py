"""Meeting-time scoring distinguishes intentional refusals from execution failures."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from langchain_core.language_models import BaseChatModel
from opentelemetry.trace import NoOpTracerProvider
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.evaluator_factory import EvaluatorFactory
from uipath.eval.models import WorkloadExecution
from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode

from agents_gym_core.hydra_agent import CoreAgentRegistry

AGENT_DIR = Path(__file__).resolve().parents[1] / "src/agents_gym_core/agents/meeting_management"


@pytest.fixture
def evaluator() -> BaseEvaluator:
    directory = AGENT_DIR / "evaluations/evaluators"
    spec = json.loads(
        (
            directory / "evaluator-meeting_management_MeetingManagementEqualsEvaluator.json"
        ).read_text()
    )
    result = EvaluatorFactory.create_evaluator(spec, evaluators_dir=directory)
    assert isinstance(result, BaseEvaluator)
    return result


async def _score(evaluator: BaseEvaluator, output: Any, expected: dict[str, Any]) -> float:
    result = await evaluator.validate_and_evaluate_criteria(
        workload_execution=WorkloadExecution(
            agent_input={}, workload_output=output, workload_trace=[], simulation_instructions=""
        ),
        evaluation_criteria={"expectedOutput": expected},
    )
    return float(result.score)


@pytest.mark.parametrize(
    "output",
    [
        {"type": "error", "code": "AGENT_RUNTIME.TERMINATION_MAX_ITERATIONS"},
        {"type": "error", "title": "RolloutTimeout", "details": "exceeded 240s"},
        {"type": "error", "code": "GENERAL_EXECUTION_ERROR"},
        {"type": "error", "title": "RuntimeError"},
    ],
)
async def test_failed_execution_does_not_satisfy_no_meeting(
    evaluator: BaseEvaluator, output: dict[str, Any]
) -> None:
    assert await _score(evaluator, output, {"NoMeeting": True}) == 0.0


async def test_valid_no_meeting_and_scheduled_outputs_keep_their_scores(
    evaluator: BaseEvaluator,
) -> None:
    assert (
        await _score(
            evaluator,
            {"StartTime": "", "EndTime": "", "Observations": "The calendar has a conflict."},
            {"NoMeeting": True},
        )
        == 1.0
    )
    meeting = {"StartTime": "2025-02-04 10:00:00", "EndTime": "2025-02-04 11:00:00"}
    assert (
        await _score(
            evaluator,
            meeting,
            {"ExactStartTime": meeting["StartTime"], "ExactEndTime": meeting["EndTime"]},
        )
        == 1.0
    )
    assert await _score(evaluator, meeting, {"NoMeeting": True}) == 0.0


@pytest.mark.parametrize("runner", ["native", "dspy"])
@pytest.mark.parametrize(
    ("code", "expected_score"),
    [
        (AgentRuntimeErrorCode.TERMINATION_LLM_RAISED_ERROR, 1.0),
        (AgentRuntimeErrorCode.TERMINATION_MAX_ITERATIONS, 0.0),
    ],
)
async def test_rollout_preserves_refusal_semantics(
    evaluator: BaseEvaluator,
    monkeypatch: pytest.MonkeyPatch,
    runner: str,
    code: AgentRuntimeErrorCode,
    expected_score: float,
) -> None:
    from agents_gym_core.prompt_optimization.gepa import rollout
    from agents_gym_core.prompt_optimization.gepa_dspy import module

    agent = CoreAgentRegistry().get_agent("meeting_management", "train")
    datapoint = next(
        dp
        for dp in agent.evaluation_set.evaluations
        if isinstance(criteria := dp.evaluation_criterias.get(evaluator.id), dict)
        and criteria.get("expectedOutput", {}).get("NoMeeting")
    )
    error = AgentRuntimeError(
        code=code, title="Execution stopped", detail="No meeting output", should_wrap=False
    )
    graph = SimpleNamespace(ainvoke=AsyncMock(side_effect=error))
    target = rollout if runner == "native" else module
    monkeypatch.setattr(target, "get_loop", lambda **_: graph)
    monkeypatch.setattr(
        target,
        "_ensure_tracing",
        lambda: {
            "span_exporter": SimpleNamespace(pop_trace=lambda _: []),
            "tracer_provider": NoOpTracerProvider(),
        },
    )
    model = cast(BaseChatModel, object())
    if runner == "native":
        result = await rollout.rollout_one(
            {}, agent, datapoint, model, {evaluator.id: evaluator}, {}
        )
        assert result.aggregated_score == expected_score
    else:
        output, _ = await module.GymAgentModule(agent, model=model)._ainvoke(datapoint.inputs)
        assert await _score(evaluator, output, {"NoMeeting": True}) == expected_score


async def test_refusal_does_not_satisfy_a_required_meeting(evaluator: BaseEvaluator) -> None:
    assert (
        await _score(
            evaluator,
            {"type": "error", "code": "AGENT_RUNTIME.TERMINATION_LLM_RAISED_ERROR"},
            {"ExactStartTime": "2025-02-04 10:00:00"},
        )
        == 0.0
    )
