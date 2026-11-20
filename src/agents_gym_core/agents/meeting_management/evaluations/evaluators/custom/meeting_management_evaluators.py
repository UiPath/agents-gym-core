"""Custom evaluators for meeting_management agent."""

import pandas as pd
from pydantic import Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import EvaluationResult, NumericEvaluationResult, WorkloadExecution

from agents_gym_core.agents.meeting_management.tools import (
    AgentOutputSchema,
    MeetingTimeConstraintsSchema,
)


class MeetingManagementEqualsCriteria(BaseEvaluationCriteria):
    """Criteria for meeting management equality."""

    expected_output: MeetingTimeConstraintsSchema = Field(
        description="Expected meeting time constraints"
    )


class MeetingManagementEqualsConfig(BaseEvaluatorConfig[MeetingManagementEqualsCriteria]):
    """Config for MeetingManagementEqualsEvaluator."""

    name: str = "MeetingManagementEqualsEvaluator"


def _evaluate_meeting_time_and_duration(
    meeting_time: MeetingTimeConstraintsSchema,
    agent_start_time: str | None,
    agent_end_time: str | None,
) -> float:
    if meeting_time.NoMeeting:
        return 1.0 if not agent_start_time and not agent_end_time else 0.0

    try:
        agent_start_time_dt = (
            pd.to_datetime(agent_start_time) if agent_start_time is not None else None
        )
    except ValueError:
        return 0.0

    try:
        agent_end_time_dt = pd.to_datetime(agent_end_time) if agent_end_time is not None else None
    except ValueError:
        return 0.0

    if meeting_time.exact_start_time and agent_start_time_dt != meeting_time.exact_start_time:
        return 0.0

    if meeting_time.exact_end_time and agent_end_time_dt != meeting_time.exact_end_time:
        return 0.0

    if meeting_time.ExactDuration:
        if agent_start_time_dt and agent_end_time_dt:
            duration = (agent_end_time_dt - agent_start_time_dt).total_seconds() / 60
            if duration != meeting_time.ExactDuration:
                return 0.0
        else:
            return 0.0

    if meeting_time.AllowedIntervalTime:
        if agent_start_time_dt and agent_end_time_dt:
            allowed_start_time = pd.to_datetime(meeting_time.AllowedIntervalTime[0])
            allowed_end_time = pd.to_datetime(meeting_time.AllowedIntervalTime[1])

            if agent_start_time_dt < allowed_start_time or agent_end_time_dt > allowed_end_time:
                return 0.0
        else:
            return 0.0

    return 1.0


class MeetingManagementEqualsEvaluator(
    BaseEvaluator[MeetingManagementEqualsCriteria, MeetingManagementEqualsConfig, str]
):
    """Deterministic evaluator that validates meeting time constraints."""

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: MeetingManagementEqualsCriteria,
    ) -> EvaluationResult:
        expected = evaluation_criteria.expected_output

        agent_output = workload_execution.workload_output
        if not isinstance(agent_output, dict):
            return NumericEvaluationResult(
                score=0.0,
                details=BaseEvaluatorJustification(
                    expected=str(expected.model_dump()),
                    actual="not a dictionary",
                ),
            )

        if agent_output.get("type") == "error" and not (
            expected.NoMeeting
            and agent_output.get("code") == "AGENT_RUNTIME.TERMINATION_LLM_RAISED_ERROR"
        ):
            return NumericEvaluationResult(
                score=0.0,
                details=BaseEvaluatorJustification(
                    expected=str(expected.model_dump()),
                    actual=str(agent_output),
                ),
            )

        parsed_output = AgentOutputSchema(**agent_output)
        if parsed_output.StartTime and parsed_output.EndTime:
            actual = MeetingTimeConstraintsSchema(
                ExactStartTime=parsed_output.StartTime,
                ExactEndTime=parsed_output.EndTime,
                NoMeeting=False,
            )
        else:
            actual = MeetingTimeConstraintsSchema(NoMeeting=True)

        score = _evaluate_meeting_time_and_duration(
            meeting_time=expected,
            agent_start_time=actual.ExactStartTime,
            agent_end_time=actual.ExactEndTime,
        )

        return NumericEvaluationResult(
            score=score,
            details=BaseEvaluatorJustification(
                expected=str(expected.model_dump()),
                actual=str(actual.model_dump()),
            ),
        )

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "MeetingManagementEqualsEvaluator"
