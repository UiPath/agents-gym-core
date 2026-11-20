"""Custom deterministic evaluators for athena_chatbot agent."""

from pydantic import Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import EvaluationResult, NumericEvaluationResult, WorkloadExecution


class ExpectedMentorCheckCriteria(BaseEvaluationCriteria):
    """Criteria for expected mentor name check."""

    expected_output: dict[str, str] | str | None = Field(default=None)


class ExpectedMentorCheckConfig(BaseEvaluatorConfig[ExpectedMentorCheckCriteria]):
    """Config for ExpectedMentorCheckEvaluator."""

    name: str = "ExpectedMentorCheckEvaluator"


class ExpectedMentorCheckJustification(BaseEvaluatorJustification):
    """Justification for ExpectedMentorCheckEvaluator.

    expected: comma-separated mentor names that should appear in the reply
    actual: the agent's reply text
    """

    found_count: int
    total_count: int


class ExpectedMentorCheckEvaluator(
    BaseEvaluator[ExpectedMentorCheckCriteria, ExpectedMentorCheckConfig, str]
):
    """Deterministically checks expected mentor names in agent reply."""

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: ExpectedMentorCheckCriteria,
    ) -> EvaluationResult:
        agent_output = workload_execution.workload_output
        if not isinstance(agent_output, dict):
            return NumericEvaluationResult(
                score=0.0,
                details=ExpectedMentorCheckJustification(
                    expected="", actual="", found_count=0, total_count=0
                ),
            )

        expected_output = evaluation_criteria.expected_output
        if isinstance(expected_output, dict):
            expected_names = expected_output.get("out_agentReply", "")
        elif isinstance(expected_output, str):
            expected_names = expected_output
        else:
            expected_names = ""

        actual_reply = agent_output.get("out_agentReply", "")
        if not isinstance(actual_reply, str):
            return NumericEvaluationResult(
                score=0.0,
                details=ExpectedMentorCheckJustification(
                    expected=expected_names,
                    actual=str(actual_reply),
                    found_count=0,
                    total_count=0,
                ),
            )

        mentors = [name.strip() for name in expected_names.split(",") if name.strip()]
        found = sum(1 for name in mentors if name in actual_reply) if mentors else 0
        score = found / len(mentors) if mentors else 1.0

        return NumericEvaluationResult(
            score=score,
            details=ExpectedMentorCheckJustification(
                expected=expected_names,
                actual=actual_reply,
                found_count=found,
                total_count=len(mentors),
            ),
        )

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "ExpectedMentorCheckEvaluator"
