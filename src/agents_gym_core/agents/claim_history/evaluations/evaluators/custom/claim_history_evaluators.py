"""Custom deterministic evaluators for claim_history agent."""

from typing import ClassVar

from pydantic import Field, ValidationError
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import EvaluationResult, NumericEvaluationResult, WorkloadExecution

from agents_gym_core.agents.claim_history.tools import ClaimHistoryLookupOutput


class ClaimHistoryLookupCriteria(BaseEvaluationCriteria):
    """Criteria placeholder for compatibility with evaluation sets."""

    expected_output: str | None = Field(default=None)


class ClaimHistoryLookupConfig(BaseEvaluatorConfig[ClaimHistoryLookupCriteria]):
    """Config for deterministic claim history lookup evaluators."""

    name: str = "ClaimHistoryLookupEvaluator"


class _ClaimHistoryLookupBaseEvaluator(
    BaseEvaluator[ClaimHistoryLookupCriteria, ClaimHistoryLookupConfig, str]
):
    """Shared logic for deterministic claim history lookup evaluators."""

    expected_error: ClassVar[bool] = False

    def _compute_score(self, output: ClaimHistoryLookupOutput) -> float:
        has_error = len(output.ErrorMessage) > 0
        has_tips = len(output.AppealTips) > 0
        if self.expected_error:
            return float(has_error and not has_tips)
        return float((not has_error) and has_tips)

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: ClaimHistoryLookupCriteria,
    ) -> EvaluationResult:
        agent_output = workload_execution.workload_output
        if not isinstance(agent_output, dict):
            return NumericEvaluationResult(
                score=0.0,
                details=BaseEvaluatorJustification(
                    expected="error" if self.expected_error else "success",
                    actual="not a dictionary",
                ),
            )

        try:
            parsed_output = ClaimHistoryLookupOutput(**agent_output)
        except ValidationError as exc:
            return NumericEvaluationResult(
                score=0.0,
                details=BaseEvaluatorJustification(
                    expected="error" if self.expected_error else "success",
                    actual=f"validation error: {exc}",
                ),
            )

        score = self._compute_score(parsed_output)
        return NumericEvaluationResult(
            score=score,
            details=BaseEvaluatorJustification(
                expected="error" if self.expected_error else "success",
                actual=str(parsed_output.model_dump()),
            ),
        )


class ClaimHistoryLookupSuccessEvaluator(_ClaimHistoryLookupBaseEvaluator):
    """Success evaluator: ErrorMessage empty and AppealTips non-empty."""

    expected_error: ClassVar[bool] = False

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "ClaimHistoryLookupSuccessEvaluator"


class ClaimHistoryLookupErrorEvaluator(_ClaimHistoryLookupBaseEvaluator):
    """Error evaluator: ErrorMessage non-empty and AppealTips empty."""

    expected_error: ClassVar[bool] = True

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "ClaimHistoryLookupErrorEvaluator"
