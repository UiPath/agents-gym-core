"""Custom evaluator for servicenow_work_order_accelerator: normalized exact match on RecommendedAction."""

import re

from pydantic import Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import EvaluationResult, NumericEvaluationResult, WorkloadExecution


class NormalizedMatchCriteria(BaseEvaluationCriteria):
    expected_recommended_action: str = Field(
        alias="expectedRecommendedAction",
        description="Expected RecommendedAction value",
    )


class NormalizedMatchConfig(BaseEvaluatorConfig[NormalizedMatchCriteria]):
    name: str = "NormalizedExactMatchRecommendedActionEvaluator"


class NormalizedExactMatchRecommendedActionEvaluator(
    BaseEvaluator[NormalizedMatchCriteria, NormalizedMatchConfig, str]
):
    """Exact match on RecommendedAction with normalization:
    - Case insensitive
    - Strips trailing periods
    - Normalizes separators (semicolons → commas)
    - Collapses whitespace
    """

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "NormalizedExactMatchRecommendedActionEvaluator"

    @staticmethod
    def _normalize(s: str) -> str:
        s = s.strip().rstrip(".").lower()
        s = s.replace(";", ",")
        s = re.sub(r"\s+", " ", s)
        return s

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: NormalizedMatchCriteria,
    ) -> EvaluationResult:
        agent_output = workload_execution.workload_output
        if not isinstance(agent_output, dict):
            return NumericEvaluationResult(
                score=0.0,
                details=BaseEvaluatorJustification(
                    expected=evaluation_criteria.expected_recommended_action,
                    actual="not a dictionary",
                ),
            )

        actual = str(agent_output.get("RecommendedAction", ""))
        expected = evaluation_criteria.expected_recommended_action

        norm_actual = self._normalize(actual)
        norm_expected = self._normalize(expected)

        score = 1.0 if norm_actual == norm_expected else 0.0
        return NumericEvaluationResult(
            score=score,
            details=BaseEvaluatorJustification(
                expected=norm_expected,
                actual=norm_actual,
            ),
        )
