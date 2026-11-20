"""Custom evaluators for the HTS classifier agent."""

import re

from pydantic import Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import EvaluationResult, NumericEvaluationResult, WorkloadExecution


class HTSCodeMatchCriteria(BaseEvaluationCriteria):
    """The expected 10-digit HTS code, e.g. 4016.99.6010."""

    expected_code: str = Field(description="Expected 10-digit HTS code")


class HTSCodeMatchConfig(BaseEvaluatorConfig[HTSCodeMatchCriteria]):
    """Config for HTSCodeMatchEvaluator."""

    name: str = "HTSCodeMatchEvaluator"
    target_output_key: str = Field(
        default="Top_HTS_code", description="The output field holding the HTS code"
    )


def _digits(code: object) -> str:
    return re.sub(r"\D", "", code) if isinstance(code, str) else ""


class HTSCodeMatchEvaluator(BaseEvaluator[HTSCodeMatchCriteria, HTSCodeMatchConfig, str]):
    """Scores 1 when the output code has the expected digits, whatever the separators.

    4016.99.6010, 4016.99.60.10 and 4016996010 all match 4016.99.6010; an 8-digit code does not.
    """

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "HTSCodeMatchEvaluator"

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: HTSCodeMatchCriteria,
    ) -> EvaluationResult:
        output = workload_execution.workload_output
        actual = (
            output.get(self.evaluator_config.target_output_key)
            if isinstance(output, dict)
            else None
        )
        expected = evaluation_criteria.expected_code
        match = _digits(actual) == _digits(expected) != ""
        return NumericEvaluationResult(
            score=1.0 if match else 0.0,
            details=BaseEvaluatorJustification(expected=expected, actual=str(actual)),
        )
