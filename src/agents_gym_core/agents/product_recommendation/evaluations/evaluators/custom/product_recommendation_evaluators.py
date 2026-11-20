"""Custom evaluators for Product Recommendation agent."""

import json

from pydantic import Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import EvaluationResult, NumericEvaluationResult, WorkloadExecution


class ProductMatchCriteria(BaseEvaluationCriteria):
    """Criteria containing the expected products list."""

    expected_products: list[dict] = Field(
        description="List of expected product dicts with Name, Brand, Price, Features, and Match fields"
    )


class ProductMatchConfig(BaseEvaluatorConfig[ProductMatchCriteria]):
    """Config for ProductMatch evaluator."""

    name: str = "ProductMatchEvaluator"
    target_output_key: str = Field(
        default="Products",
        description="The key in the agent output containing the products list",
    )


class ProductMatchJustification(BaseEvaluatorJustification):
    """Justification for ProductMatchEvaluator.

    expected: expected product names (stringified list)
    actual: matched product names (stringified list)
    unmatched: expected product names that were not found in the output
    """

    unmatched: list[str]


def _products_equal(actual: dict, expected: dict) -> bool:
    """Compare two products by Name, Brand, Price, Features, and Match (ignoring Reason)."""
    return (
        actual.get("Name") == expected.get("Name")
        and actual.get("Brand") == expected.get("Brand")
        and actual.get("Price") == expected.get("Price")
        and actual.get("Features") == expected.get("Features")
        and actual.get("Match") == expected.get("Match")
    )


class ProductMatchEvaluator(BaseEvaluator[ProductMatchCriteria, ProductMatchConfig, str]):
    """Evaluator that checks what fraction of expected products appear in the agent output.

    For each expected product, checks if there is a matching product in the actual output
    (comparing Name, Brand, Price, Features, and Match fields, ignoring Reason).
    Returns score = matched_count / total_expected.
    """

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "ProductMatchEvaluator"

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: ProductMatchCriteria,
    ) -> EvaluationResult:
        agent_output = workload_execution.workload_output
        expected_products = evaluation_criteria.expected_products
        expected_names = [p.get("Name", "?") for p in expected_products]

        if not isinstance(agent_output, dict):
            return NumericEvaluationResult(
                score=0.0,
                details=ProductMatchJustification(
                    expected=str(expected_names),
                    actual="not a dictionary",
                    unmatched=expected_names,
                ),
            )

        target_key = self.evaluator_config.target_output_key
        actual_products_raw = agent_output.get(target_key)

        if actual_products_raw is None:
            return NumericEvaluationResult(
                score=0.0,
                details=ProductMatchJustification(
                    expected=str(expected_names),
                    actual=f"key '{target_key}' not found",
                    unmatched=expected_names,
                ),
            )

        # Parse products if they are a JSON string
        if isinstance(actual_products_raw, str):
            try:
                actual_products = json.loads(actual_products_raw)
            except json.JSONDecodeError:
                return NumericEvaluationResult(
                    score=0.0,
                    details=ProductMatchJustification(
                        expected=str(expected_names),
                        actual="could not parse Products field as JSON",
                        unmatched=expected_names,
                    ),
                )
        elif isinstance(actual_products_raw, list):
            actual_products = actual_products_raw
        else:
            return NumericEvaluationResult(
                score=0.0,
                details=ProductMatchJustification(
                    expected=str(expected_names),
                    actual=f"Products field is not a list or JSON string, got {type(actual_products_raw).__name__}",
                    unmatched=expected_names,
                ),
            )

        if len(expected_products) == 0 and len(actual_products) == 0:
            return NumericEvaluationResult(
                score=1.0,
                details=BaseEvaluatorJustification(
                    expected="(empty)",
                    actual="(empty)",
                ),
            )

        if len(expected_products) == 0:
            return NumericEvaluationResult(
                score=0.0,
                details=BaseEvaluatorJustification(
                    expected="(empty)",
                    actual=str([p.get("Name") for p in actual_products]),
                ),
            )

        matched_names = []
        unmatched = []
        for expected in expected_products:
            found = any(_products_equal(actual, expected) for actual in actual_products)
            if found:
                matched_names.append(expected.get("Name", "?"))
            else:
                unmatched.append(expected.get("Name", "?"))

        score = len(matched_names) / len(expected_products)
        return NumericEvaluationResult(
            score=score,
            details=ProductMatchJustification(
                expected=str(expected_names),
                actual=str(matched_names),
                unmatched=unmatched,
            ),
        )
