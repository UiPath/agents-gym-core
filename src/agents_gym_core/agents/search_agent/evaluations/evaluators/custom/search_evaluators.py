"""Custom evaluator for search_agent: checks that output contains expected keywords."""

from pydantic import Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import EvaluationResult, NumericEvaluationResult, WorkloadExecution


class SearchContainsCriteria(BaseEvaluationCriteria):
    """Expected keywords that should appear in the search result."""

    keywords: list[str | list[str]] = Field(
        description="Keywords that should appear in the result; a list item holds alternative spellings"
    )


class SearchContainsConfig(BaseEvaluatorConfig[SearchContainsCriteria]):
    name: str = "SearchContainsEvaluator"


class SearchContainsJustification(BaseEvaluatorJustification):
    """Justification for SearchContainsEvaluator.

    expected: keywords that should appear (stringified list)
    actual: keywords that were found (stringified list)
    missing: keywords not found in the result
    found_count: number of keywords found
    total_count: total number of expected keywords
    """

    missing: list[str]
    found_count: int
    total_count: int


class SearchContainsEvaluator(BaseEvaluator[SearchContainsCriteria, SearchContainsConfig, str]):
    """Checks what fraction of expected keywords appear in the agent's result field.

    Score = (number of keywords found) / (total expected keywords).
    Case-insensitive matching.
    """

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "SearchContainsEvaluator"

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: SearchContainsCriteria,
    ) -> EvaluationResult:
        agent_output = workload_execution.workload_output
        if not isinstance(agent_output, dict):
            return NumericEvaluationResult(
                score=0.0,
                details=SearchContainsJustification(
                    expected=str(evaluation_criteria.keywords),
                    actual="not a dictionary",
                    missing=[str(kw) for kw in evaluation_criteria.keywords],
                    found_count=0,
                    total_count=len(evaluation_criteria.keywords),
                ),
            )

        result = str(agent_output.get("result", ""))
        keywords = evaluation_criteria.keywords
        if not keywords:
            return NumericEvaluationResult(
                score=1.0,
                details=SearchContainsJustification(
                    expected="(no keywords)",
                    actual="(no keywords)",
                    missing=[],
                    found_count=0,
                    total_count=0,
                ),
            )

        result_lower = result.lower()

        def found(kw: str | list[str]) -> bool:
            return any(k.lower() in result_lower for k in ([kw] if isinstance(kw, str) else kw))

        missing = [str(kw) for kw in keywords if not found(kw)]
        found_keywords = [str(kw) for kw in keywords if found(kw)]
        score = len(found_keywords) / len(keywords)

        return NumericEvaluationResult(
            score=score,
            details=SearchContainsJustification(
                expected=str(keywords),
                actual=str(found_keywords),
                missing=missing,
                found_count=len(found_keywords),
                total_count=len(keywords),
            ),
        )
