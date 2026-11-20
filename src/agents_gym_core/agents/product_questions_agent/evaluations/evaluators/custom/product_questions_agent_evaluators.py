"""Custom evaluators for Product Questions agent."""

import json

from pydantic import Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import EvaluationResult, NumericEvaluationResult, WorkloadExecution


class EqualsEscalationCriteria(BaseEvaluationCriteria):
    """Criteria containing the expected escalation question text."""

    expected_question: str = Field(
        description="The expected question text in QuestionsToBeEscalated[0].Question"
    )


class EqualsEscalationConfig(BaseEvaluatorConfig[EqualsEscalationCriteria]):
    """Config for EqualsEscalation evaluator."""

    name: str = "EqualsEscalationEvaluator"


class EqualsEscalationEvaluator(
    BaseEvaluator[EqualsEscalationCriteria, EqualsEscalationConfig, str]
):
    """Evaluator that checks if QuestionsToBeEscalated[0].Question matches the expected text.

    Extracts the first question from the agent output's QuestionsToBeEscalated array
    and compares it (case-insensitive) with the expected question text.
    """

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "EqualsEscalationEvaluator"

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: EqualsEscalationCriteria,
    ) -> EvaluationResult:
        agent_output = workload_execution.workload_output
        if not isinstance(agent_output, dict):
            return NumericEvaluationResult(
                score=0.0,
                details=BaseEvaluatorJustification(
                    expected=evaluation_criteria.expected_question,
                    actual="not a dictionary",
                ),
            )

        questions = agent_output.get("QuestionsToBeEscalated")

        # Parse if it's a JSON string
        if isinstance(questions, str):
            try:
                questions = json.loads(questions)
            except json.JSONDecodeError:
                questions = None

        # Extract first question
        actual_question = ""
        if isinstance(questions, list) and len(questions) > 0:
            first = questions[0]
            if isinstance(first, dict):
                actual_question = first.get("Question", "")
            elif isinstance(first, str):
                actual_question = first

        expected_question = evaluation_criteria.expected_question
        is_match = actual_question.lower() == expected_question.lower()

        return NumericEvaluationResult(
            score=float(is_match),
            details=BaseEvaluatorJustification(
                expected=expected_question,
                actual=actual_question,
            ),
        )
