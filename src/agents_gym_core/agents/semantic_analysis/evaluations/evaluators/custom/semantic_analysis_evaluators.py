"""Custom deterministic evaluator for semantic_analysis agent.

Computes SemanticAnalysisScore as a weighted composite of
key-phrase matching (30%), overall sentiment accuracy (30%), and sentiment
breakdown accuracy (40%).
"""

import json
from typing import Any

from pydantic import Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import (
    BaseEvaluationCriteria,
    BaseEvaluatorConfig,
    BaseEvaluatorJustification,
)
from uipath.eval.models import EvaluationResult, NumericEvaluationResult, WorkloadExecution


class SemanticAnalysisScoreCriteria(BaseEvaluationCriteria):
    """Expected output fields for the semantic analysis score evaluator."""

    overall_sentiment_score: int = Field(description="Expected overall sentiment score (-99 to 99)")
    overall_sentiment_label: str = Field(description="Expected overall sentiment label")
    positive_statements: int = Field(description="Expected count of positive statements")
    negative_statements: int = Field(description="Expected count of negative statements")
    neutral_statements: int = Field(description="Expected count of neutral statements")
    total_statements: int = Field(description="Expected total statement count")
    key_phrases: str = Field(description="JSON string of expected key phrases list")


class SemanticAnalysisScoreConfig(BaseEvaluatorConfig[SemanticAnalysisScoreCriteria]):
    """Config for the semantic analysis score evaluator."""

    name: str = "SemanticAnalysisScoreEvaluator"


class SemanticAnalysisScoreJustification(BaseEvaluatorJustification):
    """Justification for SemanticAnalysisScoreEvaluator.

    expected: expected overall sentiment label
    actual: actual overall sentiment label
    key_phrases: "X/Y" points (weight 0.3)
    overall_sentiment: "X/Y" points (weight 0.3)
    sentiment_breakdown: "X/Y" points (weight 0.4)
    """

    key_phrases: str
    overall_sentiment: str
    sentiment_breakdown: str


class SemanticAnalysisScoreEvaluator(
    BaseEvaluator[SemanticAnalysisScoreCriteria, SemanticAnalysisScoreConfig, str]
):
    """Deterministic evaluator that scores semantic analysis output using a
    weighted composite of key-phrase matching, overall sentiment, and
    sentiment breakdown accuracy.

    Scoring weights: keyPhrases 30%, overallSentiment 30%, sentimentBreakdown 40%.
    """

    @classmethod
    def get_evaluator_id(cls) -> str:
        return "SemanticAnalysisScoreEvaluator"

    def _parse_text_analysis(self, agent_output: dict[str, Any]) -> dict[str, Any] | None:
        """Extract text_analysis from agent output.

        Handles two formats:
        1. Nested: {"text_analysis": {"overallSentiment": {...}, "keyPhrases": [...], ...}}
        2. Flat: {"overallSentimentScore": ..., "overallSentimentLabel": ..., "keyPhrases": "...", ...}
        """
        ta = agent_output.get("text_analysis")
        if ta is not None:
            if isinstance(ta, str):
                ta = json.loads(ta)
            return ta

        # Flat format: reconstruct the nested structure from top-level keys
        if "overallSentimentLabel" in agent_output or "overallSentimentScore" in agent_output:
            kp = agent_output.get("keyPhrases", "[]")
            if isinstance(kp, str):
                try:
                    kp = json.loads(kp)
                except (json.JSONDecodeError, TypeError):
                    kp = []
            return {
                "overallSentiment": {
                    "score": agent_output.get("overallSentimentScore", 0),
                    "label": agent_output.get("overallSentimentLabel", ""),
                },
                "sentimentBreakdown": {
                    "positiveStatements": agent_output.get("positiveStatements", 0),
                    "negativeStatements": agent_output.get("negativeStatements", 0),
                    "neutralStatements": agent_output.get("neutralStatements", 0),
                    "totalStatements": agent_output.get("totalStatements", 0),
                },
                "keyPhrases": kp,
            }

        return None

    def _score_keyphrases(
        self, expected_kps: list[dict], actual_kps: list[dict]
    ) -> tuple[float, float]:
        """Score key phrases: up to 3 points per expected phrase (match + sentiment + confidence).

        Each expected phrase is matched by at most one actual phrase and each actual phrase is
        used once, so repeating phrases cannot score more than the maximum.
        """
        max_points = 3 * len(expected_kps)
        if max_points == 0:
            return 0.0, 0.0

        unused = [kp for kp in actual_kps if isinstance(kp, dict) and kp.get("phrase")]
        points = 0
        for exp_kp in expected_kps:
            exp_phrase = exp_kp.get("phrase", "")
            for act_kp in unused:
                act_phrase = act_kp.get("phrase", "")
                if act_phrase in exp_phrase or exp_phrase in act_phrase:
                    unused.remove(act_kp)
                    points += 1
                    if (
                        str(exp_kp.get("sentiment", "")).lower()
                        == str(act_kp.get("sentiment", "")).lower()
                    ):
                        points += 1
                        if exp_kp.get("confidence", 0) <= act_kp.get("confidence", 0):
                            points += 1
                    break
        return float(points), float(max_points)

    async def evaluate(
        self,
        workload_execution: WorkloadExecution,
        evaluation_criteria: SemanticAnalysisScoreCriteria,
    ) -> EvaluationResult:
        agent_output = workload_execution.workload_output
        if not isinstance(agent_output, dict):
            return NumericEvaluationResult(
                score=0.0,
                details=BaseEvaluatorJustification(
                    expected="valid agent output dict",
                    actual="not a dictionary",
                ),
            )

        ta = self._parse_text_analysis(agent_output)
        if ta is None:
            return NumericEvaluationResult(
                score=0.0,
                details=BaseEvaluatorJustification(
                    expected="text_analysis key present in output",
                    actual="missing text_analysis in output",
                ),
            )

        # Parse expected key phrases
        try:
            expected_kps = json.loads(evaluation_criteria.key_phrases)
        except (json.JSONDecodeError, TypeError):
            expected_kps = []

        actual_kps = ta.get("keyPhrases", [])

        # Key phrases score (30%)
        kp_points, kp_max = self._score_keyphrases(expected_kps, actual_kps)
        kp_score = kp_points / kp_max if kp_max > 0 else 0.0

        # Overall sentiment score (30%)
        overall_points = 0
        overall_max = 2
        actual_sentiment: dict = ta.get("overallSentiment", {})
        if (
            str(actual_sentiment.get("label", "")).lower()
            == evaluation_criteria.overall_sentiment_label.lower()
        ):
            overall_points += 1
        actual_score = actual_sentiment.get("score", 0)
        if (
            evaluation_criteria.overall_sentiment_score - 5
            <= actual_score
            <= evaluation_criteria.overall_sentiment_score + 5
        ):
            overall_points += 1
        overall_score = overall_points / overall_max

        # Sentiment breakdown score (40%)
        breakdown_points = 0
        breakdown_max = 4
        actual_breakdown: dict = ta.get("sentimentBreakdown", {})
        if actual_breakdown.get("positiveStatements") == evaluation_criteria.positive_statements:
            breakdown_points += 1
        if actual_breakdown.get("negativeStatements") == evaluation_criteria.negative_statements:
            breakdown_points += 1
        if actual_breakdown.get("neutralStatements") == evaluation_criteria.neutral_statements:
            breakdown_points += 1
        if actual_breakdown.get("totalStatements") == evaluation_criteria.total_statements:
            breakdown_points += 1
        breakdown_score = breakdown_points / breakdown_max

        final_score = kp_score * 0.3 + overall_score * 0.3 + breakdown_score * 0.4

        return NumericEvaluationResult(
            score=final_score,
            details=SemanticAnalysisScoreJustification(
                expected=evaluation_criteria.overall_sentiment_label,
                actual=str(actual_sentiment.get("label", "")),
                key_phrases=f"{kp_points}/{kp_max}",
                overall_sentiment=f"{overall_points}/{overall_max}",
                sentiment_breakdown=f"{breakdown_points}/{breakdown_max}",
            ),
        )
