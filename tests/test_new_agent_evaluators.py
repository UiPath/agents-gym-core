"""The custom evaluators of the added agents are bounded and exact where they should be."""

import asyncio
import json
import types
from pathlib import Path
from typing import Any

import pytest
from uipath.eval.evaluators.evaluator_factory import EvaluatorFactory

import agents_gym_core

AGENTS = Path(agents_gym_core.__file__).parent / "agents"


def _evaluator(agent: str, evaluator_id: str) -> Any:
    evaluators = AGENTS / agent / "evaluations" / "evaluators"
    config = json.loads((evaluators / f"evaluator-{evaluator_id}.json").read_text())
    return EvaluatorFactory.create_evaluator(config, evaluators_dir=evaluators)


def _score(evaluator: Any, criteria: dict[str, Any], output: dict[str, Any]) -> float:
    run = types.SimpleNamespace(workload_output=output, agent_input={}, workload_trace=[])
    parsed = evaluator.evaluation_criteria_type.model_validate(criteria)
    return float(asyncio.run(evaluator.evaluate(run, parsed)).score)


KEY_PHRASES = [
    {"phrase": "delivery time was acceptable", "sentiment": "Neutral"},
    {"phrase": "instructions were unclear", "sentiment": "Negative"},
]
SEMANTIC_CRITERIA: dict[str, Any] = {
    "overallSentimentScore": -20,
    "overallSentimentLabel": "Slightly Negative",
    "positiveStatements": 1,
    "negativeStatements": 2,
    "neutralStatements": 2,
    "totalStatements": 5,
    "keyPhrases": json.dumps(KEY_PHRASES),
}


def _semantic_output(key_phrases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overallSentimentScore": -20,
        "overallSentimentLabel": "Slightly Negative",
        "positiveStatements": 1,
        "negativeStatements": 2,
        "neutralStatements": 2,
        "totalStatements": 5,
        "keyPhrases": json.dumps(key_phrases),
    }


def test_repeating_key_phrases_does_not_raise_the_sentiment_score() -> None:
    evaluator = _evaluator("semantic_analysis", "semantic_analysis_SemanticAnalysisScoreEvaluator")
    once = _score(evaluator, SEMANTIC_CRITERIA, _semantic_output(KEY_PHRASES))
    five_times = _score(evaluator, SEMANTIC_CRITERIA, _semantic_output(KEY_PHRASES * 5))

    assert once == pytest.approx(1.0)
    assert five_times == pytest.approx(once)


@pytest.mark.parametrize(
    ("actual", "score"),
    [("4016.99.6010", 1.0), ("4016.99.60.10", 1.0), ("4016996010", 1.0), ("4016.99.60", 0.0)],
)
def test_the_hts_code_matches_by_digits(actual: str, score: float) -> None:
    evaluator = _evaluator("hts_classifier_agent", "hts_classifier_agent_HTSCodeMatchEvaluator")
    assert _score(evaluator, {"expectedCode": "4016.99.6010"}, {"Top_HTS_code": actual}) == score


@pytest.mark.parametrize(
    ("result", "score"),
    [
        ("There are eight planets.", 1.0),
        ("There are 8 planets.", 1.0),
        ("Einstein wrote E = mc².", 1.0),
        ("Pluto is no longer counted.", 0.0),
    ],
)
def test_search_keywords_accept_any_listed_spelling(result: str, score: float) -> None:
    evaluator = _evaluator("search_agent", "search_agent_SearchContainsEvaluator")
    keywords = [["E=mc²", "E = mc²"]] if "mc" in result else [["8", "eight"]]
    assert _score(evaluator, {"keywords": keywords}, {"result": result}) == score
