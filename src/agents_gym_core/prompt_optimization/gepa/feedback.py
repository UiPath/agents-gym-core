"""Synthesize textual feedback from per-evaluator outcomes.

The proposer LLM reads these blocks and infers what the agent did wrong.
The format intentionally exposes evaluator IDs and per-evaluator details
verbatim so the proposer can address specific failure modes (e.g. "Tool
``add`` was called 0 times, expected 1").
"""

from typing import Any

from pydantic import BaseModel

from agents_gym_core.prompt_optimization.gepa.rollout import RolloutResult


def feedback_for_rollout(rollout: RolloutResult) -> str:
    lines: list[str] = []
    lines.append(f"Inputs: {_short(rollout.inputs)}")
    lines.append(f"AgentOutput: {_short(rollout.output)}")
    lines.append(f"AggregateScore: {rollout.aggregated_score:.2f}")
    if rollout.error:
        lines.append(f"ExecutionError: {rollout.error}")
    for outcome in rollout.evaluator_outcomes:
        lines.append(
            f"- {outcome.evaluator_id}: score={outcome.score:.2f} | details={_short(outcome.details)}"
        )
    return "\n".join(lines)


def _short(obj: Any) -> str:
    """Stringify ``obj`` for the GEPA reflection feedback.

    Preserve full evaluator justifications, expected/actual values and tool outputs for
    reflection feedback.

    Pydantic ``BaseModel`` justifications (``LLMJudgeJustification``,
    ``ToolCallOutputJustification``, …) are dumped as indented JSON so the
    proposer reads ``"justification": "…"`` instead of the default
    ``Model(field='…', …)`` repr — cleaner field separation, easier to
    follow on multi-line ``actual``/``expected`` strings.
    """
    if obj is None:
        return ""
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)
    return str(obj)
