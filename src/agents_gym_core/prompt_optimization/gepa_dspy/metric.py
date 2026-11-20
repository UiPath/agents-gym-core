"""DSPy GEPA metric for gym agents.

dspy.GEPA expects ``metric(gold, pred, trace, pred_name, pred_trace)``
returning either a float or a ``dspy.Prediction(score=..., feedback=...)``.

We run the agent's evaluator battery against the gym graph result captured
in ``pred.output`` (set by :class:`GymAgentModule.forward`), aggregate to a
single score using the agent's ``evaluator_weights``, and synthesize a
text-feedback block the GEPA proposer can use.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import dspy
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.models import WorkloadExecution

from agents_gym_core.prompt_optimization.gepa.feedback import _short
from agents_gym_core.prompt_optimization.gepa_dspy.dataset import get_eval_item
from agents_gym_core.prompt_optimization.gepa_dspy.module import EVENT_LOOP
from agents_gym_core.runtime.evaluation_criteria import (
    is_infrastructure_failure,
    prepare_evaluation_criteria,
)
from agents_gym_core.utils.span_normalize import normalize_tool_spans

logger = logging.getLogger(__name__)


def make_gym_metric(
    *,
    evaluators: dict[str, BaseEvaluator],
    evaluator_weights: dict[str, float],
    llm_calls_dir: Path | None = None,
):
    """Return a metric callable bound to ``evaluators`` and ``weights``.

    If ``llm_calls_dir`` is provided, every rollout's LLM-kind OTEL spans
    are appended to ``rollouts.jsonl`` for langsmith-style local inspection.
    """

    def gym_metric(
        gold: dspy.Example,
        pred: dspy.Prediction | None,
        trace: Any | None = None,
        pred_name: str | None = None,
        pred_trace: Any | None = None,
    ) -> dspy.Prediction:
        item = get_eval_item(gold)
        # `pred` is None when the program crashed before forward() returned.
        if pred is None:
            return dspy.Prediction(score=0.0, feedback="Agent crashed before producing output.")

        # See module.py: payload fields are prefixed with `agent_` to avoid
        # colliding with bound methods on dspy.Example (.inputs, .labels, ...).
        output = getattr(pred, "agent_output", None)
        spans = getattr(pred, "agent_spans", []) or []
        inputs = getattr(pred, "agent_inputs", None) or dict(item.inputs)

        # Optional langsmith-style local dump of agent rollout LLM calls. dspy.GEPA calls the
        # metric a second time (with `pred_name`) on the same prediction to collect reflection
        # feedback; only the first, module-level call logs, so each rollout is written once.
        if llm_calls_dir is not None and pred_name is None:
            from agents_gym_core.prompt_optimization.gepa.io_logger import dump_rollout_spans

            try:
                dump_rollout_spans(
                    spans,
                    Path(llm_calls_dir) / "rollouts.jsonl",
                    datapoint=item.name,
                )
            except Exception as e:
                logger.warning("dump_rollout_spans failed for %s: %s", item.name, e)

        agent_execution = WorkloadExecution(
            agent_input=inputs,
            workload_output=output if isinstance(output, dict) else {"content": str(output)},
            workload_trace=normalize_tool_spans(spans),
            simulation_instructions="",
        )

        outcomes: list[tuple[str, float, Any]] = []
        weighted_sum = 0.0
        weight_total = 0.0
        evaluator_timeout_s = 180.0  # cap on one evaluator call
        for evaluator_id, criteria in item.evaluation_criterias.items():
            if evaluator_id not in evaluators:
                continue
            evaluator = evaluators[evaluator_id]
            weight = evaluator_weights.get(evaluator_id, 1.0)
            if is_infrastructure_failure(output):
                score = 0.0
                details = output.get("details")
                outcomes.append((evaluator_id, score, details))
                if weight > 0:
                    weight_total += weight
                continue
            try:
                # Run the evaluator on the shared event loop with a per-call timeout.
                async def _run_evaluator() -> Any:
                    return await asyncio.wait_for(
                        evaluator.validate_and_evaluate_criteria(
                            workload_execution=agent_execution,
                            evaluation_criteria=prepare_evaluation_criteria(
                                evaluator, agent_execution, criteria
                            ),
                        ),
                        timeout=evaluator_timeout_s,
                    )

                result = asyncio.run_coroutine_threadsafe(_run_evaluator(), EVENT_LOOP).result()
                score = float(result.score)
                details = result.details
            except TimeoutError:
                score = 0.0
                details = f"evaluator_timeout: exceeded {evaluator_timeout_s}s"
            except Exception as e:
                score = 0.0
                details = f"evaluator_error: {type(e).__name__}: {e}"
            outcomes.append((evaluator_id, score, details))
            if weight > 0:
                weighted_sum += score * weight
                weight_total += weight

        aggregate = weighted_sum / weight_total if weight_total else 0.0

        feedback_lines: list[str] = []
        feedback_lines.append(f"Inputs: {_short(inputs)}")
        feedback_lines.append(f"AgentOutput: {_short(output)}")
        feedback_lines.append(f"AggregateScore: {aggregate:.2f}")
        for eid, score, details in outcomes:
            feedback_lines.append(f"- {eid}: score={score:.2f} | details={_short(details)}")
        feedback = "\n".join(feedback_lines)

        return dspy.Prediction(score=aggregate, feedback=feedback)

    return gym_metric
