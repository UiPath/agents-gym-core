"""Rollout: run a candidate against datapoints + score it.

A rollout takes a candidate (``dict[str, str]``), overlays it onto a
loaded :class:`Agent` (only the prompt text changes — tools, schemas,
evaluators are reused), compiles a fresh basic-loop graph, invokes it,
and runs the agent's evaluator battery against the result.

The evaluator interface mirrors what
``agents_gym_core.runtime.runtime_helpers.RuntimeHelpers.run_and_save_evaluator``
does, minus the disk I/O.
"""

import asyncio
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from opentelemetry.sdk.trace import ReadableSpan
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.models import WorkloadExecution
from uipath.eval.models.evaluation_set import EvaluationItem
from uipath_langchain.agent.exceptions import AgentRuntimeError

from agents_gym_core.loops import get_loop
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION
from agents_gym_core.prompt_optimization.gepa.state import Candidate
from agents_gym_core.runtime.evaluation_criteria import (
    is_infrastructure_failure,
    prepare_evaluation_criteria,
)
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.span_normalize import normalize_tool_spans
from agents_gym_core.utils.trace_utils import PerTraceSpanExporter, setup_tracer

logger = logging.getLogger(__name__)

# Candidate key for the agent's main system instruction.
SYSTEM_PROMPT_KEY = "system_prompt"


@dataclass
class EvaluatorOutcome:
    evaluator_id: str
    score: float
    details: Any


@dataclass
class RolloutResult:
    datapoint_name: str
    inputs: dict[str, Any]
    output: Any
    evaluator_outcomes: list[EvaluatorOutcome]
    aggregated_score: float
    error: str | None = None


_tracer_state: dict[str, Any] = {}
_tracer_lock = threading.Lock()


def _ensure_tracing() -> dict[str, Any]:
    """Set up the OTel tracer and exporters once per process and cache their handles.

    A lock serializes provider initialization so parallel rollouts share one tracing session.
    """
    with _tracer_lock:
        if "span_exporter" not in _tracer_state:
            span_exporter, tracer_provider, log_exporter = setup_tracer(experiment_dir=None)
            _tracer_state["span_exporter"] = span_exporter
            _tracer_state["tracer_provider"] = tracer_provider
            _tracer_state["log_exporter"] = log_exporter
    return _tracer_state


def apply_candidate(agent: Agent, candidate: Candidate) -> Agent:
    """Return a shallow-copied Agent with candidate's instructions overlaid.

    Tools, schemas, and evaluation_set are shared by reference — only the
    prompt text fields are replaced. This is safe because the loop builds
    fresh tool bindings on each compile."""
    update: dict[str, Any] = {}
    if SYSTEM_PROMPT_KEY in candidate:
        update["system_prompt"] = candidate[SYSTEM_PROMPT_KEY]
    return agent.model_copy(update=update)


async def rollout_one(
    candidate: Candidate,
    agent: Agent,
    datapoint: EvaluationItem,
    model: BaseChatModel,
    evaluators: dict[str, BaseEvaluator],
    evaluator_weights: dict[str, float],
    *,
    loop_name: str = "basic",
    loop_version: str = DEFAULT_LOOP_VERSION,
    rollout_timeout_s: float = 240.0,
    llm_calls_dir: "Path | None" = None,
) -> RolloutResult:
    """Run ``candidate`` on a single datapoint and score it.

    ``rollout_timeout_s`` bounds each graph invocation in a concurrent batch.
    """
    state = _ensure_tracing()
    span_exporter: PerTraceSpanExporter = state["span_exporter"]
    tracer_provider = state["tracer_provider"]
    tracer = tracer_provider.get_tracer(__name__)

    mutated = apply_candidate(agent, candidate)

    compiled = get_loop(
        loop_name=loop_name,
        version=loop_version,
        scenario=mutated,
        models={"model": model},
    )

    inputs = dict(datapoint.inputs)
    output: Any
    error: str | None = None
    spans: list[ReadableSpan] = []
    with tracer.start_as_current_span(name="gepa_rollout") as root_span:
        trace_id = root_span.get_span_context().trace_id
        try:
            output = await asyncio.wait_for(
                compiled.ainvoke(
                    inputs,
                    config={"configurable": {"thread_id": str(uuid4())}},
                ),
                timeout=rollout_timeout_s,
            )
        except TimeoutError:
            output = {
                "type": "error",
                "code": "GENERAL_EXECUTION_ERROR",
                "title": "RolloutTimeout",
                "details": f"exceeded {rollout_timeout_s}s",
            }
            error = f"RolloutTimeout: exceeded {rollout_timeout_s}s"
            logger.warning("rollout timeout on %s after %.0fs", datapoint.name, rollout_timeout_s)
        except Exception as e:
            output = {
                "type": "error",
                "code": e.error_info.code
                if isinstance(e, AgentRuntimeError)
                else "GENERAL_EXECUTION_ERROR",
                "title": type(e).__name__,
                "details": str(e),
            }
            error = f"{type(e).__name__}: {e}"
            logger.warning("rollout error on %s: %s", datapoint.name, e)
    spans = span_exporter.pop_trace(trace_id)

    # Flatten langgraph tool-message payloads so ToolCallOutputEvaluator finds ``content``
    # at the top level.
    normalized_spans = normalize_tool_spans(spans)

    # Optional langsmith-style local dump of LLM calls from this rollout's
    # OTEL spans (agent path; reflection path is handled separately).
    if llm_calls_dir is not None:
        from agents_gym_core.prompt_optimization.gepa.io_logger import dump_rollout_spans

        try:
            dump_rollout_spans(
                spans,
                Path(llm_calls_dir) / "rollouts.jsonl",
                datapoint=datapoint.name,
            )
        except Exception as e:
            logger.warning("dump_rollout_spans failed for %s: %s", datapoint.name, e)

    agent_execution = WorkloadExecution(
        agent_input=inputs,
        workload_output=output if isinstance(output, dict) else {"content": str(output)},
        workload_trace=normalized_spans,
        simulation_instructions="",
    )

    outcomes: list[EvaluatorOutcome] = []
    weighted_sum = 0.0
    weight_total = 0.0
    for evaluator_id, criteria in datapoint.evaluation_criterias.items():
        if evaluator_id not in evaluators:
            continue
        evaluator = evaluators[evaluator_id]
        weight = evaluator_weights.get(evaluator_id, 1.0)
        if is_infrastructure_failure(output):
            score = 0.0
            details = output.get("details")
        else:
            try:
                result = await evaluator.validate_and_evaluate_criteria(
                    workload_execution=agent_execution,
                    evaluation_criteria=prepare_evaluation_criteria(
                        evaluator, agent_execution, criteria
                    ),
                )
                score = float(result.score)
                details = result.details
            except Exception as e:
                score = 0.0
                details = f"evaluator_error: {type(e).__name__}: {e}"
                logger.warning("evaluator %s crashed on %s: %s", evaluator_id, datapoint.name, e)
        outcomes.append(EvaluatorOutcome(evaluator_id=evaluator_id, score=score, details=details))
        # Weight 0 means: still run (for feedback) but don't contribute to aggregate
        if weight > 0:
            weighted_sum += score * weight
            weight_total += weight

    aggregated = weighted_sum / weight_total if weight_total else 0.0

    return RolloutResult(
        datapoint_name=datapoint.name,
        inputs=inputs,
        output=output,
        evaluator_outcomes=outcomes,
        aggregated_score=aggregated,
        error=error,
    )


async def rollout_batch(
    candidate: Candidate,
    agent: Agent,
    datapoints: list[EvaluationItem],
    model: BaseChatModel,
    evaluators: dict[str, BaseEvaluator],
    evaluator_weights: dict[str, float],
    *,
    max_concurrency: int = 8,
    loop_name: str = "basic",
    loop_version: str = DEFAULT_LOOP_VERSION,
    llm_calls_dir: Path | None = None,
) -> list[RolloutResult]:
    """Run ``candidate`` on multiple datapoints concurrently."""
    sem = asyncio.Semaphore(max_concurrency)

    async def _one(dp: EvaluationItem) -> RolloutResult:
        async with sem:
            return await rollout_one(
                candidate,
                agent,
                dp,
                model,
                evaluators,
                evaluator_weights,
                loop_name=loop_name,
                loop_version=loop_version,
                llm_calls_dir=llm_calls_dir,
            )

    return await asyncio.gather(*[_one(dp) for dp in datapoints])
