"""Streaming execution and evaluation.

Each execution is evaluated as soon as it finishes and then released, so at most
``max_concurrency`` executions and their OTEL traces are held in memory. The work is
done by ``ExecutionManager.run_one()`` and ``EvaluationManager.evaluate_one()``.
"""

from collections import defaultdict
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from opentelemetry.sdk.trace import TracerProvider
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.models.evaluation_set import EvaluationItem

from agents_gym_core.hydra_agent import CoreAgentRegistry
from agents_gym_core.runtime.run_utils import GraphDatapoint
from agents_gym_core.runtime.runners.evaluation_manager import EvaluationManager
from agents_gym_core.runtime.runners.execution_manager import ExecutionManager
from agents_gym_core.runtime.runtime_config import RuntimeConfig
from agents_gym_core.utils.general_utils import gather_limited
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.trace_utils import InMemoryLogExporter, PerTraceSpanExporter

EvaluationResults = dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]


async def run_and_evaluate_all(
    graphs_list: list[GraphDatapoint],
    registry: CoreAgentRegistry,
    num_datapoints: int,
    config: RuntimeConfig,
    *,
    evaluators: dict[str, BaseEvaluator] | None = None,
    tracing: tuple[PerTraceSpanExporter, TracerProvider, InMemoryLogExporter] | None = None,
    failed: list[str] | None = None,
) -> EvaluationResults:
    """Run and evaluate every graph datapoint, evaluating each execution as soon as it ends.

    ``num_datapoints`` sizes the progress bar. Returns agent_key -> loop_key ->
    evaluator_id -> datapoint_key -> result_data. A datapoint run that raises is left out of
    the results (the agent and the evaluators record their own errors, so this is anything
    else, such as criteria that fail validation); one line per such run is added to ``failed``.
    """
    if not graphs_list:
        return defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    experiment_dir = config.base_output_dir / config.experiment_name
    print(f"Running agents with streaming evaluation... (experiment: {experiment_dir})")

    exec_mgr = ExecutionManager(registry=registry, config=config)
    exec_mgr.setup_tracing(tracing)

    eval_mgr = EvaluationManager(registry=registry, config=config, evaluators=evaluators)

    async def _run_and_evaluate(
        graph: CompiledStateGraph,
        datapoint: EvaluationItem,
        agent: Agent,
        run_number: int,
    ) -> list[dict[str, Any]]:
        try:
            execution = await exec_mgr.run_one(graph, datapoint, agent, run_number)
            if execution is None:
                return []
            return await eval_mgr.evaluate_one(execution, agent)
        except Exception as exc:
            if failed is not None:
                failed.append(
                    f"{agent.name}:{agent.version} {datapoint.name} run {run_number}: "
                    f"{type(exc).__name__}: {str(exc)[:200]}"
                )
            raise

    all_results: list[list[dict[str, Any]]] = await gather_limited(
        max_concurrency=config.max_concurrency,
        items=graphs_list,
        func=_run_and_evaluate,
        total=num_datapoints,
        show_progress=True,
        tqdm_description="Running & evaluating",
    )

    combined: EvaluationResults = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for result_list in all_results:
        for entry in result_list:
            combined[entry["agent_key"]][entry["loop_key"]][entry["evaluator_id"]][
                entry["datapoint_key"]
            ] = entry["result_data"]

    return combined
