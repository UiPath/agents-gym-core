"""Build and run Gym agents, compare loops, and evaluate cases from Python."""

import math
import re
import string
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from agents_gym_core.hydra_agent import CoreAgentRegistry
from agents_gym_core.hydra_agent.evaluation_loader import check_datapoint_names
from agents_gym_core.loops import get_loop
from agents_gym_core.loops.deep_agents_loop import (
    build_deep_agents_input,
    normalize_deep_agents_output,
)
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION, LoopName
from agents_gym_core.utils.gym_types import Agent

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from uipath.eval.evaluators import BaseEvaluator
    from uipath.eval.models.evaluation_set import EvaluationItem

    from agents_gym_core.utils.trace_utils import InMemoryLogExporter, PerTraceSpanExporter


class GymRuntime:
    """Use an existing chat model with Gym's registry and the agent's loop.

    With the default ``loop_name`` and ``loop_version``, an agent with its own ``loop:`` (the
    public benchmarks' DSPy programs) runs that loop, as in the testbed, and any other agent
    the basic loop. ``loop_name`` and ``loop_version`` select a different existing Gym loop.
    """

    def __init__(
        self,
        model: BaseChatModel,
        *,
        registry: CoreAgentRegistry | None = None,
        embeddings: Embeddings | None = None,
        loop_version: str = DEFAULT_LOOP_VERSION,
        loop_name: str = str(LoopName.BASIC),
    ) -> None:
        self.model = model
        self.embeddings = embeddings
        self.registry = CoreAgentRegistry(load_evaluations=False) if registry is None else registry
        self.loop_version = loop_version
        self.loop_name = loop_name

    def graph(
        self,
        agent_name: str,
        *,
        version: str = "base",
        tools: list[BaseTool] | None = None,
        input_schema: type[BaseModel] | None = None,
    ) -> CompiledStateGraph:
        """Build a native LangGraph graph; reuse it for streaming or repeated calls."""
        return self._build(agent_name, version, tools, input_schema)[0]

    def _build(
        self,
        agent_name: str,
        version: str,
        tools: list[BaseTool] | None,
        input_schema: type[BaseModel] | None,
    ) -> tuple[CompiledStateGraph, Agent, str]:
        """The compiled graph, the agent and the loop it runs on."""
        if tools is None:
            agent = self.registry.get_agent(agent_name, agent_version=version)
        else:
            agent = self.registry.get_agent(agent_name, agent_version=version, tools=tools)
        if input_schema is not None:
            agent.input_schema = input_schema
        models: dict[str, BaseChatModel | Embeddings] = {"model": self.model}
        if self.embeddings is not None:
            models["embeddings"] = self.embeddings
        loop_name, loop_version = self.loop_name, self.loop_version
        # As in the testbed, an agent's own loop replaces the default basic loop.
        if (
            agent.loop
            and self.loop_name == str(LoopName.BASIC)
            and self.loop_version == DEFAULT_LOOP_VERSION
        ):
            name, _, own_version = agent.loop.partition(":")
            loop_name, loop_version = name, own_version or DEFAULT_LOOP_VERSION
        if (
            input_schema is not None
            and loop_name != str(LoopName.DSPY_PROGRAM)  # runs its own program, no user prompt
            and not agent.is_conversational
        ):
            _check_user_prompt_fields(agent_name, agent.user_prompt, input_schema)
        graph = get_loop(loop_name, loop_version, agent, models)
        graph._gym_agent = agent  # pyright: ignore[reportAttributeAccessIssue]
        return graph, agent, loop_name

    async def run(
        self,
        agent_name: str,
        inputs: dict[str, Any],
        *,
        version: str = "base",
        config: RunnableConfig | None = None,
        tools: list[BaseTool] | None = None,
        input_schema: type[BaseModel] | None = None,
    ) -> dict[str, Any]:
        """Validate inputs, invoke the agent, and return its output as a dictionary."""
        graph, agent, loop_name = self._build(agent_name, version, tools, input_schema)
        if loop_name == LoopName.DEEP_AGENTS:  # takes messages, not the agent's input schema
            validated = agent.input_schema.model_validate(inputs).model_dump()
            result = await graph.ainvoke(build_deep_agents_input(agent, validated), config=config)
            return normalize_deep_agents_output(result)
        validated = graph.get_input_schema().model_validate(inputs)
        result = await graph.ainvoke(validated, config=config)
        return result.model_dump() if isinstance(result, BaseModel) else result

    async def evaluate(
        self,
        agent_name: str,
        cases: Sequence["EvaluationItem"],
        evaluators: Sequence["BaseEvaluator"],
        *,
        version: str = "base",
        graph: CompiledStateGraph | None = None,
        graphs: Sequence[CompiledStateGraph] | None = None,
        repeats: int = 1,
        concurrency: int = 1,
        judge_model: BaseChatModel | None = None,
        evaluator_weights: dict[str, float] | None = None,
        output_dir: Path | str = "experiments/evaluation",
        tracing: tuple["PerTraceSpanExporter", "TracerProvider", "InMemoryLogExporter"]
        | None = None,
    ) -> dict[str, Any]:
        """Evaluate selected Gym graphs in one experiment using the native runners.

        ``graphs`` compares loop configurations with shared concurrency and reports.
        Agent evaluator weights are retained unless explicitly overridden. Supplied
        tracing is reused without replacing its provider or exporters. ``judge_model``
        configures LLM evaluators without mutating the supplied evaluator instances.
        ``failed_datapoint_runs`` records unscored cases; ``failed_evaluator_runs`` records
        evaluator exceptions of agent runs that did not fail. ``evaluation_complete`` is false
        when either list has entries.
        """
        from agents_gym_core.runtime.run_utils import GraphDatapoint, evaluator_exception_lines
        from agents_gym_core.runtime.runners.result_aggregator import ResultAggregator
        from agents_gym_core.runtime.runners.streaming_runner import run_and_evaluate_all
        from agents_gym_core.runtime.runtime_config import RuntimeConfig

        if repeats < 1 or concurrency < 1:
            raise ValueError("repeats and concurrency must be positive")
        if not cases or len({case.name for case in cases}) != len(cases):
            raise ValueError("Provide cases with unique names")
        check_datapoint_names([case.name for case in cases])  # each gets its own run directory
        if graph is not None and graphs is not None:
            raise ValueError("Provide graph or graphs, not both")
        checks = {check.id: check for check in evaluators}
        if len(checks) != len(evaluators):
            raise ValueError("Evaluator IDs must be unique")
        if judge_model is not None:
            from functools import partial

            from uipath.eval.evaluators.llm_as_judge_evaluator import LLMJudgeMixin

            from agents_gym_core.models.judge_routing import routed_llm_service
            from agents_gym_core.models.utils import get_model_name

            checks = {
                name: check.model_copy(
                    update={
                        "evaluator_config": check.evaluator_config.model_copy(
                            update={"model": get_model_name(judge_model)}
                        ),
                        "llm_service": partial(
                            routed_llm_service, chat_model=judge_model, is_judge=True
                        ),
                    }
                )
                if isinstance(check, LLMJudgeMixin)
                else check
                for name, check in checks.items()
            }
        if evaluator_weights is not None:
            if evaluator_weights.keys() != checks.keys():
                raise ValueError("Provide one weight for each evaluator ID")
            if any(
                not math.isfinite(weight) or weight < 0 for weight in evaluator_weights.values()
            ) or not any(evaluator_weights.values()):
                raise ValueError("Weights must be finite, nonnegative and not all zero")
        for case in cases:
            missing = case.evaluation_criterias.keys() - checks.keys()
            if missing:
                raise ValueError(f"Unknown evaluator IDs: {sorted(missing)}")
        selected = (
            list(graphs)
            if graphs is not None
            else [graph if graph is not None else self.graph(agent_name, version=version)]
        )
        if not selected:
            raise ValueError("Provide at least one graph")
        prepared: list[tuple[CompiledStateGraph, Agent]] = []
        agents: dict[tuple[str, str], Agent] = {}
        seen = set()
        for compiled in selected:
            agent = getattr(compiled, "_gym_agent", None)
            if not isinstance(agent, Agent) or agent.name != agent_name:
                raise ValueError("Use graphs built by GymRuntime.graph() for this agent")
            key = (agent.name, agent.version, getattr(compiled, "loop_details").loop_key)
            if key in seen:
                raise ValueError("Each graph must have a distinct agent/version/loop key")
            seen.add(key)
            agent = agent.model_copy()
            agent.evaluation_set = agent.evaluation_set.model_copy(
                update={"evaluations": list(cases)}
            )
            if evaluator_weights is not None:
                agent.evaluator_weights = dict(evaluator_weights)
            agents[(agent.name, agent.version)] = agent
            prepared.append((compiled, agent))
        work: list[GraphDatapoint] = [
            (compiled, case, agent, run)
            for run in range(1, repeats + 1)
            for case in cases
            for compiled, agent in prepared
        ]
        output_dir = Path(output_dir).resolve()
        config = RuntimeConfig(
            num_runs=repeats,
            max_concurrency=concurrency,
            mode="overwrite",
            base_output_dir=output_dir.parent,
            experiment_name=output_dir.name,
        )
        own_tracing = tracing is None
        session = _acquire_tracing() if own_tracing else tracing
        failed: list[str] = []
        try:
            results = await run_and_evaluate_all(
                work,
                self.registry,
                len(work),
                config,
                evaluators=checks,
                tracing=session,
                failed=failed,
            )
        finally:
            if own_tracing:
                _release_tracing()
        aggregator = ResultAggregator(
            work, [case.name for case in cases], list(agents.values()), config
        )
        report = aggregator.finalize(results, use_metric_weights=True, output_dir=output_dir)
        report["failed_datapoint_runs"] = failed
        failed_evaluators = evaluator_exception_lines(results)
        report["failed_evaluator_runs"] = failed_evaluators
        report["evaluation_complete"] = not (failed or failed_evaluators)
        if failed:
            warnings.warn(
                f"Evaluation incomplete: {len(failed)} of {len(work)} datapoint runs failed; "
                "aggregated scores cover only completed cases. See failed_datapoint_runs.",
                RuntimeWarning,
                stacklevel=2,
            )
        if failed_evaluators:
            warnings.warn(
                f"Evaluation incomplete: {len(failed_evaluators)} evaluator runs raised. "
                "See failed_evaluator_runs before using aggregate scores.",
                RuntimeWarning,
                stacklevel=2,
            )
        return report


def _check_user_prompt_fields(
    agent_name: str, user_prompt: str, input_schema: type[BaseModel]
) -> None:
    """Raise if ``user_prompt`` uses a field ``input_schema`` lacks.

    The basic loop would otherwise replace the whole user prompt with a generic one
    (``loops.nodes.create_messages``).
    """
    try:
        fields = {
            re.split(r"[.\[]", field, maxsplit=1)[0]
            for _, field, _, _ in string.Formatter().parse(user_prompt)
            if field
        }
    except ValueError:
        return  # A malformed template already fails when the run starts.
    missing = sorted(fields - set(input_schema.model_fields))
    if missing:
        raise ValueError(
            f"{agent_name}'s user prompt uses {', '.join(missing)}, which "
            f"input_schema {input_schema.__name__} does not define"
        )


_TRACING: tuple["PerTraceSpanExporter", "TracerProvider", "InMemoryLogExporter"] | None = None
_TRACING_OWNED = False
_TRACING_USERS = 0


def _acquire_tracing() -> tuple["PerTraceSpanExporter", "TracerProvider", "InMemoryLogExporter"]:
    """The tracing session shared by the ``evaluate()`` calls running in this process.

    ``setup_tracer()`` replaces the process's providers and exporters, so overlapping calls
    each running it would leave the earlier ones without spans; spans are kept per trace, so
    they share one session. A live session from a ``setup_tracer()`` elsewhere (the caller's
    own, the testbed's, GEPA's) is reused as is; otherwise one is created for these calls.
    """
    global _TRACING, _TRACING_OWNED, _TRACING_USERS
    from agents_gym_core.utils import trace_utils

    current = trace_utils.current_tracing_session()
    if _TRACING is None or _TRACING is not current:
        if current is not None:
            _TRACING, _TRACING_OWNED = current, False
        else:
            _TRACING, _TRACING_OWNED = trace_utils.setup_tracer(), True
    _TRACING_USERS += 1
    return _TRACING


def _release_tracing() -> None:
    """After the last running ``evaluate()``, end the session if these calls created it, so
    tracing stays opt-in for later ``run()`` calls. A session set up elsewhere is left running."""
    global _TRACING, _TRACING_OWNED, _TRACING_USERS
    from agents_gym_core.utils import trace_utils

    _TRACING_USERS -= 1
    if _TRACING_USERS == 0 and _TRACING is not None:
        if _TRACING_OWNED and trace_utils.current_tracing_session() is _TRACING:
            _TRACING[1].shutdown()
        _TRACING, _TRACING_OWNED = None, False
