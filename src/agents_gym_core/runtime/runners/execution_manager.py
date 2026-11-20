"""Execution management for agent runs with caching."""

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from tqdm.std import tqdm
from uipath.eval.models import WorkloadExecution
from uipath.eval.models.evaluation_set import EvaluationItem
from uipath.runtime.errors.contract import UiPathErrorCategory
from uipath_langchain.agent.exceptions import AgentRuntimeError

from agents_gym_core.hydra_agent import CoreAgentRegistry
from agents_gym_core.loops import LoopDetails
from agents_gym_core.loops.types import LoopName
from agents_gym_core.models.caching import caching_run_context
from agents_gym_core.runtime.cost_tracking import (
    COST_FILENAME,
    CostTrackingCallback,
    save_cost_record,
)
from agents_gym_core.runtime.run_utils import (
    AgentDatapointExecution,
    ExecutionMetadata,
    GraphDatapoint,
    get_datapoint_dir,
    get_experiment_path,
    load_execution,
    should_run_datapoint,
)
from agents_gym_core.runtime.runtime_config import RuntimeConfig
from agents_gym_core.utils.general_utils import gather_limited
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.span_normalize import normalize_tool_spans
from agents_gym_core.utils.trace_utils import (
    AGENT_OUTPUT_FILENAME,
    OUTPUT_FILENAME,
    InMemoryLogExporter,
    PerTraceSpanExporter,
    info,
    save_agent_output,
    save_execution_data,
    setup_tracer,
    warning,
)


def _error_title(error: BaseException) -> str:
    """The exception's type name. The UiPath LLM client raises a bare ``UiPathError`` when a
    request got no HTTP response (connection refused, timeout), raised from the provider error;
    that error's type names the failure."""
    if type(error).__name__ == "UiPathError" and error.__cause__ is not None:
        return type(error.__cause__).__name__
    return type(error).__name__


def _remove_saved_outputs(experiment_dir: Path, datapoint_name: str, run_number: int) -> None:
    """Delete the output.json, agent_output.json and cost.json an earlier run of the datapoint left.

    A run without saved traces writes none of them, so a later mode that reuses outputs would
    score the earlier run's, and the cost summary would count its cost.
    """
    datapoint_dir = get_datapoint_dir(experiment_dir, datapoint_name, run_number)
    for filename in (OUTPUT_FILENAME, AGENT_OUTPUT_FILENAME, COST_FILENAME):
        (datapoint_dir / filename).unlink(missing_ok=True)


class ExecutionManager:
    """Orchestrates agent execution with tracing and caching."""

    def __init__(
        self,
        registry: CoreAgentRegistry,
        config: RuntimeConfig,
        graphs_list: list[GraphDatapoint] | None = None,
        num_datapoints: int = 0,
    ):
        """``graphs_list`` and ``num_datapoints`` are used only by the batch :meth:`run`."""
        self.graphs_list = graphs_list or []
        self.num_datapoints = num_datapoints
        self.registry = registry
        self.config = config
        self._span_exporter: PerTraceSpanExporter | None = None
        self._tracer_provider: TracerProvider | None = None
        self._log_exporter: InMemoryLogExporter | None = None

    def setup_tracing(
        self,
        tracing: tuple[PerTraceSpanExporter, TracerProvider, InMemoryLogExporter] | None = None,
    ) -> None:
        """Use supplied tracing or initialize it before run_one()."""
        self._span_exporter, self._tracer_provider, self._log_exporter = tracing or setup_tracer()

    async def run_one(
        self,
        graph: CompiledStateGraph,
        datapoint: EvaluationItem,
        agent: Agent,
        run_number: int,
    ) -> AgentDatapointExecution | None:
        """Load one datapoint run from disk when the mode allows it, otherwise run it with tracing.

        Requires :meth:`setup_tracing`.
        """
        assert self._span_exporter and self._tracer_provider and self._log_exporter, (
            "Call setup_tracing() before run_one()"
        )

        loop_details = graph.loop_details  # pyright: ignore[reportAttributeAccessIssue]
        experiment_dir = get_experiment_path(
            base_output_dir=self.config.base_output_dir,
            experiment_name=self.config.experiment_name,
            agent=agent,
            loop_details=loop_details,
        )

        if self.config.mode != "overwrite" and not should_run_datapoint(
            experiment_dir,
            datapoint.name,
            self.config.mode,
            run_number,
            datapoint.evaluation_criterias.keys(),
        ):
            directory_provider = self.registry.get_directory_provider()
            execution = await asyncio.to_thread(
                load_execution,
                experiment_dir,
                datapoint,
                directory_provider,
                agent,
                run_number,
                loop_details,
            )
            if execution is not None:
                return execution
            # output.json is gone: run the datapoint.

        agent_experiment_dir = experiment_dir if self.config.save_traces else None
        if not self.config.save_traces:
            _remove_saved_outputs(experiment_dir, datapoint.name, run_number)
        return await self._run_single_agent_with_tracing(
            graph,
            datapoint,
            self._span_exporter,
            self._tracer_provider,
            self._log_exporter,
            agent,
            loop_details,
            agent_experiment_dir,
            run_number,
        )

    @staticmethod
    def _extract_last_assistant_content(spans: list[ReadableSpan]) -> str:
        """Content of the last LLM output message in the spans, or "" if there is none.

        A conversational run that ends without end_execution has its reply only in the trace.
        """
        for span in reversed(spans):
            attributes = span.attributes or {}

            # OpenInference keys: llm.output_messages.N.message.content, or
            # llm.output_messages.N.message.contents.M.message_content.text (conversational).
            for key, value in attributes.items():
                if key.startswith("llm.output_messages."):
                    if key.endswith(".message.content") or key.endswith(".message_content.text"):
                        content = str(value)
                        if content:
                            return content
        return ""

    @staticmethod
    async def _run_single_agent_with_tracing(
        graph: CompiledStateGraph,
        datapoint: EvaluationItem,
        span_exporter: PerTraceSpanExporter,
        tracer_provider: TracerProvider,
        log_exporter: InMemoryLogExporter,
        agent: Agent,
        loop_details: LoopDetails,
        output_dir: Path | None = None,
        run_number: int | None = None,
    ) -> AgentDatapointExecution:
        """Run one datapoint under a root span, collecting its trace and LLM cost.

        An agent exception becomes an error output. With ``output_dir`` (the experiment
        directory), output.json and cost.json are written to the datapoint directory.
        """
        tracer = tracer_provider.get_tracer(__name__)
        with tracer.start_as_current_span(
            name=f"{loop_details.loop_name}:{loop_details.loop_version} / {datapoint.name} / run {run_number}",
            attributes={"openinference.span.kind": "CHAIN", "gym.agent": agent.name},
        ) as root_span:
            trace_id = root_span.get_span_context().trace_id
            # An output_path input is redirected into the datapoint directory, so generated
            # files (plots, CSVs) sit next to output.json and results/.
            inputs = datapoint.inputs
            if output_dir is not None and "output_path" in inputs:
                dp_dir = get_datapoint_dir(output_dir, datapoint.name, run_number)
                inputs = {**inputs, "output_path": str(dp_dir / "generated_content")}

            # The deep-agents loop takes {"messages": [HumanMessage(...)]}, not the typed inputs.
            if loop_details.loop_name == LoopName.DEEP_AGENTS:
                from agents_gym_core.loops.deep_agents_loop import build_deep_agents_input

                inputs = build_deep_agents_input(agent, inputs)

            cost_callback = CostTrackingCallback()

            # SimulationTool reads the input mocking prompt from the run config (through
            # var_child_runnable_config), which keeps it out of the agent's input.
            input_mocking_prompt = (
                datapoint.input_mocking_strategy.prompt if datapoint.input_mocking_strategy else ""
            )

            try:
                with caching_run_context(run_number or 1):
                    result = await graph.ainvoke(
                        inputs,
                        config={
                            "configurable": {
                                "thread_id": str(uuid4()),
                                "input_mocking_prompt": input_mocking_prompt,
                                "datapoint_inputs": datapoint.inputs or {},
                            },
                            "callbacks": [cost_callback],
                        },
                    )
            except AgentRuntimeError as e:
                result = {
                    "type": "error",
                    "code": e.error_info.code,
                    "title": e.error_info.title,
                    "category": e.error_info.category,
                    "details": e.error_info.detail,
                }
                warning("Agent execution failed", **result)
            except Exception as e:
                result = {
                    "type": "error",
                    "code": "GENERAL_EXECUTION_ERROR",
                    "title": _error_title(e),
                    "category": UiPathErrorCategory.SYSTEM,
                    "details": str(e),
                }
                warning("Agent execution failed", **result)

        # Removes this trace's spans from the shared exporter, freeing their memory.
        filtered_spans = span_exporter.pop_trace(trace_id)

        # A conversational run that ends without end_execution returns None.
        if result is None:
            content = ExecutionManager._extract_last_assistant_content(filtered_spans)
            result = {"content": content}

        # Deep-agents graphs return {"messages": [AIMessage(...), ...]}, which is not
        # JSON-serializable; keep the end_execution output instead. Error dicts have no
        # "messages" key and reach the evaluator unchanged.
        if (
            loop_details.loop_name == LoopName.DEEP_AGENTS
            and isinstance(result, dict)
            and "messages" in result
        ):
            from agents_gym_core.loops.deep_agents_loop import normalize_deep_agents_output

            result = normalize_deep_agents_output(result)

        workload_execution = WorkloadExecution(
            agent_input=datapoint.inputs,
            workload_output=result,
            workload_trace=normalize_tool_spans(filtered_spans),
            simulation_instructions=datapoint.input_mocking_strategy.prompt
            if datapoint.input_mocking_strategy
            else "",
        )

        if output_dir and filtered_spans:
            datapoint_dir = get_datapoint_dir(output_dir, datapoint.name, run_number)
            # Before output.json, which marks the run as done, so a resumed run always finds it.
            save_agent_output(datapoint_dir, result)
            save_execution_data(datapoint_dir, filtered_spans)

        # The result aggregator reads cost from cost.json, so a datapoint loaded from disk
        # later still counts. A run with no LLM calls writes none and removes one an earlier
        # run of the datapoint left, which is not this run's cost.
        if output_dir:
            datapoint_dir = get_datapoint_dir(output_dir, datapoint.name, run_number)
            if cost_callback.record.num_llm_calls > 0:
                save_cost_record(datapoint_dir, cost_callback.record)
            else:
                (datapoint_dir / COST_FILENAME).unlink(missing_ok=True)

        return AgentDatapointExecution(
            datapoint=datapoint,
            workload_execution=workload_execution,
            run_number=run_number if run_number is not None else 1,
            agent_name=agent.name,
            agent_version=agent.version,
            loop_details=loop_details,
            cost=cost_callback.record if cost_callback.record.num_llm_calls > 0 else None,
            ran=True,
        )

    async def _filter_and_load_cached(
        self,
        graphs_list: list[GraphDatapoint],
    ) -> tuple[
        list[GraphDatapoint],
        list[AgentDatapointExecution],
        dict[int, ExecutionMetadata],
        int,
    ]:
        """Split the graphs into those to run and those whose executions load from disk.

        Returns (graphs_to_run, loaded_executions, metadata_map, skipped_count).
        """
        skipped_count = 0
        skipped_items: list[tuple[EvaluationItem, Agent, CompiledStateGraph, int]] = []
        filtered_graphs = []

        execution_metadata_map: dict[int, ExecutionMetadata] = {}

        if self.config.mode != "overwrite":
            for graph, datapoint, agent, run_number in graphs_list:
                loop_details = graph.loop_details  # pyright: ignore[reportAttributeAccessIssue]

                experiment_dir = get_experiment_path(
                    base_output_dir=self.config.base_output_dir,
                    experiment_name=self.config.experiment_name,
                    agent=agent,
                    loop_details=loop_details,
                )

                if should_run_datapoint(
                    experiment_dir,
                    datapoint.name,
                    self.config.mode,
                    run_number,
                    datapoint.evaluation_criterias.keys(),
                ):
                    filtered_graphs.append((graph, datapoint, agent, run_number))
                else:
                    skipped_count += 1
                    skipped_items.append((datapoint, agent, graph, run_number))
                    if self.config.verbose:
                        info(
                            "Skipping datapoint (already executed)",
                            datapoint_name=datapoint.name,
                            agent_name=agent.name,
                            mode=self.config.mode,
                        )

            if self.config.verbose:
                info(
                    "Filtered datapoints based on cache",
                    total_datapoints=self.num_datapoints,
                    to_run=len(filtered_graphs),
                    skipped=skipped_count,
                    mode=self.config.mode,
                )
        else:
            filtered_graphs = graphs_list

        loaded_executions: list[AgentDatapointExecution] = []
        if skipped_items:
            if self.config.verbose:
                info("Loading skipped datapoints from disk", count=len(skipped_items))

            directory_provider = self.registry.get_directory_provider()

            async def _load_one_async(
                datapoint: EvaluationItem,
                agent: Agent,
                graph: CompiledStateGraph,
                run_number: int,
            ) -> tuple[AgentDatapointExecution | None, LoopDetails, Agent, int]:
                loop_details = graph.loop_details  # pyright: ignore[reportAttributeAccessIssue]
                experiment_dir = get_experiment_path(
                    base_output_dir=self.config.base_output_dir,
                    experiment_name=self.config.experiment_name,
                    agent=agent,
                    loop_details=loop_details,
                )
                execution = await asyncio.to_thread(
                    load_execution,
                    experiment_dir,
                    datapoint,
                    directory_provider,
                    agent,
                    run_number,
                    loop_details,
                )
                return execution, loop_details, agent, run_number

            results: list[
                tuple[AgentDatapointExecution | None, LoopDetails, Agent, int]
            ] = await gather_limited(
                max_concurrency=32,
                items=skipped_items,
                func=_load_one_async,
                total=len(skipped_items),
                show_progress=True,
                tqdm_description="Loading previous experiments",
            )

            for loaded_execution, loop_details, agent, run_number in results:
                if loaded_execution:
                    loaded_executions.append(loaded_execution)
                    execution_id = id(loaded_execution)
                    execution_metadata_map[execution_id] = ExecutionMetadata(
                        loop_details=loop_details,
                        agent=agent,
                        run_number=run_number,
                    )

        return filtered_graphs, loaded_executions, execution_metadata_map, skipped_count

    def _build_metadata_map(
        self,
        graphs_list: list[GraphDatapoint],
        executions: list[AgentDatapointExecution],
        metadata_map: dict[int, ExecutionMetadata],
    ) -> None:
        """Add an ExecutionMetadata entry for each new execution to ``metadata_map``, in place."""
        exec_index: dict[tuple[str, int, str, str, str], AgentDatapointExecution] = {}
        for execution in executions:
            key = (
                execution.datapoint.name,
                execution.run_number,
                execution.agent_name,
                execution.agent_version,
                execution.loop_key,
            )
            exec_index[key] = execution

        for graph, datapoint, agent, run_number in graphs_list:
            loop_details = graph.loop_details  # pyright: ignore[reportAttributeAccessIssue]
            key = (datapoint.name, run_number, agent.name, agent.version, loop_details.loop_key)
            execution = exec_index.get(key)
            if execution is not None:
                execution_id = id(execution)
                if execution_id not in metadata_map:
                    metadata_map[execution_id] = ExecutionMetadata(
                        loop_details=loop_details,
                        agent=agent,
                        run_number=run_number,
                    )

    async def run(self) -> tuple[list[AgentDatapointExecution], dict[int, ExecutionMetadata]]:
        """Run every graph datapoint, or load it from disk when the mode allows.

        Returns the executions and their metadata keyed by ``id(execution)``.
        """
        (
            filtered_graphs,
            loaded_executions,
            execution_metadata_map,
            skipped_count,
        ) = await self._filter_and_load_cached(self.graphs_list)

        experiment_dir = self.config.base_output_dir / self.config.experiment_name
        print(f"Running agents... (experiment: {experiment_dir})")

        if not filtered_graphs:
            if self.config.verbose:
                info("No datapoints to run (all already executed)", mode=self.config.mode)
            if self.num_datapoints > 0:
                pbar = tqdm(total=self.num_datapoints, initial=self.num_datapoints)
                pbar.close()
            return loaded_executions, execution_metadata_map

        if self.config.verbose:
            info(
                "Starting agent execution with trace collection",
                num_datapoints=len(filtered_graphs),
                max_concurrency=self.config.max_concurrency,
                mode=self.config.mode,
            )

        span_exporter, tracer_provider, log_exporter = setup_tracer(experiment_dir=None)

        def run_agent_fn(
            graph: CompiledStateGraph,
            datapoint: EvaluationItem,
            agent: Agent,
            run_number: int,
        ) -> Awaitable[AgentDatapointExecution]:
            loop_details = graph.loop_details  # pyright: ignore[reportAttributeAccessIssue]
            agent_experiment_dir = get_experiment_path(
                base_output_dir=self.config.base_output_dir,
                experiment_name=self.config.experiment_name,
                agent=agent,
                loop_details=loop_details,
            )
            if not self.config.save_traces:
                _remove_saved_outputs(agent_experiment_dir, datapoint.name, run_number)
            return self._run_single_agent_with_tracing(
                graph,
                datapoint,
                span_exporter,
                tracer_provider,
                log_exporter,
                agent,
                loop_details,
                agent_experiment_dir if self.config.save_traces else None,
                run_number,
            )

        new_executions: list[AgentDatapointExecution] = await gather_limited(
            max_concurrency=self.config.max_concurrency,
            items=filtered_graphs,
            func=run_agent_fn,
            total=self.num_datapoints,
            initial=skipped_count,
            show_progress=True,
        )

        self._build_metadata_map(filtered_graphs, new_executions, execution_metadata_map)

        all_executions = loaded_executions + new_executions
        return all_executions, execution_metadata_map
