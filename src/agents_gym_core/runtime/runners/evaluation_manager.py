"""Evaluation management for agent runs."""

from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm.std import tqdm
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import EvaluationResult

from agents_gym_core.hydra_agent import CoreAgentRegistry, EvaluationLoader
from agents_gym_core.loops import LoopDetails
from agents_gym_core.runtime.run_utils import (
    AgentDatapointExecution,
    ExecutionMetadata,
    get_experiment_path,
    load_cached_evaluator_result,
    should_run_evaluator,
)
from agents_gym_core.runtime.runtime_config import RuntimeConfig
from agents_gym_core.runtime.runtime_helpers import RuntimeHelpers
from agents_gym_core.utils.general_utils import gather_limited
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.trace_utils import info, warning


class EvaluationManager:
    """Manages evaluation task building, caching, and execution."""

    def __init__(
        self,
        registry: CoreAgentRegistry,
        config: RuntimeConfig,
        executions: list[AgentDatapointExecution] | None = None,
        loaded_agents: list[Agent] | None = None,
        metadata_map: dict[int, ExecutionMetadata] | None = None,
        *,
        evaluators: dict[str, BaseEvaluator] | None = None,
    ):
        """``executions``, ``loaded_agents`` and ``metadata_map`` (keyed by ``id(execution)``)
        are used only by the batch :meth:`run`; streaming calls :meth:`evaluate_one`."""
        self.executions = executions or []
        self.loaded_agents = loaded_agents or []
        self.metadata_map = metadata_map or {}
        self.registry = registry
        self.config = config
        self.evaluators = evaluators
        self.helpers = RuntimeHelpers(config)
        # Evaluators by agent name. EvaluationLoader.get_evaluators rebuilds them from disk
        # on every call, re-executing custom evaluator modules into sys.modules, and
        # evaluate_one runs once per datapoint. Evaluators hold no per-datapoint state.
        self._evaluators_cache: dict[str, dict[str, BaseEvaluator]] = {}

        # (agent_name, agent_version, loop_key) -> executions; batch mode only.
        self.agent_executions_map: dict[tuple[str, str, str], list[AgentDatapointExecution]] = (
            defaultdict(list)
        )
        for execution in self.executions:
            key = (
                execution.agent_name,
                execution.agent_version,
                execution.loop_key,
            )
            self.agent_executions_map[key].append(execution)

    def _try_load_cached_result(
        self,
        experiment_dir: Path,
        datapoint_name: str,
        evaluator: BaseEvaluator,
        agent_name: str,
        run_number: int | None = None,
    ) -> dict[str, Any] | None:
        """Load a saved evaluator result; None if it is missing or unreadable."""
        try:
            return load_cached_evaluator_result(
                experiment_dir, datapoint_name, evaluator.id, evaluator, run_number
            )
        except Exception as e:
            if self.config.verbose:
                warning(
                    "Failed to load cached evaluation result",
                    datapoint_name=datapoint_name,
                    evaluator_id=evaluator.id,
                    agent_name=agent_name,
                    error=str(e),
                )
            return None

    def _get_evaluators(self, agent_name: str) -> dict[str, BaseEvaluator]:
        """An agent's evaluators, built once per agent name (see ``_evaluators_cache``)."""
        if self.evaluators is not None:
            return self.evaluators
        cached = self._evaluators_cache.get(agent_name)
        if cached is None:
            cached = EvaluationLoader.get_evaluators(
                agent_name=agent_name,
                directory_provider=self.registry.get_directory_provider(),
            )
            self._evaluators_cache[agent_name] = cached
        return cached

    def _build_evaluation_tasks(
        self,
    ) -> tuple[
        list[tuple[Agent, BaseEvaluator, AgentDatapointExecution, LoopDetails]],
        dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]],
        int,
        int,
    ]:
        """List the evaluations to run and load the saved results the mode allows reusing.

        Returns (tasks, skipped_results, total_count, skipped_count).
        """
        all_tasks: list[tuple[Agent, BaseEvaluator, AgentDatapointExecution, LoopDetails]] = []
        skipped_results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(dict))
        )
        total_evaluations = 0
        skipped_evaluations = 0

        for (
            agent_name,
            agent_version,
            loop_key,
        ), executions in self.agent_executions_map.items():
            agent = next(
                (
                    a
                    for a in self.loaded_agents
                    if a.name == agent_name and a.version == agent_version
                ),
                None,
            )
            if not agent:
                continue

            evaluators = self._get_evaluators(agent.name)

            for execution in executions:
                execution_id = id(execution)
                metadata = self.metadata_map.get(execution_id)
                if not metadata:
                    continue
                loop_details = metadata.loop_details

                evaluation_criterias = execution.datapoint.evaluation_criterias
                for evaluator_id in evaluation_criterias:
                    if evaluator_id not in evaluators:
                        continue

                    total_evaluations += 1
                    datapoint_name = execution.datapoint.name
                    experiment_dir = get_experiment_path(
                        base_output_dir=self.config.base_output_dir,
                        experiment_name=self.config.experiment_name,
                        agent=agent,
                        loop_details=loop_details,
                    )
                    run_number = execution.run_number

                    # A datapoint whose agent ran in this invocation is always evaluated again.
                    if not getattr(execution, "ran", False) and not should_run_evaluator(
                        experiment_dir,
                        datapoint_name,
                        evaluator_id,
                        self.config.mode,
                        run_number,
                        execution.datapoint.evaluation_criterias.keys(),
                    ):
                        # A saved result that cannot be read is evaluated again below, so it
                        # does not drop out of the aggregates.
                        cached_result = self._try_load_cached_result(
                            experiment_dir,
                            datapoint_name,
                            evaluators[evaluator_id],
                            agent.name,
                            run_number,
                        )
                        if cached_result:
                            skipped_evaluations += 1
                            if self.config.verbose:
                                info(
                                    "Skipping evaluator (already executed)",
                                    datapoint_name=datapoint_name,
                                    evaluator_id=evaluator_id,
                                    agent_name=agent.name,
                                    mode=self.config.mode,
                                )
                            # The same metadata fields as a freshly evaluated result.
                            cached_result["agent_name"] = agent.name
                            cached_result["agent_version"] = agent.version
                            cached_result["loop_details"] = loop_details.model_dump()
                            cached_result["run_number"] = run_number
                            # The split comes from the agent, as a result file may not record it.
                            cached_result["split"] = agent.datapoint_splits.get(
                                datapoint_name, "all_data"
                            )
                            datapoint_key = f"{datapoint_name}:run_{run_number}"
                            agent_key = f"{agent.name}__{agent.version}"
                            skipped_results[agent_key][loop_key][evaluator_id][datapoint_key] = (
                                cached_result
                            )
                            continue

                    all_tasks.append(
                        (
                            agent,
                            evaluators[evaluator_id],
                            execution,
                            loop_details,
                        )
                    )

        return all_tasks, skipped_results, total_evaluations, skipped_evaluations

    async def _run_evaluator_task(
        self,
        agent: Agent,
        evaluator: BaseEvaluator,
        execution: AgentDatapointExecution,
        loop_details: LoopDetails,
    ) -> tuple[
        str,
        str,
        LoopDetails,
        str,
        str,
        int,
        dict[str, Any] | None,
        dict[str, Any] | str,
        EvaluationResult,
        str | None,
        str | None,
    ]:
        """Run one evaluator task and save its result to disk.

        Returns (agent_name, agent_version, loop_details, evaluator_id, datapoint_name,
        run_number, evaluation_criteria, agent_output, result, judge_model, judge_prompt).
        """
        run_number = execution.run_number

        experiment_dir = get_experiment_path(
            base_output_dir=self.config.base_output_dir,
            experiment_name=self.config.experiment_name,
            agent=agent,
            loop_details=loop_details,
        )

        (
            evaluator_id,
            evaluation_criteria,
            agent_output,
            result,
            judge_model,
            judge_prompt,
        ) = await self.helpers.run_and_save_evaluator(
            agent=agent,
            evaluator=evaluator,
            execution=execution,
            experiment_dir=experiment_dir,
            run_number=run_number,
        )

        return (
            agent.name,
            agent.version,
            loop_details,
            evaluator_id,
            execution.datapoint.name,
            run_number,
            evaluation_criteria,
            agent_output,
            result,
            judge_model,
            judge_prompt,
        )

    def _build_result_data(
        self,
        agent_name: str,
        agent_version: str,
        loop_details: LoopDetails,
        evaluator_id: str,
        datapoint_name: str,
        run_number: int,
        evaluation_criteria: dict[str, Any] | None,
        agent_output: dict[str, Any] | str,
        result: EvaluationResult,
        judge_model: str | None,
        judge_prompt: str | None,
        split: str = "all_data",
    ) -> dict[str, Any]:
        """Wrap an evaluator result as an entry of :meth:`evaluate_one`."""
        result_data: dict[str, Any] = {
            "result": result,
            "evaluation_criteria": evaluation_criteria,
            "agent_output": agent_output,
            "agent_name": agent_name,
            "agent_version": agent_version,
            "loop_details": loop_details.model_dump(),
            "run_number": run_number,
            "split": split,
        }
        if judge_model:
            result_data["judge_model"] = judge_model
        if judge_prompt:
            result_data["judge_prompt"] = judge_prompt
        return {
            "agent_key": f"{agent_name}__{agent_version}",
            "loop_key": loop_details.loop_key,
            "evaluator_id": evaluator_id,
            "datapoint_key": f"{datapoint_name}:run_{run_number}",
            "result_data": result_data,
        }

    async def evaluate_one(
        self,
        execution: AgentDatapointExecution,
        agent: Agent,
    ) -> list[dict[str, Any]]:
        """Run each of the datapoint's evaluators, reusing saved results the mode allows.

        Returns entries with ``agent_key``, ``loop_key``, ``evaluator_id``, ``datapoint_key``
        and ``result_data``.
        """
        loop_details = execution.loop_details
        if not loop_details:
            return []

        evaluators = self._get_evaluators(agent.name)

        experiment_dir = get_experiment_path(
            base_output_dir=self.config.base_output_dir,
            experiment_name=self.config.experiment_name,
            agent=agent,
            loop_details=loop_details,
        )

        results: list[dict[str, Any]] = []
        run_number = execution.run_number
        datapoint_name = execution.datapoint.name
        agent_key = f"{agent.name}__{agent.version}"
        loop_key = loop_details.loop_key
        datapoint_key = f"{datapoint_name}:run_{run_number}"

        for evaluator_id in execution.datapoint.evaluation_criterias:
            if evaluator_id not in evaluators:
                continue

            evaluator = evaluators[evaluator_id]

            # A datapoint whose agent ran in this invocation is always evaluated again.
            if not getattr(execution, "ran", False) and not should_run_evaluator(
                experiment_dir,
                datapoint_name,
                evaluator_id,
                self.config.mode,
                run_number,
                execution.datapoint.evaluation_criterias.keys(),
            ):
                cached_result = self._try_load_cached_result(
                    experiment_dir, datapoint_name, evaluator, agent.name, run_number
                )
                if cached_result:
                    cached_result["agent_name"] = agent.name
                    cached_result["agent_version"] = agent.version
                    cached_result["loop_details"] = loop_details.model_dump()
                    cached_result["run_number"] = run_number
                    # A result file that does not record its split takes the agent's.
                    cached_result.setdefault(
                        "split", agent.datapoint_splits.get(datapoint_name, "all_data")
                    )
                    results.append(
                        {
                            "agent_key": agent_key,
                            "loop_key": loop_key,
                            "evaluator_id": evaluator_id,
                            "datapoint_key": datapoint_key,
                            "result_data": cached_result,
                        }
                    )
                    continue
                # A saved result that cannot be read is evaluated again.

            (
                agent_name,
                agent_version,
                ret_loop_details,
                eval_id,
                dp_name,
                ret_run_number,
                evaluation_criteria,
                agent_output,
                result,
                judge_model,
                judge_prompt,
            ) = await self._run_evaluator_task(agent, evaluator, execution, loop_details)

            results.append(
                self._build_result_data(
                    agent_name,
                    agent_version,
                    ret_loop_details,
                    eval_id,
                    dp_name,
                    ret_run_number,
                    evaluation_criteria,
                    agent_output,
                    result,
                    judge_model,
                    judge_prompt,
                    split=agent.datapoint_splits.get(dp_name, "all_data"),
                )
            )

        return results

    async def run(self) -> dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]]:
        """Evaluate all executions; returns agent_key -> loop_key -> evaluator_id ->
        datapoint_key -> result_data."""
        print("\nRunning evaluations...")
        if self.config.verbose:
            info(
                "Agents executed successfully",
                num_datapoints=len(self.executions),
                num_agents=len(self.loaded_agents),
            )

        all_evaluation_tasks, skipped_results, total_evaluations, skipped_evaluations = (
            self._build_evaluation_tasks()
        )

        if all_evaluation_tasks:
            evaluation_results = await gather_limited(
                max_concurrency=self.config.max_concurrency,
                items=all_evaluation_tasks,
                func=self._run_evaluator_task,
                total=total_evaluations,
                initial=skipped_evaluations,
                show_progress=True,
            )

            combined_results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = (
                defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
            )
            agent_splits_lookup: dict[tuple[str, str], dict[str, str]] = {
                (a.name, a.version): a.datapoint_splits for a in self.loaded_agents
            }

            for (
                agent_name,
                agent_version,
                loop_details,
                evaluator_id,
                datapoint_name,
                run_number,
                evaluation_criteria,
                agent_output,
                result,
                judge_model,
                judge_prompt,
            ) in evaluation_results:
                split = agent_splits_lookup.get((agent_name, agent_version), {}).get(
                    datapoint_name, "all_data"
                )
                result_data: dict[str, Any] = {
                    "result": result,
                    "evaluation_criteria": evaluation_criteria,
                    "agent_output": agent_output,
                    "agent_name": agent_name,
                    "agent_version": agent_version,
                    "loop_details": loop_details.model_dump(),
                    "run_number": run_number,
                    "split": split,
                }
                if judge_model:
                    result_data["judge_model"] = judge_model
                if judge_prompt:
                    result_data["judge_prompt"] = judge_prompt
                datapoint_key = f"{datapoint_name}:run_{run_number}"
                agent_key = f"{agent_name}__{agent_version}"
                loop_key = loop_details.loop_key
                combined_results[agent_key][loop_key][evaluator_id][datapoint_key] = result_data
        else:
            if self.config.verbose:
                info("No evaluations to run (all already completed)", mode=self.config.mode)
            if total_evaluations > 0:
                pbar = tqdm(total=total_evaluations, initial=total_evaluations)
                pbar.close()
            combined_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

        for agent_key, loop_results in skipped_results.items():
            for loop_key, evaluator_results in loop_results.items():
                for evaluator_id, datapoint_results in evaluator_results.items():
                    for datapoint_name, result_data in datapoint_results.items():
                        combined_results[agent_key][loop_key][evaluator_id][datapoint_name] = (
                            result_data
                        )

        return combined_results
