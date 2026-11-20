"""Experiment orchestrator coordinating all phases of evaluation runs."""

import json
import logging
from pathlib import Path
from typing import Any

from agents_gym_core.hydra_agent import CoreAgentRegistry
from agents_gym_core.models.chat import OpenAIModels
from agents_gym_core.runtime.run_utils import (
    AggregationMode,
    RunMode,
    get_experiment_path,
    print_experiment_inventory,
)
from agents_gym_core.runtime.runners.agent_setup import AgentSetup
from agents_gym_core.runtime.runners.result_aggregator import ResultAggregator
from agents_gym_core.runtime.runners.streaming_runner import run_and_evaluate_all
from agents_gym_core.runtime.runtime_config import RuntimeConfig
from agents_gym_core.utils.gym_settings_core import GymSettingsCore
from agents_gym_core.utils.gym_types import Agent

_logger = logging.getLogger(__name__)


class ExperimentOrchestrator:
    """Runs an experiment: agent setup, execution, evaluation and result aggregation."""

    def __init__(
        self,
        agents: list[tuple[str, str]],
        registry: CoreAgentRegistry,
        loops: list[tuple[str, str]] | None = None,
        models: list[dict[str, str]] = [{"model": OpenAIModels.gpt_4o_2024_11_20.name}],
        max_concurrency: int = 20,
        verbose: bool = False,
        max_datapoints: int = 0,
        print_trace: bool = False,
        parallel_tool_calls: bool = False,
        base_output_dir: Path | None = None,
        experiment_name: str = "experiment",
        mode: RunMode = "fill",
        num_runs: int = 1,
        aggregation_mode: AggregationMode = AggregationMode.DATAPOINT_RUN,
        handle_tool_errors: bool = False,
        save_traces: bool = True,
        settings: GymSettingsCore | None = None,
        datapoint_names: list[str] | None = None,
        splits: list[str] | None = None,
    ):
        """Build the RuntimeConfig of the run.

        Args:
            agents: (agent_name, agent_version) pairs.
            loops: (loop_name, version) pairs; defaults to ``[("basic", DEFAULT_LOOP_VERSION)]``.
            models: One model config per set of graphs, e.g. ``[{"model": "gpt-4o-2024-11-20"},
                {"model": "gpt-4.1-mini", "planning_llm": "gpt-4o"}]``.
            base_output_dir: Defaults to ``./experiments``.
            num_runs: Runs per datapoint, saved in ``run_1/``, ``run_2/``, ...
            datapoint_names: Exact datapoint names to run (None = all); overrides ``splits``.
            splits: Split labels to run, e.g. ``["train"]``, from eval-set filename suffixes.
        """
        self.agents = agents
        self.loops = loops
        self.registry = registry

        if not base_output_dir:
            base_output_dir = Path("./experiments")

        self.config = RuntimeConfig(
            models=models,
            max_concurrency=max_concurrency,
            max_datapoints=max_datapoints,
            datapoint_names=datapoint_names,
            splits=splits,
            verbose=verbose,
            print_trace=print_trace,
            parallel_tool_calls=parallel_tool_calls,
            base_output_dir=base_output_dir,
            experiment_name=experiment_name,
            mode=mode,
            num_runs=num_runs,
            aggregation_mode=aggregation_mode,
            handle_tool_errors=handle_tool_errors,
            save_traces=save_traces,
            settings=settings,
        )

    async def run_async(self) -> dict[str, Any]:
        """Load the agents, run and evaluate every datapoint, then aggregate and save.

        Returns the dict of :meth:`ResultAggregator.finalize`; its ``evaluation_results`` map
        agent_key -> loop_key -> evaluator_id -> datapoint_key -> result_data,
        ``failed_datapoint_runs`` lists the datapoint runs that raised and are not scored,
        ``loaded_agents`` the agents that loaded and ``agent_load_errors`` the errors of those
        that did not.
        """
        setup = AgentSetup(
            agents_spec=self.agents,
            registry=self.registry,
            loops=self.loops,
            config=self.config,
        )
        loaded_agents = setup.load_agents()
        graphs_list, num_datapoints = setup.generate_graphs(loaded_agents, self.config.num_runs)
        all_datapoint_names = [datapoint.name for _, datapoint, _, _ in graphs_list]

        if all_datapoint_names and loaded_agents:
            experiment_dirs, agents_to_inventory, datapoint_names = setup.compute_inventory_targets(
                graphs_list
            )
            if experiment_dirs:
                print_experiment_inventory(
                    experiment_dirs,
                    agents_to_inventory,
                    self.config.max_datapoints,
                    datapoint_names=datapoint_names,
                    mode=self.config.mode,
                    num_runs=self.config.num_runs,
                )

        self._save_agent_details(graphs_list)

        failed: list[str] = []
        evaluation_results = await run_and_evaluate_all(
            graphs_list=graphs_list,
            registry=self.registry,
            num_datapoints=num_datapoints,
            config=self.config,
            failed=failed,
        )

        aggregator = ResultAggregator(
            graphs_list=graphs_list,
            all_datapoint_names=all_datapoint_names,
            loaded_agents=loaded_agents,
            config=self.config,
        )

        aggregation_output_dir = self.config.base_output_dir / self.config.experiment_name

        output = aggregator.finalize(
            evaluation_results,
            agg=self.config.aggregation_mode,
            output_dir=aggregation_output_dir,
        )
        output["failed_datapoint_runs"] = failed
        output["loaded_agents"] = loaded_agents
        output["agent_load_errors"] = setup.load_errors
        return output

    def _save_agent_details(self, graphs_list: list) -> None:
        """Write agent_details.json once per experiment directory."""
        saved: set[str] = set()
        for graph, _datapoint, agent, _run in graphs_list:
            loop_details = getattr(graph, "loop_details", None)
            if not loop_details:
                continue
            experiment_dir = get_experiment_path(
                self.config.base_output_dir,
                self.config.experiment_name,
                agent,
                loop_details,
            )
            key = str(experiment_dir)
            if key in saved:
                continue
            saved.add(key)
            try:
                experiment_dir.mkdir(parents=True, exist_ok=True)
                details = _build_agent_details(agent, loop_details)
                (experiment_dir / "agent_details.json").write_text(
                    json.dumps(details, indent=2, default=str) + "\n"
                )
            except Exception as e:
                _logger.warning(f"Failed to save agent_details.json: {e}")


def _build_agent_details(agent: Agent, loop_details: Any) -> dict[str, Any]:
    """JSON-serializable summary of the agent's configuration."""
    input_fields = {}
    for name, field_info in agent.input_schema.model_fields.items():
        input_fields[name] = {
            "type": str(field_info.annotation),
            "description": field_info.description or "",
        }

    output_fields = {}
    for name, field_info in agent.output_schema.model_fields.items():
        output_fields[name] = {
            "type": str(field_info.annotation),
            "description": field_info.description or "",
        }

    tools_info = []
    for tool in agent.tools:
        tools_info.append(
            {
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", ""),
            }
        )

    return {
        "name": agent.name,
        "version": agent.version,
        "loop": agent.loop,
        "loop_details": {
            "loop_name": loop_details.loop_name if loop_details else "",
            "loop_version": getattr(loop_details, "loop_version", ""),
            "models": getattr(loop_details, "models", {}),
        },
        "tags": agent.tags,
        "system_prompt": agent.system_prompt,
        "user_prompt": agent.user_prompt,
        "signature_prompts": agent.signature_prompts,
        "input_schema": input_fields,
        "output_schema": output_fields,
        "tools": tools_info,
        "evaluator_weights": agent.evaluator_weights,
        "evaluation_set_name": getattr(agent.evaluation_set, "name", "")
        if agent.evaluation_set
        else "",
        "evaluation_set_id": getattr(agent.evaluation_set, "id", "")
        if agent.evaluation_set
        else "",
        "num_datapoints": len(agent.evaluation_set.evaluations) if agent.evaluation_set else 0,
    }
