"""Agent setup and graph generation for evaluation runs."""

import logging
from pathlib import Path
from typing import Any

from agents_gym_core.hydra_agent import CoreAgentRegistry
from agents_gym_core.loops import LoopName
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION
from agents_gym_core.runtime.experiment_generator import agent_graphs_with_datapoints
from agents_gym_core.runtime.run_utils import GraphDatapoint
from agents_gym_core.runtime.runtime_config import RuntimeConfig
from agents_gym_core.runtime.runtime_helpers import RuntimeHelpers
from agents_gym_core.utils.gym_types import Agent

logger = logging.getLogger(__name__)


class AgentSetup:
    """Handles agent loading, default resolution, and graph generation."""

    def __init__(
        self,
        agents_spec: list[tuple[str, str]],
        registry: CoreAgentRegistry,
        config: RuntimeConfig,
        loops: list[tuple[str, str]] | None = None,
    ):
        """``agents_spec`` and ``loops`` are (name, version) pairs; ``loops`` defaults to
        ``[("basic", DEFAULT_LOOP_VERSION)]``."""
        self.agents_spec = agents_spec
        self.registry = registry
        self.loops = loops or [(LoopName.BASIC, DEFAULT_LOOP_VERSION)]
        self.config = config
        self.helpers = RuntimeHelpers(config)
        self.load_errors: list[Exception] = []

    def load_agents(self) -> list[Agent]:
        """Load the agents through the registry; agents that fail to load are logged and skipped,
        and their errors kept in ``load_errors``."""
        loaded: list[Agent] = []
        failed: list[tuple[str, str, Exception]] = []
        for agent_name, agent_version in self.agents_spec:
            try:
                agent = self.registry.get_agent(agent_name, agent_version=agent_version)
                loaded.append(agent)
            except Exception as e:
                failed.append((agent_name, agent_version, e))
        if failed:
            logger.warning(f"{len(failed)} agent(s) failed to load:")
            for name, version, error in failed:
                logger.warning(f"  - {name}:{version} — {error}")
        self.load_errors = [error for _, _, error in failed]
        return loaded

    def generate_graphs(
        self, loaded_agents: list[Agent], num_runs: int = 1
    ) -> tuple[list[GraphDatapoint], int]:
        """Build the graph datapoints of every model config and their total count.

        Each model config gets its own graphs; its loop_key encodes the models the loop uses.
        A config identical to an earlier one is skipped, and so is a graph whose loop_key an
        earlier config produced for the same agent (a role the loop does not use leaves the
        key unchanged).
        """
        per_model_graphs: list[list[GraphDatapoint]] = []
        total_datapoints = 0

        seen_configs: list[dict[str, Any]] = []
        seen_loop_keys: set[tuple[str, str, str]] = set()
        for models_config in self.config.models:
            if models_config in seen_configs:
                continue
            # A copy, as agent_graphs_with_datapoints adds the default embeddings to the config.
            seen_configs.append(dict(models_config))
            graphs_generator, num_datapoints = agent_graphs_with_datapoints(
                agents=loaded_agents,
                loops=self.loops,
                models=models_config,
                max_datapoints=self.config.max_datapoints,
                print_trace=self.config.print_trace,
                parallel_tool_calls=self.config.parallel_tool_calls,
                num_runs=num_runs,
                handle_tool_errors=self.config.handle_tool_errors,
                settings=self.config.settings,
                datapoint_names=self.config.datapoint_names,
                splits=self.config.splits,
            )
            graphs: list[GraphDatapoint] = []
            config_loop_keys: set[tuple[str, str, str]] = set()
            for graph_datapoint in graphs_generator:
                graph, _datapoint, agent, _run = graph_datapoint
                loop_key = graph.loop_details.loop_key  # pyright: ignore[reportAttributeAccessIssue]
                key = (agent.name, agent.version, loop_key)
                if key in seen_loop_keys:
                    num_datapoints -= 1
                    continue
                config_loop_keys.add(key)
                graphs.append(graph_datapoint)
            seen_loop_keys |= config_loop_keys
            per_model_graphs.append(graphs)
            total_datapoints += num_datapoints

        # Interleave the model configs round-robin so concurrent execution works on all of
        # them at once instead of finishing one before starting the next.
        all_graphs: list[GraphDatapoint] = []
        if per_model_graphs:
            max_len = max(len(g) for g in per_model_graphs)
            for i in range(max_len):
                for model_graphs in per_model_graphs:
                    if i < len(model_graphs):
                        all_graphs.append(model_graphs[i])

        return all_graphs, total_datapoints

    def compute_inventory_targets(
        self, graphs_list: list[GraphDatapoint]
    ) -> tuple[list[Path], list[Agent], list[list[str]]]:
        """Experiment dirs, agents and selected datapoint names for the inventory, one per
        (agent, version, loop_key)."""
        graphs_without_run = [(g, d, a) for g, d, a, _run in graphs_list]
        return self.helpers.compute_inventory_targets(graphs_without_run)
