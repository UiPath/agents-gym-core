"""Standalone callable agents use the existing registry's loading and validation."""

from langchain.tools import BaseTool
from omegaconf import DictConfig

from agents_gym_core.hydra_agent.core_agent_registry import CoreAgentRegistry
from agents_gym_core.loops.callable_loop import CallableAgent
from agents_gym_core.utils.directory_provider import DirectoryProvider
from agents_gym_core.utils.gym_types import Agent


class ProjectAgentRegistry(CoreAgentRegistry):
    @staticmethod
    def _build_agent(
        cfg: DictConfig,
        agent_name: str,
        directory_provider: DirectoryProvider,
        *,
        tools: list[BaseTool] | None = None,
        load_evaluations: bool = True,
    ) -> CallableAgent:
        agent = CoreAgentRegistry._build_agent(
            cfg, agent_name, directory_provider, tools=tools, load_evaluations=load_evaluations
        )
        reference = str(cfg.agent.get("entrypoint", ""))
        if directory_provider.check_for_placeholder(reference):
            if ":" not in reference.partition("::")[2]:
                raise ValueError(
                    f"agent.entrypoint must be core::<file>.py:<function>, got {reference!r}"
                )
            path, _, function = reference.rpartition(":")
            reference = f"{directory_provider.solve_path(agent_name, path)}:{function}"
        return CallableAgent(
            **{name: getattr(agent, name) for name in Agent.model_fields}, entrypoint=reference
        )
