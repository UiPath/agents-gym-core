"""Registry for discovering and loading agents from Hydra configuration."""

import os
import threading
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from omegaconf import DictConfig, OmegaConf
from uipath.eval.models.evaluation_set import EvaluationSet

from agents_gym_core.hydra_agent.evaluation_loader import EvaluationLoader
from agents_gym_core.hydra_agent.schema_builder import SchemaBuilder
from agents_gym_core.hydra_agent.tool_instantiator import ToolInstantiator
from agents_gym_core.loops import DEFAULT_LOOP_VERSION, get_uncompiled_loop
from agents_gym_core.models.chat import OpenAIModels, get_chat_model
from agents_gym_core.utils.directory_provider import CoreDirectoryProvider, DirectoryProvider
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.prompt_loader import PromptLoader


class CoreAgentRegistry:
    """Registry for discovering and loading agents."""

    load_evaluations: bool = True

    # Resolved base configs keyed by agent directory. Stored as plain dicts so each caller
    # gets its own DictConfig and one agent's mutations never reach another.
    _hydra_config_cache: dict[str, dict] = {}
    _hydra_lock = threading.Lock()

    def __init__(
        self, *, load_evaluations: bool = True, directory_provider: DirectoryProvider | None = None
    ):
        self.core_directory_provider = directory_provider or CoreDirectoryProvider()
        self.load_evaluations = load_evaluations

    def get_agent(
        self, agent_name: str, agent_version: str = "base", *, tools: list[BaseTool] | None = None
    ):
        """Load ``agent_name`` at ``agent_version`` (``"base"`` or a ``versions.yaml`` entry).

        The first provider from ``_get_providers()`` that has the agent supplies the base
        config; the first whose ``versions.yaml`` has the version supplies the override.
        """
        providers = self._get_providers()

        base_provider = next((p for p in providers if p._agent_exists(agent_name)), None)
        if base_provider is None:
            searched = " or ".join(type(p).__name__ for p in providers)
            raise FileNotFoundError(f"Agent '{agent_name}' not found in {searched}")

        base_dir = base_provider.get_agent_dir(agent_name)
        # Result keys are ``<agent>__<version>``, split at the first ``__``.
        if "__" in base_dir.name:
            raise ValueError(
                f"Agent name '{base_dir.name}' contains '__', which separates the agent from "
                f"its version in result keys. Rename the agent directory."
            )
        base_cfg = CoreAgentRegistry._load_agent_base_config(agent_name, base_provider)

        version_cfg = None
        if agent_version not in (None, "", "base"):
            for provider in providers:
                version_cfg = CoreAgentRegistry._load_agent_config_version(
                    agent_name, agent_version, provider, raise_on_missing=False
                )
                if version_cfg is not None:
                    break
            if version_cfg is None:
                raise FileNotFoundError(
                    f"Version '{agent_version}' not found for agent '{agent_name}'"
                )

        # Struct mode off, so a version can add keys absent from agent.yaml (e.g. signature_demos).
        if version_cfg is None:
            merged_cfg = base_cfg
        else:
            OmegaConf.set_struct(base_cfg, False)
            merged_cfg = OmegaConf.merge(base_cfg, version_cfg)

        if not isinstance(merged_cfg, DictConfig):
            merged_cfg = OmegaConf.create(OmegaConf.to_container(merged_cfg, resolve=True))
        if not isinstance(merged_cfg, DictConfig):
            raise TypeError("Merged config is not a DictConfig")

        # A version that leaves ``version: base`` would overwrite the base output directory.
        if agent_version not in (None, "", "base"):
            merged_version = merged_cfg.get("version", "base")
            if merged_version == "base":
                raise ValueError(
                    f"Version '{agent_version}' for agent '{agent_name}' does not set "
                    f"a 'version' field in versions.yaml. Add 'version: {agent_version}' "
                    f"to the '{agent_version}' entry to avoid output directory collisions."
                )

        return self._build_agent(
            merged_cfg,
            base_dir.name,
            self.get_directory_provider(),
            tools=tools,
            load_evaluations=self.load_evaluations,
        )

    @staticmethod
    def _load_agent_config_version(
        agent_name: str,
        agent_version: str,
        directory_provider: DirectoryProvider,
        raise_on_missing: bool = True,
    ):
        """Return the provider's ``versions.yaml`` entry for ``agent_version``, minus ``defaults``.

        ``None`` for the base version. A missing agent directory, file or entry raises, or
        returns ``None`` when ``raise_on_missing`` is False.
        """
        if agent_version in (None, "", "base"):
            return None

        try:
            agent_dir = directory_provider.get_agent_dir(agent_name)
        except FileNotFoundError:
            if raise_on_missing:
                raise
            return None

        versions_file = agent_dir / "versions.yaml"

        if not versions_file.exists():
            if raise_on_missing:
                raise FileNotFoundError(
                    f"Version '{agent_version}' requested but versions.yaml not found at {versions_file}"
                )
            return None

        versions_cfg = OmegaConf.load(versions_file)
        if not isinstance(versions_cfg, DictConfig):
            versions_cfg = OmegaConf.create(OmegaConf.to_container(versions_cfg, resolve=True))
        if not isinstance(versions_cfg, DictConfig):
            if raise_on_missing:
                raise TypeError(
                    f"{versions_file} must contain a mapping of version -> override config"
                )
            return None

        if agent_version not in versions_cfg:
            if raise_on_missing:
                raise FileNotFoundError(
                    f"Version '{agent_version}' not found in {versions_file}. "
                    f"Available versions: {list(versions_cfg.keys())}"
                )
            return None

        version_cfg = versions_cfg.get(agent_version)
        if version_cfg is None:
            return None
        if not isinstance(version_cfg, DictConfig):
            version_cfg = OmegaConf.create(OmegaConf.to_container(version_cfg, resolve=True))
        return CoreAgentRegistry._strip_defaults(version_cfg)

    @staticmethod
    def _load_agent_base_config(
        agent_name: str,
        directory_provider: DirectoryProvider,
    ) -> DictConfig:
        """Compose the provider's ``agent.yaml`` with Hydra, caching the result per directory.

        The cache skips the GlobalHydra clear, initialize and compose cycle on repeated loads.

        Raises:
            FileNotFoundError: If ``agent.yaml`` does not exist.
        """
        agent_dir = directory_provider.get_agent_dir(agent_name)
        config_file = agent_dir / "agent.yaml"

        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file not found for agent '{agent_name}': {config_file}"
            )

        cache_key = str(agent_dir.resolve())

        if cache_key in CoreAgentRegistry._hydra_config_cache:
            return OmegaConf.create(CoreAgentRegistry._hydra_config_cache[cache_key])

        with CoreAgentRegistry._hydra_lock:
            if cache_key in CoreAgentRegistry._hydra_config_cache:
                return OmegaConf.create(CoreAgentRegistry._hydra_config_cache[cache_key])

            GlobalHydra.instance().clear()

            initialize_config_dir(config_dir=str(agent_dir.resolve()), version_base=None)
            cfg = compose(config_name="agent")

            resolved_config = OmegaConf.to_container(cfg, resolve=True)
            if isinstance(resolved_config, dict):
                CoreAgentRegistry._hydra_config_cache[cache_key] = resolved_config

        return cfg

    @staticmethod
    def _strip_defaults(cfg: object | None):
        """Drop a ``defaults`` key, which is Hydra composition syntax rather than config data."""
        if cfg is None:
            return None
        if isinstance(cfg, DictConfig) and "defaults" in cfg:
            return OmegaConf.create({k: v for k, v in cfg.items() if k != "defaults"})
        return cfg

    @staticmethod
    def _build_agent(
        cfg: DictConfig,
        agent_name: str,
        directory_provider: DirectoryProvider,
        *,
        tools: list[BaseTool] | None = None,
        load_evaluations: bool = True,
    ) -> Agent:
        """Construct an Agent from a loaded config."""
        if load_evaluations:
            labelled_eval_set = EvaluationLoader.load_evaluation_set(
                cfg, agent_name, directory_provider
            )
            evaluation_set = labelled_eval_set.evaluation_set
            datapoint_splits = labelled_eval_set.datapoint_splits
        else:
            from agents_gym_core.hydra_agent.agent_init import AgentInit

            if AgentInit.has_init(cfg):
                AgentInit.check_or_raise(
                    cfg,
                    agent_name,
                    directory_provider.get_agent_dir(agent_name),
                    directory_provider,
                )
            evaluation_set = EvaluationSet(id=agent_name, name=agent_name)
            datapoint_splits = {}
        agent_version = getattr(cfg, "version", "base")
        tools = ToolInstantiator.instantiate_tools(cfg.agent) if tools is None else list(tools)
        input_schema, output_schema = SchemaBuilder.create_input_output_schemas(cfg)

        # dspy-program agents may omit system_prompt: their instructions live in the DSPy
        # module's signatures.
        if hasattr(cfg.agent, "system_prompt"):
            system_prompt = PromptLoader.load_prompt_from_file(
                cfg.agent.system_prompt, agent_name, directory_provider
            )
        else:
            system_prompt = ""
        user_prompt = PromptLoader.load_prompt_from_file(
            cfg.agent.user_prompt, agent_name, directory_provider
        )

        evaluator_weights: dict[str, float] = {}
        if hasattr(cfg.agent, "evaluator_weights"):
            raw_weights = OmegaConf.to_container(cfg.agent.evaluator_weights, resolve=True)
            if isinstance(raw_weights, dict):
                evaluator_weights = {str(k): float(v) for k, v in raw_weights.items()}

        tags: list[str] = []
        if hasattr(cfg, "metadata") and hasattr(cfg.metadata, "tags"):
            raw_tags = OmegaConf.to_container(cfg.metadata.tags, resolve=True)
            if isinstance(raw_tags, list):
                tags = [str(tag) for tag in raw_tags]

        # signature_prompts and signature_demos are used only by the dspy-program loop.
        signature_prompts: dict[str, str] = {}
        if hasattr(cfg.agent, "signature_prompts"):
            raw_sig = OmegaConf.to_container(cfg.agent.signature_prompts, resolve=True)
            if isinstance(raw_sig, dict):
                for field_name, prompt_ref in raw_sig.items():
                    signature_prompts[str(field_name)] = PromptLoader.load_prompt_from_file(
                        str(prompt_ref), agent_name, directory_provider
                    )

        signature_demos: dict[str, list[dict]] = {}
        if hasattr(cfg.agent, "signature_demos"):
            import json

            raw_demos = OmegaConf.to_container(cfg.agent.signature_demos, resolve=True)
            if isinstance(raw_demos, dict):
                for field_name, demo_ref in raw_demos.items():
                    demo_path = directory_provider.solve_path(agent_name, str(demo_ref))
                    with open(demo_path) as df:
                        payload = json.load(df)
                    demos = payload.get("demos", []) if isinstance(payload, dict) else payload
                    signature_demos[str(field_name)] = demos

        return Agent(
            name=agent_name,
            version=str(agent_version),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            input_schema=input_schema,
            output_schema=output_schema,
            tools=tools,
            evaluation_set=evaluation_set,
            evaluator_weights=evaluator_weights,
            tags=tags,
            datapoint_splits=datapoint_splits,
            loop=str(cfg.loop) if hasattr(cfg, "loop") else "",
            signature_prompts=signature_prompts,
            signature_demos=signature_demos,
        )

    def _get_providers(self) -> list[DirectoryProvider]:
        """Return directory providers to search. Subclasses override to add more."""
        return [self.core_directory_provider]

    def list_all_agents(self) -> list[str]:
        """Return all agent names (deduplicated across providers)."""
        agents: set[str] = set()
        for provider in self._get_providers():
            agents_dir = provider.get_agents_dir()
            if agents_dir.exists():
                agents.update(
                    d.name
                    for d in agents_dir.iterdir()
                    if d.is_dir() and (d / "agent.yaml").exists()
                )
        return sorted(agents)

    def list_agent_versions(self, agent_name: str) -> list[str]:
        """Return the agent's version names across providers, always including ``"base"``."""
        versions: set[str] = {"base"}
        for provider in self._get_providers():
            if provider._version_exists(agent_name):
                versions_file = provider.get_agent_dir(agent_name) / "versions.yaml"
                cfg = OmegaConf.load(versions_file)
                if isinstance(cfg, DictConfig):
                    versions.update(str(k) for k in cfg if k != "base")
        return sorted(versions)

    def list_agents_by_tag(self, tag: str) -> list[tuple[str, str]]:
        """Return ``(agent_name, "base")`` for every agent whose metadata tags include *tag*.

        A provider whose ``get_implicit_tag()`` is *tag* contributes all of its agents. On
        duplicate names the first provider wins.
        """
        seen: set[str] = set()
        results: list[tuple[str, str]] = []
        for provider in self._get_providers():
            if provider.get_implicit_tag() == tag:
                for name, ver in self._list_all_agents_in_dir(provider.get_agents_dir()):
                    if name not in seen:
                        seen.add(name)
                        results.append((name, ver))
            else:
                for name, ver in self._scan_directory_for_tag(provider.get_agents_dir(), tag):
                    if name not in seen:
                        seen.add(name)
                        results.append((name, ver))
        return results

    @staticmethod
    def _list_all_agents_in_dir(agents_dir: Path) -> list[tuple[str, str]]:
        """Return (agent_name, "base") for every agent in a directory."""
        if not agents_dir.exists():
            return []
        results: list[tuple[str, str]] = []
        for agent_dir in sorted(agents_dir.iterdir()):
            if agent_dir.is_dir() and (agent_dir / "agent.yaml").exists():
                results.append((agent_dir.name, "base"))
        return results

    @staticmethod
    def _scan_directory_for_tag(agents_dir: Path, tag: str) -> list[tuple[str, str]]:
        """Scan an agents directory and return agents matching a tag."""
        if not agents_dir.exists():
            return []
        results: list[tuple[str, str]] = []
        for agent_dir in sorted(agents_dir.iterdir()):
            config_file = agent_dir / "agent.yaml"
            if not config_file.exists():
                continue
            cfg = OmegaConf.load(config_file)
            raw_tags = OmegaConf.select(cfg, "metadata.tags", default=None)
            if raw_tags is None:
                continue
            tags = OmegaConf.to_container(raw_tags, resolve=True)
            if isinstance(tags, list) and tag in tags:
                results.append((agent_dir.name, "base"))
        return results

    def get_directory_provider(self) -> DirectoryProvider:
        return self.core_directory_provider

    async def create_uncompiled_graph(
        self, agent_name: str, agent_version: str = "base"
    ) -> StateGraph:
        """Return the agent's uncompiled loop graph, for SDK entrypoints to compile.

        The loop is the agent's own ``loop:``, else ``basic``. The model is
        ``$GYM_DEFAULT_MODEL``, else gpt-4o-2024-11-20.
        """
        agent_scenario = self.get_agent(agent_name, agent_version)
        model_name = os.environ.get("GYM_DEFAULT_MODEL") or OpenAIModels.gpt_4o_2024_11_20.name
        loop_name, loop_version = "basic", DEFAULT_LOOP_VERSION
        if agent_scenario.loop:
            name, _, own_version = agent_scenario.loop.partition(":")
            loop_name, loop_version = name, own_version or DEFAULT_LOOP_VERSION
        return get_uncompiled_loop(
            loop_name=loop_name,
            version=loop_version,
            scenario=agent_scenario,
            models={"model": get_chat_model(model_name)},
            print_trace=False,
            parallel_tool_calls=False,
        )
