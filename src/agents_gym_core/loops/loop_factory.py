"""Loop factory for creating different agent loop variants.

Loops can be specified in two ways:

1. **By name** — a short identifier registered in :class:`LoopName`::

       --loops basic:any

2. **By Python path** — a fully-qualified dotted path to a ``BasicLoop``
   subclass (or factory function), analogous to the ``_target_`` pattern
   used for tools in ``agent.yaml``::

       --loops my_package.loops.MyLoop

   Path-based loops require no registration in the ``LoopName`` enum —
   they are imported and instantiated directly.  This is useful for
   experiment-specific loops that shouldn't pollute the global registry.
"""

import inspect
import logging
from importlib import import_module
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from uipath_langchain.agent.react.types import AgentGraphConfig

from agents_gym_core.loops.basic_loop import BasicLoop
from agents_gym_core.loops.deep_agents_loop import (
    create_deep_agents_graph,
    create_deep_agents_uncompiled_graph,
)
from agents_gym_core.loops.dspy_program_loop import (
    create_dspy_program_graph,
    create_dspy_program_uncompiled_graph,
)
from agents_gym_core.loops.types import (
    DEFAULT_LOOP_VERSION,
    LoopDetails,
    LoopName,
)
from agents_gym_core.models.utils import get_model_name
from agents_gym_core.utils.gym_settings_core import GymSettingsCore, gym_settings_core
from agents_gym_core.utils.gym_types import Agent

logger = logging.getLogger(__name__)


def _is_dotted_path(name: str) -> bool:
    """Return True if *name* looks like a fully-qualified Python path."""
    return "." in name


def _import_loop_class(dotted_path: str) -> Any:
    """Import a loop class or graph factory from a fully-qualified Python path.

    The path should be ``module.path.ClassName``, e.g.
    ``my_package.loops.MyLoop``.

    Returns the class or callable found at that path.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the class does not exist in the module.
    """
    module_path, _, class_name = dotted_path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid loop path {dotted_path!r}: expected 'module.path.ClassName'")
    module = import_module(module_path)
    return getattr(module, class_name)


def _resolve_dotted_path_loop(
    dotted_path: str,
    version: str,
    scenario: Agent,
    model: "BaseChatModel",
    models: dict[str, "BaseChatModel | Embeddings"],
    config: "AgentGraphConfig",
    settings: "GymSettingsCore",
    **kwargs: Any,
) -> "BasicLoop | StateGraph":
    """Resolve *dotted_path* into a loop instance or an uncompiled graph.

    A target may be either a ``BasicLoop`` subclass or a graph factory. Graph
    factories receive the subset of standard factory arguments declared by
    their signature and must return an uncompiled ``StateGraph``.
    """
    target = _import_loop_class(dotted_path)
    if isinstance(target, type) and issubclass(target, BasicLoop):
        if version != target._loop_version:
            logger.warning(
                "Version %r passed for %s but the class declares _loop_version=%r. "
                "The class version is used for behaviour; %r is only recorded in LoopDetails.",
                version,
                dotted_path,
                target._loop_version,
                version,
            )
        return _create_loop_instance_from_class(
            target, dotted_path, version, scenario, model, models, config, settings, **kwargs
        )

    if not callable(target) or isinstance(target, type):
        raise TypeError(
            f"{dotted_path} resolved to {target!r}, which is neither a BasicLoop "
            "subclass nor a graph factory"
        )

    available_arguments: dict[str, Any] = {
        "scenario": scenario,
        "model": model,
        "models": models,
        "config": config,
        "settings": settings,
        **kwargs,
    }
    signature = inspect.signature(target)
    accepts_extra_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    factory_arguments = (
        available_arguments
        if accepts_extra_keywords
        else {
            name: value
            for name, value in available_arguments.items()
            if name in signature.parameters
        }
    )
    graph = target(**factory_arguments)
    if not isinstance(graph, StateGraph):
        raise TypeError(
            f"graph factory {dotted_path} returned {type(graph)!r}; expected an uncompiled StateGraph"
        )
    existing_details = getattr(graph, "loop_details", None)
    graph.loop_details = LoopDetails(  # pyright: ignore[reportAttributeAccessIssue]
        loop_name=dotted_path,
        loop_version=version,
        models=(
            existing_details.models
            if isinstance(existing_details, LoopDetails)
            else {"model": get_model_name(model)}
        ),
    )
    return graph


def get_default_graph_config(settings: GymSettingsCore) -> AgentGraphConfig:
    """Get the default AgentGraphConfig for loops.

    Note: is_conversational is set to False here. The factory will override it
    based on scenario tags in get_uncompiled_loop.
    """
    return AgentGraphConfig(
        llm_messages_limit=settings.UIPATH_AGENTS_GYM_LOOP_RECURSION_LIMIT,
        thinking_messages_limit=settings.UIPATH_AGENTS_GYM_NO_TOOL_MESSAGES_LIMIT,
        parallel_tool_calls=settings.UIPATH_AGENTS_GYM_PARALLEL_TOOL_CALLS,
        is_conversational=False,
    )


def get_loop(
    loop_name: str,
    version: str,
    scenario: Agent,
    models: dict[str, BaseChatModel | Embeddings],  # chat models + optional "embeddings" key
    config: AgentGraphConfig | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    settings: GymSettingsCore | None = None,
    **kwargs: Any,
) -> CompiledStateGraph:
    """Build and compile an agent loop, returning a ready-to-invoke ``CompiledStateGraph``.

    For BasicLoop-based loops this delegates to ``loop.build_compiled_graph()`` which
    correctly passes the memory store.  Graph factories (``dspy-program``, dotted-path
    factories) return a ``StateGraph`` that is compiled directly.

    ``loop_name`` may be a :class:`LoopName` member (e.g. ``"basic"``) **or** a
    fully-qualified Python path to a ``BasicLoop`` subclass (e.g.
    ``"my_package.loops.MyLoop"``).

    Args:
        loop_name: The loop type — a :class:`LoopName` value or a dotted Python path.
        version: The loop version (e.g. "auto", "any").
        scenario: The agent scenario
        models: Dict of model objects keyed by role (e.g. "model", "planning_llm", "embeddings").
            Values are ``BaseChatModel`` for chat roles and ``Embeddings`` for embedding roles.
        config: Optional AgentGraphConfig to control loop behaviour.
        checkpointer: Optional checkpointer (e.g. ``MemorySaver``). When provided,
            the compiled graph is NOT cached to avoid sharing state across executions.
        settings: Optional GymSettingsCore object to override default settings
        **kwargs: Additional arguments passed to the loop constructor

    Returns:
        A ``CompiledStateGraph`` with ``loop_details`` attached as an attribute.

    """
    settings = settings or gym_settings_core
    if config is None:
        config = get_default_graph_config(settings=settings)

    # Always override is_conversational based on scenario tags
    config.is_conversational = scenario.is_conversational

    model = models["model"]
    assert isinstance(model, BaseChatModel), f"'model' must be a BaseChatModel, got {type(model)}"

    # --- Dotted-path loops: import a BasicLoop class or graph factory directly ---
    if isinstance(loop_name, str) and _is_dotted_path(loop_name):
        resolved = _resolve_dotted_path_loop(
            loop_name, version, scenario, model, models, config, settings, **kwargs
        )
        if isinstance(resolved, BasicLoop):
            return resolved.build_compiled_graph(checkpointer=checkpointer)
        compiled = resolved.compile(checkpointer=checkpointer)
        compiled.loop_details = resolved.loop_details  # pyright: ignore[reportAttributeAccessIssue]
        return compiled

    # --- Named loops: convert string to LoopName enum ---
    if isinstance(loop_name, str):
        loop_name = LoopName(loop_name)

    # Deep Agents loop: the deepagents harness (todos, filesystem, sub-agents)
    if loop_name == LoopName.DEEP_AGENTS:
        _check_deep_agents_version(version)
        return create_deep_agents_graph(
            scenario, model, config=config, checkpointer=checkpointer, **kwargs
        )

    # Special case: DSPy program loop — version IS the dotted class path, e.g.:
    #   --loops dspy-program:gepa_artifact.benchmarks.hover.hover_program.HoverMultiHop
    if loop_name == LoopName.DSPY_PROGRAM:
        return create_dspy_program_graph(
            scenario,
            model,
            program_class=version,
            models=models,
            checkpointer=checkpointer,
            **kwargs,
        )

    # BasicLoop-based loops — use build_compiled_graph() which passes store=self.memory
    loop = _create_loop_instance(
        loop_name,
        version,
        scenario,
        model,
        models,
        config,
        settings,
        **kwargs,
    )
    return loop.build_compiled_graph(checkpointer=checkpointer)


def get_uncompiled_loop(
    loop_name: str,
    version: str,
    scenario: Agent,
    models: dict[str, BaseChatModel | Embeddings],
    config: AgentGraphConfig | None = None,
    settings: GymSettingsCore | None = None,
    **kwargs: Any,
) -> StateGraph:
    """Build an agent loop, returning an uncompiled ``StateGraph``.

    This is useful for SDK entrypoints where users need to compile the graph themselves.

    ``loop_name`` may be a :class:`LoopName` member **or** a fully-qualified
    Python path — see :func:`get_loop` for details.

    Args:
        loop_name: The loop type — a :class:`LoopName` value or a dotted Python path.
        version: The loop version (e.g. "auto", "any").
        scenario: The agent scenario
        models: Dict of model objects keyed by role (e.g. "model", "planning_llm", "embeddings").
            Values are ``BaseChatModel`` for chat roles and ``Embeddings`` for embedding roles.
        config: Optional AgentGraphConfig
        settings: Optional GymSettingsCore object to override default settings
        **kwargs: Additional arguments passed to the loop constructor

    Returns:
        An uncompiled ``StateGraph`` with ``loop_details`` attached as an attribute.

    """
    settings = settings or gym_settings_core
    if config is None:
        config = get_default_graph_config(settings=settings)

    # Always override is_conversational based on scenario tags
    config.is_conversational = scenario.is_conversational

    model = models["model"]
    assert isinstance(model, BaseChatModel), f"'model' must be a BaseChatModel, got {type(model)}"

    # --- Dotted-path loops: import a BasicLoop class or graph factory directly ---
    if isinstance(loop_name, str) and _is_dotted_path(loop_name):
        resolved = _resolve_dotted_path_loop(
            loop_name, version, scenario, model, models, config, settings, **kwargs
        )
        return resolved.build_graph() if isinstance(resolved, BasicLoop) else resolved

    # --- Named loops: convert string to LoopName enum ---
    if isinstance(loop_name, str):
        loop_name = LoopName(loop_name)

    # Deep Agents loop: the deepagents harness (todos, filesystem, sub-agents)
    if loop_name == LoopName.DEEP_AGENTS:
        _check_deep_agents_version(version)
        return create_deep_agents_uncompiled_graph(scenario, model, config=config, **kwargs)

    # Special case: DSPy program loop — version IS the dotted class path.
    if loop_name == LoopName.DSPY_PROGRAM:
        return create_dspy_program_uncompiled_graph(
            scenario, model, program_class=version, models=models, **kwargs
        )

    loop = _create_loop_instance(
        loop_name,
        version,
        scenario,
        model,
        models,
        config,
        settings,
        **kwargs,
    )
    return loop.build_graph()


def _create_loop_instance(
    loop_name: str,
    version: str,
    scenario: Agent,
    model: BaseChatModel,
    models: dict[str, BaseChatModel | Embeddings],
    config: AgentGraphConfig,
    settings: GymSettingsCore,
    **kwargs: Any,
) -> BasicLoop:
    """Create a BasicLoop (or subclass) instance with all model dependencies resolved."""
    loop_cls = _get_loop_class(loop_name, version)
    if loop_cls is None:
        raise ValueError(f"Unknown loop: {loop_name}:{version}")

    return _create_loop_instance_from_class(
        loop_cls, loop_name, version, scenario, model, models, config, settings, **kwargs
    )


def _create_loop_instance_from_class(
    loop_cls: type[BasicLoop],
    loop_name: str,
    version: str,
    scenario: Agent,
    model: BaseChatModel,
    models: dict[str, BaseChatModel | Embeddings],
    config: AgentGraphConfig,
    settings: GymSettingsCore,
    **kwargs: Any,
) -> BasicLoop:
    """Instantiate a BasicLoop subclass with all model dependencies resolved.

    Every model role other than ``model`` (``planning_llm``, ``embeddings``, ...) is passed
    to the constructor by name; a loop ignores the roles it does not take.

    When *loop_name* is a dotted path string, the resulting ``LoopDetails``
    stores it as-is so that experiment output directories and loop keys
    remain unique and traceable.
    """
    loop_kwargs: dict[str, Any] = {
        **{role: role_model for role, role_model in models.items() if role != "model"},
        **settings.UIPATH_AGENTS_GYM_SUMMARIZATION_SETTINGS,
        **kwargs,
    }
    instance = loop_cls(scenario=scenario, model=model, config=config, **loop_kwargs)

    # For dotted-path loops, override loop_name in details so experiment
    # output uses the dotted path (not the class's _loop_name which may
    # still reference a LoopName enum).
    if isinstance(loop_name, str) and _is_dotted_path(loop_name):
        instance.details = LoopDetails(
            loop_name=loop_name,
            loop_version=version,
            models=instance.details.models,
        )

    return instance


def _check_deep_agents_version(version: str) -> None:
    """``deep-agents`` has one version, ``auto``."""
    if version != DEFAULT_LOOP_VERSION:
        raise ValueError(f"Unknown loop: {LoopName.DEEP_AGENTS}:{version}")


def _get_loop_class(loop_name: str, version: str) -> type | None:
    """Get the loop class for a given LoopName and version.

    Discovers all BasicLoop subclasses automatically via __subclasses__.
    """

    def all_subclasses(cls: type) -> list[type]:
        subs = cls.__subclasses__()
        return subs + [s for c in subs for s in all_subclasses(c)]

    for cls in [BasicLoop, *all_subclasses(BasicLoop)]:
        if cls._loop_name == loop_name and cls._loop_version == version:
            return cls

    return None
