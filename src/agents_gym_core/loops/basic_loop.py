"""Basic agent loop implementation."""

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel
from uipath_langchain.agent.guardrails.actions import GuardrailAction
from uipath_langchain.agent.guardrails.guardrails_factory import BaseGuardrail
from uipath_langchain.agent.react.guardrails.guardrails_subgraph import (
    create_agent_init_guardrails_subgraph,
    create_agent_terminate_guardrails_subgraph,
    create_llm_guardrails_subgraph,
    create_tools_guardrails_subgraph,
)
from uipath_langchain.agent.react.init_node import (
    create_init_node,
)
from uipath_langchain.agent.react.router import (
    create_route_agent,
)
from uipath_langchain.agent.react.router_conversational import create_route_agent_conversational
from uipath_langchain.agent.react.terminate_node import (
    create_terminate_node,
)
from uipath_langchain.agent.react.tools import create_flow_control_tools
from uipath_langchain.agent.react.types import (
    AgentGraphConfig,
    AgentGraphNode,
)
from uipath_langchain.agent.react.utils import create_state_with_input

from agents_gym_core.common_tools.memory_tools import create_memory_tools
from agents_gym_core.loops.nodes import (
    create_custom_llm_node,
    create_messages,
    create_tool_node,
)
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION, LoopDetails, LoopName
from agents_gym_core.loops.utils import apply_tool_embeddings
from agents_gym_core.models.utils import get_model_name
from agents_gym_core.utils.gym_types import Agent


class BasicLoop:
    """Basic agent loop following UiPath SDK's create_agent pattern.

    Structure matches create_agent but allows customization via add_worker_graph method.
    Reuses the SDK's create_init_node and create_terminate_node; the LLM node is create_custom_llm_node.

    Class attributes:
        _loop_name: A LoopName enum value for registered loops, or a plain
            string for experiment-local loops resolved via dotted path.
        _loop_version: Version name (e.g. "auto", "any").
        default_tool_choice: ``tool_choice`` when the caller passes none.
    """

    _loop_name: ClassVar[str] = LoopName.BASIC
    _loop_version: ClassVar[str] = DEFAULT_LOOP_VERSION
    default_tool_choice: ClassVar[Literal["auto", "any"]] = "auto"

    def __init__(
        self,
        scenario: Agent,
        model: BaseChatModel,
        *,
        config: AgentGraphConfig | None = None,
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None = None,
        print_trace: bool | None = None,
        tool_choice: Literal["auto", "any"] | None = None,
        parallel_tool_calls: bool | None = None,
        strict: bool | None = None,
        handle_tool_errors: bool = False,
        embeddings: Embeddings | None = None,
        with_memory: bool = False,
        memory_tools: bool = False,
        graceful_termination: bool = False,
        detect_tool_repetition: bool = False,
        **kwargs: Any,
    ):
        self.config = config or AgentGraphConfig()
        self.scenario = scenario
        self.tools = self._create_tools(scenario.tools, embeddings, memory_tools)
        self.input_schema = scenario.input_schema
        self.output_schema = scenario.output_schema
        self.model = model
        self.guardrails = guardrails
        self.print_trace = print_trace
        self.tool_choice: Literal["auto", "any"] = tool_choice or self.default_tool_choice
        self.parallel_tool_calls = (
            parallel_tool_calls
            if parallel_tool_calls is not None
            else self.config.parallel_tool_calls
        )
        self.strict = strict
        self.handle_tool_errors = handle_tool_errors
        self.graceful_termination = graceful_termination
        self.detect_tool_repetition = detect_tool_repetition

        # Create LoopDetails with model info
        model_name = get_model_name(model)
        models: dict[str, str] = {"model": model_name}
        if embeddings is not None:
            models["embeddings"] = get_model_name(embeddings)
        self.details = LoopDetails(
            loop_name=self._loop_name,
            loop_version=self._loop_version,
            models=models,
        )

        # Memory store setup (separate from memory tools)
        self.memory: InMemoryStore | None = (
            InMemoryStore(
                index={
                    "embed": embeddings,
                    "dims": getattr(embeddings, "dimensions", None)
                    or len(embeddings.embed_query("test")),
                }
            )
            if with_memory and embeddings
            else None
        )

    @property
    def loop_key(self) -> str:
        """Get the loop key from details."""
        return self.details.loop_key

    def _create_messages(self, state: BaseModel) -> Sequence[SystemMessage | HumanMessage]:
        """Create messages from input state using scenario prompts.

        Args:
            state: The typed input matching input_schema

        Returns:
            Sequence of SystemMessage and HumanMessage
        """
        return create_messages(self.scenario, state)

    def _create_tools(
        self,
        scenario_tools: Sequence[BaseTool],
        embeddings: Embeddings | None = None,
        memory_tools: bool = False,
    ) -> Sequence[BaseTool]:
        """Create the list of tools for the loop, applying any necessary transformations."""
        main_tools = apply_tool_embeddings(scenario_tools, embeddings)
        if memory_tools:
            main_tools.extend(create_memory_tools())
        return main_tools

    def _create_all_nodes(self, builder: StateGraph) -> list[str]:
        """Create and add all nodes to the builder (reusable by subclasses).

        Returns:
            List of tool node names for routing
        """
        agent_tools = self.tools
        flow_control_tools: list[BaseTool] = (
            [] if self.config.is_conversational else create_flow_control_tools(self.output_schema)
        )

        llm_tools: list[BaseTool] = [*agent_tools, *flow_control_tools]
        self.llm_tools = llm_tools

        init_node = create_init_node(
            self._create_messages,
            self.input_schema,
            self.config.is_conversational,
        )

        tool_nodes = create_tool_node(
            agent_tools,
            handle_tool_errors=self.handle_tool_errors,
        )
        tool_nodes_with_guardrails = create_tools_guardrails_subgraph(
            tool_nodes,  # pyright: ignore[reportArgumentType]
            self.guardrails,
            input_schema=self.input_schema,
        )
        terminate_node = create_terminate_node(self.output_schema, self.config.is_conversational)

        init_with_guardrails_subgraph = create_agent_init_guardrails_subgraph(
            (AgentGraphNode.GUARDED_INIT, init_node),
            self.guardrails,
            input_schema=self.input_schema,
        )
        builder.add_node(AgentGraphNode.INIT, init_with_guardrails_subgraph)

        for tool_name, tool_node in tool_nodes_with_guardrails.items():
            builder.add_node(tool_name, tool_node)

        terminate_with_guardrails_subgraph = create_agent_terminate_guardrails_subgraph(
            (AgentGraphNode.GUARDED_TERMINATE, terminate_node),
            self.guardrails,
            input_schema=self.input_schema,
        )
        builder.add_node(AgentGraphNode.TERMINATE, terminate_with_guardrails_subgraph)

        llm_node = create_custom_llm_node(
            self.model,
            llm_tools,
            llm_messages_limit=self.config.llm_messages_limit,
            thinking_messages_limit=self.config.thinking_messages_limit,
            is_conversational=self.config.is_conversational,
            tool_choice=self.tool_choice,
            parallel_tool_calls=self.parallel_tool_calls,
            strict=self.strict,
            print_trace=self.print_trace,
            graceful_termination=self.graceful_termination,
            detect_tool_repetition=self.detect_tool_repetition,
        )

        llm_with_guardrails_subgraph = create_llm_guardrails_subgraph(
            (AgentGraphNode.LLM, llm_node), self.guardrails, input_schema=self.input_schema
        )
        builder.add_node(AgentGraphNode.AGENT, llm_with_guardrails_subgraph)

        return list(tool_nodes_with_guardrails.keys())

    def add_worker_graph(
        self,
        builder: StateGraph,
    ) -> None:
        """Add all worker nodes and edges to the graph.

        Args:
            builder: The StateGraph to add nodes and edges to
        """
        tool_node_names = self._create_all_nodes(builder)

        builder.add_edge(START, AgentGraphNode.INIT)
        builder.add_edge(AgentGraphNode.INIT, AgentGraphNode.AGENT)

        if self.config.is_conversational:
            route_agent = create_route_agent_conversational()
            target_node_names = [
                *tool_node_names,
                AgentGraphNode.TERMINATE,
            ]
        else:
            # The LLM node handles stalls (text-only replies).
            route_agent = create_route_agent()
            target_node_names = [
                AgentGraphNode.AGENT,
                *tool_node_names,
                AgentGraphNode.TERMINATE,
            ]

        builder.add_conditional_edges(
            AgentGraphNode.AGENT,
            route_agent,
            target_node_names,
        )

        if self.config.is_conversational:
            target_node_names.append(AgentGraphNode.AGENT)

        self._add_tool_edges(builder, tool_node_names, route_agent, target_node_names)

        builder.add_edge(AgentGraphNode.TERMINATE, END)

    def _add_tool_edges(
        self,
        builder: StateGraph,
        tool_node_names: list[str],
        route_agent: Any,
        target_node_names: list[str],
    ) -> None:
        """Add routing edges from tool nodes back into the graph.

        Override in subclasses to intercept tool output (e.g. insert a
        memory-record or summarization step between tools and the router).
        """
        for tool_name in tool_node_names:
            builder.add_conditional_edges(tool_name, route_agent, target_node_names)

    def _create_graph_state(self) -> type[BaseModel]:
        """Create the graph state model for this loop.

        Override in subclasses to extend the state with additional fields
        (e.g. ``context`` for summarization).
        """
        return create_state_with_input(self.input_schema)

    def build_graph(self) -> StateGraph:
        """Build an uncompiled agent graph following SDK's create_agent pattern.

        Returns a ``StateGraph`` that follows the
        INIT → AGENT ↔ TOOLS loop, terminated by the TERMINATE node.
        Call :meth:`build_compiled_graph` instead for a ready-to-invoke graph.

        Usage:
            # Preferred — with the gym settings, as the testbed builds it:
            compiled = get_loop("basic", "auto", scenario, {"model": llm})
            result = await compiled.ainvoke(agent_input)

            # Direct construction uses AgentGraphConfig() defaults unless a config is passed:
            loop = BasicLoop(scenario, llm, config=config)
            compiled = loop.build_compiled_graph()

            # Low-level — if you need the uncompiled graph:
            graph = loop.build_graph()
            compiled = graph.compile()

        Returns:
            Uncompiled StateGraph with ``loop_details`` attached.
        """
        graph_state = self._create_graph_state()

        builder: StateGraph = StateGraph(
            graph_state,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
        )
        builder.loop_details = self.details  # pyright: ignore[reportAttributeAccessIssue]

        # Add all worker nodes and edges (overridable)
        self.add_worker_graph(builder)

        return builder

    def build_compiled_graph(self, **compile_kwargs: Any) -> CompiledStateGraph:
        """Build and compile the agent graph, caching the result when safe.

        Caching is skipped when a ``checkpointer`` is provided because a
        stateful checkpointer (e.g. ``MemorySaver``) is bound to the compiled
        graph and sharing it across executions would leak state between agents.

        Args:
            **compile_kwargs: Keyword arguments forwarded to ``graph.compile()``.

        Returns:
            A compiled, ready-to-invoke graph.
        """
        if compile_kwargs.get("checkpointer") is not None:
            return self._compile_graph(**compile_kwargs)

        if not hasattr(self, "_compiled_graph"):
            self._compiled_graph = self._compile_graph(**compile_kwargs)
        return self._compiled_graph

    def _compile_graph(self, **compile_kwargs: Any) -> CompiledStateGraph:
        graph = self.build_graph()
        compiled = graph.compile(store=self.memory, **compile_kwargs)
        compiled.loop_details = graph.loop_details  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        return compiled


class BasicLoopAny(BasicLoop):
    """Basic loop with ``tool_choice="any"``: a tool call on every turn."""

    _loop_version: ClassVar[str] = "any"
    default_tool_choice: ClassVar[Literal["auto", "any"]] = "any"


class BasicLoopAutoGraceful(BasicLoop):
    """Basic loop that forces ``end_execution`` at the call limit or on repeated
    identical tool calls, instead of failing the run."""

    _loop_version: ClassVar[str] = "auto-graceful"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, graceful_termination=True, detect_tool_repetition=True, **kwargs)


class BasicLoopAnyGraceful(BasicLoopAny):
    """``basic:any`` with the graceful termination of ``basic:auto-graceful``."""

    _loop_version: ClassVar[str] = "any-graceful"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, graceful_termination=True, detect_tool_repetition=True, **kwargs)
