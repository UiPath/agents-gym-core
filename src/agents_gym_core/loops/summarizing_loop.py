"""Summarizing loop - adds recursive summarization before each LLM call."""

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from uipath_langchain.agent.guardrails.actions import GuardrailAction
from uipath_langchain.agent.guardrails.guardrails_factory import BaseGuardrail
from uipath_langchain.agent.react.router import create_route_agent
from uipath_langchain.agent.react.router_conversational import create_route_agent_conversational
from uipath_langchain.agent.react.types import AgentGraphConfig, AgentGraphNode
from uipath_langchain.agent.react.utils import create_state_with_input

from agents_gym_core.loops.basic_loop import BasicLoop
from agents_gym_core.loops.nodes import create_summarization_node
from agents_gym_core.loops.types import ExtendedGraphNode, LoopName
from agents_gym_core.models.utils import get_model_name
from agents_gym_core.utils.gym_types import Agent


class SummarizingLoop(BasicLoop):
    """Summarizing loop adds recursive message summarization before each LLM call.

    Uses langmem's SummarizationNode to compress older messages into a running
    summary at the configured token thresholds. Summary requests can include the
    agent's workflow instructions to track pending steps.

    Graph structure:
        START -> INIT -> SUMMARIZING -> AGENT <-> TOOLS -> TERMINATE
                         ^                    |
                         |____________________|
                         (tools & agent self-loop go through summarizing)
    """

    _loop_name: ClassVar[str] = LoopName.SUMMARIZING

    def __init__(
        self,
        scenario: Agent,
        model: BaseChatModel,
        *,
        config: AgentGraphConfig | None = None,
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None = None,
        tool_choice: Literal["auto", "any"] | None = None,
        strict: bool | None = None,
        print_trace: bool | None = None,
        summarizing_llm: BaseChatModel | None = None,
        max_summary_tokens: int = 800,
        max_tokens_before_summary: int = 800,
        max_tokens: int = 4000,
        initial_summary_prompt: str | None = "core::summary_initial_prompt.md",
        existing_summary_prompt: str | None = "core::summary_existing_prompt.md",
        inject_agent_instructions: bool = True,
        **kwargs: Any,
    ):
        super().__init__(
            scenario=scenario,
            model=model,
            config=config,
            guardrails=guardrails,
            tool_choice=tool_choice,
            strict=strict,
            print_trace=print_trace,
            graceful_termination=True,
            **kwargs,
        )

        self.inject_agent_instructions = inject_agent_instructions
        self.summarization_kwargs: dict[str, Any] = {
            "max_summary_tokens": max_summary_tokens,
            "max_tokens_before_summary": max_tokens_before_summary,
            "max_tokens": max_tokens,
            "initial_summary_prompt": initial_summary_prompt,
            "existing_summary_prompt": existing_summary_prompt,
        }

        if summarizing_llm is not None:
            self.summarizing_llm = summarizing_llm
            self.details = self.details.with_model(
                "summarizing_llm", get_model_name(summarizing_llm)
            )
        else:
            self.summarizing_llm = model

    def _create_graph_state(self) -> type[BaseModel]:
        """Extend the base graph state with a context field for summarization."""
        base_state = create_state_with_input(self.input_schema)

        class SummarizingAgentGraphState(base_state):  # type: ignore[misc]
            context: dict[str, Any] = Field(default_factory=dict)

        return SummarizingAgentGraphState

    def add_worker_graph(self, builder: StateGraph) -> None:
        """Add summarizing node to BasicLoop's graph.

        Inserts a SUMMARIZING node between INIT and AGENT, and routes
        tools and agent self-loops back through SUMMARIZING before each agent call.
        """
        tool_node_names = self._create_all_nodes(builder)

        # Override summarization token budget for agents tagged with "long_summary".
        summarization_kwargs = self.summarization_kwargs.copy()
        if "long_summary" in self.scenario.tags:
            summarization_kwargs.update(
                max_summary_tokens=2500,
                max_tokens_before_summary=2500,
                max_tokens=8000,
            )

        # Add summarizing node — inject agent instructions to help pipeline agents track pending steps.
        summarization_node = create_summarization_node(
            self.summarizing_llm,
            system_prompt=self.scenario.system_prompt if self.inject_agent_instructions else None,
            **summarization_kwargs,
        )
        builder.add_node(ExtendedGraphNode.SUMMARIZING, summarization_node)

        # Wire: START -> INIT -> SUMMARIZING -> AGENT
        builder.add_edge(START, AgentGraphNode.INIT)
        builder.add_edge(AgentGraphNode.INIT, ExtendedGraphNode.SUMMARIZING)
        builder.add_edge(ExtendedGraphNode.SUMMARIZING, AgentGraphNode.AGENT)

        # Setup routing from AGENT — remap AGENT self-edge through SUMMARIZING
        if self.config.is_conversational:
            route_agent = create_route_agent_conversational()
        else:
            route_agent = create_route_agent()

        builder.add_conditional_edges(
            AgentGraphNode.AGENT,
            route_agent,
            {
                AgentGraphNode.AGENT: ExtendedGraphNode.SUMMARIZING,
                **{t: t for t in tool_node_names},
                AgentGraphNode.TERMINATE: AgentGraphNode.TERMINATE,
            },
        )

        # Route from tools back through SUMMARIZING
        for tool_name in tool_node_names:
            builder.add_conditional_edges(
                tool_name,
                route_agent,
                {
                    AgentGraphNode.AGENT: ExtendedGraphNode.SUMMARIZING,
                    **{t: t for t in tool_node_names},
                    AgentGraphNode.TERMINATE: AgentGraphNode.TERMINATE,
                },
            )

        builder.add_edge(AgentGraphNode.TERMINATE, END)


class SummarizingLoopAny(SummarizingLoop):
    """Summarizing loop with ``tool_choice="any"``: a tool call on every turn."""

    _loop_version: ClassVar[str] = "any"
    default_tool_choice: ClassVar[Literal["auto", "any"]] = "any"
