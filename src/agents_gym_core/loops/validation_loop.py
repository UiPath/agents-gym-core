"""Validation loop - adds validation step before termination."""

from collections.abc import Sequence
from functools import partial
from typing import Any, ClassVar, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from uipath.runtime.errors.contract import UiPathErrorCategory
from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.guardrails.actions import GuardrailAction
from uipath_langchain.agent.guardrails.guardrails_factory import BaseGuardrail
from uipath_langchain.agent.react import AgentGraphState
from uipath_langchain.agent.react.tools import create_flow_control_tools
from uipath_langchain.agent.react.types import AgentGraphConfig, AgentGraphNode
from uipath_langchain.agent.react.utils import (
    extract_current_tool_call_index,
    find_latest_ai_message,
)

from agents_gym_core.loops.basic_loop import BasicLoop
from agents_gym_core.loops.nodes import ValidationPrompt, create_validation_node
from agents_gym_core.loops.types import ExtendedGraphNode, LoopName
from agents_gym_core.models.utils import get_model_name
from agents_gym_core.utils.gym_types import Agent


def route_to_validator(state: AgentGraphState, tool_node_names: list[str]) -> str:
    """Route the next unanswered tool call: end_execution/raise_error to the validator, any
    other tool to its node, one call at a time; back to the agent once every call has a result.

    Each tool node answers the call it is routed for, so parallel calls run in order and two
    calls of the same tool both run.
    """
    messages = state.messages
    last_ai = find_latest_ai_message(messages)
    if last_ai is None or not last_ai.tool_calls:
        return AgentGraphNode.AGENT

    index = extract_current_tool_call_index(messages)
    if index is None:
        return AgentGraphNode.AGENT

    name = last_ai.tool_calls[index]["name"]
    if name in {"end_execution", "raise_error"}:
        return ExtendedGraphNode.VALIDATION
    if name in tool_node_names:
        return name
    raise AgentRuntimeError(
        code=AgentRuntimeErrorCode.ROUTING_ERROR,
        title="Agent routed to an unknown destination",
        detail=f"The agent called '{name}', which is not one of its tools.",
        category=UiPathErrorCategory.SYSTEM,
    )


def route_after_validator(
    state: AgentGraphState,
) -> str:
    """Route based on validation result.

    When validation passes, AIMessage remains -> route to TERMINATE
    When validation fails, HumanMessage with feedback -> route to AGENT
    """
    last_message = state.messages[-1]

    # If last message is HumanMessage, validation failed - route to AGENT
    if isinstance(last_message, HumanMessage):
        return AgentGraphNode.AGENT

    # If last message is AIMessage, validation passed - route to TERMINATE
    return AgentGraphNode.TERMINATE


class ValidationLoop(BasicLoop):
    """Validation loop adds a validation step before termination.

    Graph structure:
        START -> INIT -> AGENT <-> TOOLS
                           |
                           v (end_execution/raise_error)
                       VALIDATION
                           |
                           v (PASSED -> TERMINATE, FAILED -> AGENT)
    """

    _loop_name: ClassVar[str] = LoopName.VALIDATION

    def __init__(
        self,
        scenario: Agent,
        model: BaseChatModel,
        *,
        config: AgentGraphConfig | None = None,
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None = None,
        tool_choice: Literal["auto", "any"] | None = None,
        parallel_tool_calls: bool | None = None,
        strict: bool | None = None,
        print_trace: bool | None = None,
        validation_llm: BaseChatModel | None = None,
        validator_system_prompt: str = ValidationPrompt.VALIDATION_SYSTEM_PROMPT,
        validator_user_prompt: str = ValidationPrompt.VALIDATION_USER_PROMPT,
        **kwargs: Any,
    ):
        # Validation routes through the structured termination tools.
        if config is not None and config.is_conversational:
            raise ValueError(
                "ValidationLoop requires structured termination through end_execution or raise_error. "
                "Use BasicLoop with is_conversational=True instead."
            )
        if scenario.is_conversational:
            raise ValueError(
                f"Agent '{scenario.name}' uses conversational execution. "
                "ValidationLoop requires structured termination through end_execution or raise_error. "
                "Use BasicLoop for this agent."
            )
        super().__init__(
            scenario=scenario,
            model=model,
            config=config,
            guardrails=guardrails,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            strict=strict,
            print_trace=print_trace,
            **kwargs,
        )

        self.validator_system_prompt = validator_system_prompt
        self.validator_user_prompt = validator_user_prompt

        if validation_llm is not None:
            self.validation_llm = validation_llm
            self.details = self.details.with_model("validation_llm", get_model_name(validation_llm))
        else:
            self.validation_llm = model

    def add_worker_graph(self, builder: StateGraph) -> None:
        """Add all worker nodes including validation with custom routing."""
        tool_node_names = self._create_all_nodes(builder)

        agent_tools = list(self.scenario.tools)
        flow_control_tools: list[BaseTool] = create_flow_control_tools(self.scenario.output_schema)
        llm_tools: list[BaseTool] = [*agent_tools, *flow_control_tools]

        validation_node = create_validation_node(
            validation_llm=self.validation_llm,
            tools=llm_tools,
            system_prompt=self.validator_system_prompt,
            user_prompt=self.validator_user_prompt,
        )
        builder.add_node(ExtendedGraphNode.VALIDATION, validation_node)

        builder.add_edge(START, AgentGraphNode.INIT)
        builder.add_edge(AgentGraphNode.INIT, AgentGraphNode.AGENT)

        target_node_names = [AgentGraphNode.AGENT, ExtendedGraphNode.VALIDATION, *tool_node_names]

        builder.add_conditional_edges(
            AgentGraphNode.AGENT,
            partial(route_to_validator, tool_node_names=tool_node_names),
            target_node_names,
        )

        # Tools use conditional edges back (same routing logic as AGENT)
        for tool_name in tool_node_names:
            builder.add_conditional_edges(
                tool_name,
                partial(route_to_validator, tool_node_names=tool_node_names),
                target_node_names,
            )

        # Route after validation: to AGENT if failed, to TERMINATE if passed
        builder.add_conditional_edges(
            ExtendedGraphNode.VALIDATION,
            route_after_validator,
            [AgentGraphNode.AGENT, AgentGraphNode.TERMINATE],
        )

        builder.add_edge(AgentGraphNode.TERMINATE, END)
