"""Planning loops - adds planning step before execution."""

from collections.abc import Sequence
from typing import Any, ClassVar, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from uipath_langchain.agent.guardrails.actions import GuardrailAction
from uipath_langchain.agent.guardrails.guardrails_factory import BaseGuardrail
from uipath_langchain.agent.react import AgentGraphState
from uipath_langchain.agent.react.types import AgentGraphConfig, AgentGraphNode

from agents_gym_core.loops.basic_loop import BasicLoop
from agents_gym_core.loops.nodes import (
    PlanningPrompt,
    PlanPromptTemplate,
    create_planning_node,
)
from agents_gym_core.loops.types import ExtendedGraphNode, LoopName
from agents_gym_core.models.utils import get_model_name
from agents_gym_core.utils.gym_types import Agent


class PlanningLoop(BasicLoop):
    """Planning loop adds explicit planning before execution.

    Uses a two-stage LLM approach:
    1. Planning LLM: Creates a strategic plan based on the scenario
    2. Execution LLM: Follows the plan to complete the task

    Graph structure:
        START -> INIT -> PLANNING -> AGENT <-> TOOLS -> TERMINATE
    """

    _loop_name: ClassVar[str] = LoopName.PLANNING

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
        planning_llm: BaseChatModel | None = None,
        planner_system_prompt: str = PlanningPrompt.PLANNING_SYSTEM_PROMPT_PLATFORM,
        planner_user_prompt: str = PlanningPrompt.PLANNING_USER_PROMPT_PLATFORM,
        plan_insertion_prompt: PlanPromptTemplate = PlanPromptTemplate.PLAN_PROMPT_TEMPLATE_INCLUDE_END_EXECUTION,
        **kwargs: Any,
    ):
        super().__init__(
            scenario=scenario,
            model=model,
            guardrails=guardrails,
            config=config,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            strict=strict,
            print_trace=print_trace,
            **kwargs,
        )

        self.planner_system_prompt = planner_system_prompt
        self.planner_user_prompt = planner_user_prompt
        self.plan_insertion_prompt = plan_insertion_prompt

        if planning_llm is not None:
            self.planning_llm = planning_llm
            self.details = self.details.with_model("planning_llm", get_model_name(planning_llm))
        else:
            self.planning_llm = model

    def incorporate_plan(self, state: AgentGraphState, plan: str) -> None:
        """Incorporate the generated plan into the state.

        Args:
            state: Current state with messages
            plan: Generated plan from planning LLM
        """
        state.messages.append(HumanMessage(content=self.plan_insertion_prompt.format(plan=plan)))

    def add_worker_graph(self, builder: StateGraph) -> None:
        """Add planning node to the existing BasicLoop graph.

        Inserts a planning step between INIT and AGENT nodes.

        Args:
            builder: The StateGraph to add nodes and edges to
        """
        super().add_worker_graph(builder)

        planning_node = create_planning_node(
            planning_llm=self.planning_llm,
            incorporate_plan_func=self.incorporate_plan,
            tools=list(self.scenario.tools),
            system_prompt=self.planner_system_prompt,
            user_prompt=self.planner_user_prompt,
        )
        builder.add_node(ExtendedGraphNode.PLANNING, planning_node)

        builder.edges.remove((START, AgentGraphNode.INIT))
        builder.edges.remove((AgentGraphNode.INIT, AgentGraphNode.AGENT))

        builder.add_edge(START, AgentGraphNode.INIT)
        builder.add_edge(AgentGraphNode.INIT, ExtendedGraphNode.PLANNING)
        builder.add_edge(ExtendedGraphNode.PLANNING, AgentGraphNode.AGENT)


class PlanningSystemPlanLoop(PlanningLoop):
    """Planning loop that incorporates plan into the system message.

    Instead of appending the plan as a new message, this variant replaces
    the system message (index 0) with a version that includes the plan.
    """

    _loop_name: ClassVar[str] = LoopName.PLANNING
    _loop_version: ClassVar[str] = "system-plan"

    def __init__(
        self,
        scenario: Agent,
        model: BaseChatModel,
        *,
        config: AgentGraphConfig | None = None,
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None = None,
        print_trace: bool | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Literal["auto", "any"] | None = None,
        planning_llm: BaseChatModel | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            scenario=scenario,
            model=model,
            config=config,
            guardrails=guardrails,
            print_trace=print_trace,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            planning_llm=planning_llm,
            plan_insertion_prompt=PlanPromptTemplate.PLAN_PROMPT_TEMPLATE_SYSTEM_MESSAGE,
            **kwargs,
        )

    def incorporate_plan(self, state: AgentGraphState, plan: str) -> None:
        """Incorporate plan by replacing the system message."""
        state.messages[0] = SystemMessage(
            content=self.plan_insertion_prompt.format(
                prompt=state.messages[0].content,
                plan=plan,
            ),
            id=state.messages[0].id,
        )


class PlanningUserPlanLoop(PlanningLoop):
    """Planning loop that incorporates plan into the user message.

    Instead of appending the plan as a new message, this variant replaces
    the user message (index 1) with a version that includes the plan.
    """

    _loop_name: ClassVar[str] = LoopName.PLANNING
    _loop_version: ClassVar[str] = "user-plan"

    def __init__(
        self,
        scenario: Agent,
        model: BaseChatModel,
        *,
        config: AgentGraphConfig | None = None,
        guardrails: Sequence[tuple[BaseGuardrail, GuardrailAction]] | None = None,
        print_trace: bool | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: Literal["auto", "any"] | None = None,
        planning_llm: BaseChatModel | None = None,
        **kwargs: Any,
    ):
        super().__init__(
            scenario=scenario,
            model=model,
            config=config,
            guardrails=guardrails,
            print_trace=print_trace,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
            planning_llm=planning_llm,
            plan_insertion_prompt=PlanPromptTemplate.PLAN_PROMPT_TEMPLATE_USER_MESSAGE,
            **kwargs,
        )

    def incorporate_plan(self, state: AgentGraphState, plan: str) -> None:
        """Incorporate plan by replacing the user message."""
        state.messages[1] = HumanMessage(
            content=self.plan_insertion_prompt.format(
                prompt=state.messages[1].content,
                plan=plan,
            ),
            id=state.messages[1].id,
        )
