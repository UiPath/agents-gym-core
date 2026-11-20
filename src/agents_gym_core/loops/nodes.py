import json
import re
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal, TypeVar
from uuid import uuid4

from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langgraph.graph._node import StateNode
from langgraph.types import Command
from langmem.short_term import SummarizationNode
from langmem.short_term.summarization import (
    DEFAULT_EXISTING_SUMMARY_PROMPT,
    DEFAULT_INITIAL_SUMMARY_PROMPT,
)
from pydantic import BaseModel
from uipath.runtime.errors import UiPathErrorCategory
from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.messages.message_utils import replace_tool_calls
from uipath_langchain.agent.react import AgentGraphState
from uipath_langchain.agent.react.constants import DEFAULT_MAX_LLM_MESSAGES
from uipath_langchain.agent.react.llm_node import (
    StateT,
    _filter_control_flow_tool_calls,
)
from uipath_langchain.agent.react.types import FLOW_CONTROL_TOOLS
from uipath_langchain.agent.react.utils import count_consecutive_tool_less_turns
from uipath_langchain.agent.tools import create_tool_node as uipath_create_tool_node
from uipath_langchain.agent.tools.static_args import StaticArgsHandler
from uipath_langchain.agent.tools.tool_node import UiPathToolNode
from uipath_langchain.chat.handlers import get_payload_handler

from agents_gym_core.loops.loop_directory_provider import LoopDirectoryProvider
from agents_gym_core.loops.utils import (
    break_consecutive_ai_turns,
    force_end_invoke,
    get_tool_wrappers,
)
from agents_gym_core.models.chat.clients import HuggingFaceChatModels, VllmChatModels
from agents_gym_core.models.utils import get_model_name
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.prompt_loader import PromptLoader
from agents_gym_core.utils.text_utils import content_to_text
from agents_gym_core.utils.tool_utils import tool_to_str
from agents_gym_core.utils.trace_utils import warning

NODE_PROMPTS_DIR = "node_prompts"

# Tool-call directive for registered HuggingFace and vLLM models at the thinking limit
# or on an empty-response retry. Other models use a ``tool_choice="any"`` rebind.
_HF_FORCE_TOOL_NUDGE = HumanMessage(content="You must call a tool now.")

# What the model reads when a tool returns an empty list.
EMPTY_TOOL_RESULT = "[]"


class EmptyResultVisibleToolNode(UiPathToolNode):
    """UiPathToolNode that keeps an empty tool result visible to the model.

    Render ``content=[]`` as the literal text ``"[]"`` so the model receives an explicit
    empty-list result, such as an empty contact search. Other tool results pass through
    unchanged.
    """

    def _process_result(self, call: Any, result: Any) -> Any:
        output = super()._process_result(call, result)
        update = output.update if isinstance(output, Command) else output
        messages = update.get("messages", []) if isinstance(update, dict) else []
        for message in messages:
            if isinstance(message, ToolMessage) and message.content == []:
                message.content = EMPTY_TOOL_RESULT
        return output


def create_tool_node(
    tools: Sequence[BaseTool],
    *,
    handle_tool_errors: bool = True,
) -> Mapping[str, Runnable]:
    """Create individual tool nodes for each tool.

    Args:
        tools: Sequence of tools to create nodes for.
        handle_tool_errors: If True, wrap tools with error handling so exceptions
            are returned as error messages. If False, use the original uipath_langchain
            behavior where exceptions propagate and crash the loop. Defaults to True.

    Returns:
        Dict mapping tool.name -> EmptyResultVisibleToolNode.
    """
    if not handle_tool_errors:
        # uipath_langchain's own nodes (a tool's own wrappers only), re-created so
        # that an empty result stays visible.
        return {
            name: EmptyResultVisibleToolNode(
                node.tool, wrapper=node.wrapper, awrapper=node.awrapper
            )
            for name, node in uipath_create_tool_node(tools).items()
        }
    return {
        tool.name: EmptyResultVisibleToolNode(tool, wrapper=w, awrapper=aw)
        for tool in tools
        for w, aw in [get_tool_wrappers(tool)]
    }


def create_messages(scenario: Agent, state: BaseModel) -> Sequence[SystemMessage | HumanMessage]:
    """Create messages from input state using scenario prompts.

    For conversational agents with an 'input' field, the input is used directly
    as the user message (ignoring user_prompt.md). For other agents, the
    user_prompt template is formatted with the state fields.

    Args:
        state: The typed input matching input_schema

    Returns:
        Sequence of SystemMessage and HumanMessage
    """
    state_dict = state.model_dump()

    # For conversational agents: use 'input' field directly as user message
    if scenario.is_conversational:
        user_input = state_dict.get("input", "")
        if user_input:
            return _with_ids(
                [
                    SystemMessage(content=scenario.system_prompt),
                    HumanMessage(content=user_input),
                ]
            )

    # For non-conversational agents: format user_prompt with state fields
    try:
        messages = [
            SystemMessage(content=scenario.system_prompt),
            HumanMessage(content=scenario.user_prompt.format_map(state_dict)),
        ]
    except KeyError as exc:
        warning(
            f"User prompt template failed (missing key {exc}), "
            "falling back to generic prompt. "
            "Check for unescaped braces in user_prompt.md."
        )
        messages = [
            SystemMessage(content=scenario.system_prompt),
            HumanMessage(content=f"Help me with: {state.model_dump_json()}"),
        ]

    return _with_ids(messages) if scenario.is_conversational else messages


def _with_ids(
    messages: list[SystemMessage | HumanMessage],
) -> list[SystemMessage | HumanMessage]:
    """Give each message an id. Conversational init writes the first messages with
    ``Overwrite``, which assigns none, and the summarizing loop's SummarizationNode needs them."""
    for message in messages:
        message.id = message.id or str(uuid4())
    return messages


# Tool-less completions allowed before the loop asks for a tool call.
DEFAULT_MAX_CONSECUTIVE_THINKING_MESSAGES = 0

InputT = TypeVar("InputT", bound=BaseModel)


def create_custom_llm_node(
    model: BaseChatModel,
    tools: Sequence[BaseTool] | None = None,
    input_schema: type[InputT] | None = None,
    is_conversational: bool = False,
    llm_messages_limit: int = DEFAULT_MAX_LLM_MESSAGES,
    thinking_messages_limit: int = DEFAULT_MAX_CONSECUTIVE_THINKING_MESSAGES,
    tool_choice: Literal["auto", "any"] = "auto",
    parallel_tool_calls: bool | None = None,
    strict: bool | None = None,
    print_trace: bool | None = None,
    graceful_termination: bool = False,
    detect_tool_repetition: bool = False,
) -> StateNode[AgentGraphState, None]:
    """Invoke the model with tools, call limits, repetition checks and an empty-response retry.

    At the thinking limit, request a tool call through provider binding options or an
    in-context directive for registered HuggingFace and vLLM models.

    Args:
        graceful_termination: Controls limit-handling behaviour.
            Both modes use a single ``total_llm_calls`` counter in
            ``inner_state.tools_storage`` (survives summarization).
            ``False`` (default/classic): Raises ``AgentRuntimeError`` at the limit.
            ``True`` (graceful): Forces ``end_execution`` at the limit.
        detect_tool_repetition: When ``True``, forces
            ``end_execution`` when any identical tool call repeats
            ≥ ``max(llm_messages_limit // 3, 2)`` times. Independent of ``graceful_termination``.
    """
    bindable_tools = list(tools) if tools else []

    async def llm_node(state: StateT):  # pyright: ignore[reportInvalidTypeVarUse]
        messages: list[AnyMessage] = state.messages
        # Per call: the handler resolves static args from the input once and keeps them, and
        # one compiled graph serves every datapoint.
        static_args_handler = StaticArgsHandler()

        consecutive_thinking_messages = count_consecutive_tool_less_turns(messages)
        static_schema_tools = static_args_handler.initialize(
            bindable_tools, state, input_schema or type(state)
        )

        # Call-budget checks
        # tools_storage lives on AgentGraphState.inner_state and survives summarization.
        # total_llm_calls is the single counter for both classic and graceful modes
        # (equivalent to counting AIMessages but reliable across summarization).
        inner_state = getattr(state, "inner_state", None)
        tools_storage: dict[str, Any] = getattr(inner_state, "tools_storage", None) or {}
        total_llm_calls: int = tools_storage.get("total_llm_calls", 0)

        force_end_execution = False
        if not is_conversational and total_llm_calls >= llm_messages_limit:
            if not graceful_termination:
                # Classic: raise hard error at limit.
                raise AgentRuntimeError(
                    code=AgentRuntimeErrorCode.TERMINATION_MAX_ITERATIONS,
                    title=f"Maximum iterations of '{llm_messages_limit}' reached.",
                    detail="Verify the agent's trajectory or consider increasing the max iterations in the agent's settings.",
                    category=UiPathErrorCategory.USER,
                )
            else:
                # Graceful: force end_execution at limit.
                force_end_execution = True

        # Request end_execution at the identical-tool-call repetition limit.
        if detect_tool_repetition and not is_conversational and not force_end_execution:
            tool_counts: dict[str, int] = tools_storage.get("tool_counts", {})
            if tool_counts:
                max_tool_repeat = max(llm_messages_limit // 3, 2)
                if max(tool_counts.values()) >= max_tool_repeat:
                    force_end_execution = True

        # Request a tool call at the configured thinking-message limit.
        # The model-specific directive or binding options are applied below.
        force_tool_call = (
            not is_conversational
            and bool(bindable_tools)
            and consecutive_thinking_messages >= thinking_messages_limit
        )
        # Enforce the limit if the next completion remains tool-less after that request.
        if force_tool_call and consecutive_thinking_messages > thinking_messages_limit:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.THINKING_LIMIT_EXCEEDED,
                title="Agent exceeded consecutive completions limit without producing tool calls.",
                detail=f"Completions: {consecutive_thinking_messages}, max: {thinking_messages_limit}. "
                "If you are using a BYOM configuration, verify your model deployment respects "
                "tool_choice or equivalent.",
                category=UiPathErrorCategory.SYSTEM,
            )
        model_name = get_model_name(model)
        is_hf_model = model_name in HuggingFaceChatModels or model_name in VllmChatModels

        if force_end_execution:
            # Force end_execution tool, so the model cannot make further agent tool calls.
            response = await force_end_invoke(
                model, messages, static_schema_tools, parallel_tool_calls, strict
            )
        else:
            # Apply the configured tool-choice policy.
            bind_kwargs = get_payload_handler(model).get_tool_binding_kwargs(
                static_schema_tools,
                tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                strict_mode=strict,
            )
            if force_tool_call and not is_hf_model:
                bind_kwargs.update(
                    get_payload_handler(model).get_tool_binding_kwargs(
                        static_schema_tools,
                        "any",
                        parallel_tool_calls=parallel_tool_calls,
                        strict_mode=strict,
                    )
                )

            # Normalize registered Anthropic turn alternation independently of tool forcing.
            # Registered HuggingFace and vLLM models receive the tool-call directive at the limit.
            invoke_messages: Sequence[AnyMessage] = (
                break_consecutive_ai_turns(messages, model)
                if not is_conversational and bindable_tools
                else messages
            )
            if force_tool_call and is_hf_model:
                invoke_messages = list(invoke_messages) + [_HF_FORCE_TOOL_NUDGE]

            llm = (
                model.bind_tools(static_schema_tools, **bind_kwargs)
                if static_schema_tools
                else model
            )
            response = await llm.ainvoke(invoke_messages)

            # Retry an empty response once with a tool-call requirement:
            # a directive for registered HuggingFace/vLLM models, binding options otherwise.
            if (
                isinstance(response, AIMessage)
                and not response.content
                and not response.tool_calls
                and static_schema_tools
            ):
                if is_hf_model:
                    retry_llm = llm
                    retry_messages: Sequence[AnyMessage] = list(messages) + [_HF_FORCE_TOOL_NUDGE]
                else:
                    retry_bind_kwargs = get_payload_handler(model).get_tool_binding_kwargs(
                        static_schema_tools,
                        "any",
                        parallel_tool_calls=parallel_tool_calls,
                        strict_mode=strict,
                    )
                    retry_llm = model.bind_tools(static_schema_tools, **retry_bind_kwargs)
                    # Apply the same provider turn alternation before the retry.
                    retry_messages = (
                        break_consecutive_ai_turns(messages, model)
                        if not is_conversational and bindable_tools
                        else messages
                    )
                response = await retry_llm.ainvoke(retry_messages)

        if not isinstance(response, AIMessage):
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.LLM_INVALID_RESPONSE,
                title=f"LLM returned {type(response).__name__} invalid response.",
                detail="The language model returned an unexpected response type."
                "If you are using a BYOM configuration, verify your model deployment.",
                category=UiPathErrorCategory.SYSTEM,
            )
        if print_trace:
            print(f"\nAssistant: {response.content}")
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    print(f"Tool Call: {tool_name}")

        # filter out flow control tools when multiple tool calls exist
        if response.tool_calls:
            filtered_tool_calls = _filter_control_flow_tool_calls(response.tool_calls)
            if len(filtered_tool_calls) != len(response.tool_calls):
                response = replace_tool_calls(response, filtered_tool_calls)

        static_args_handler.apply_to_response(response.tool_calls)

        result: dict[str, Any] = {"messages": [response]}

        # Always persist the LLM call counter (both modes need it).
        # Track tool-call repetitions when detect_tool_repetition is enabled.
        storage_update: dict[str, Any] = {"total_llm_calls": total_llm_calls + 1}
        if detect_tool_repetition and response.tool_calls:
            tool_counts_update: dict[str, int] = dict(tools_storage.get("tool_counts", {}))
            for tc in response.tool_calls:
                if tc["name"] not in FLOW_CONTROL_TOOLS:
                    call_key = f"{tc['name']}:{json.dumps(tc.get('args', {}), sort_keys=True)}"
                    tool_counts_update[call_key] = tool_counts_update.get(call_key, 0) + 1
            storage_update["tool_counts"] = tool_counts_update
        result["inner_state"] = {"tools_storage": storage_update}

        return result

    return llm_node


# ========================================
# Planning Node Functions
# ========================================


class PlanningPrompt(StrEnum):
    """Planning prompt templates stored as markdown files."""

    PLANNING_SYSTEM_PROMPT_INITIAL = PromptLoader.load_prompt_from_file(
        "core::planning_system_prompt_initial.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )
    PLANNING_SYSTEM_PROMPT_TOOLS = PromptLoader.load_prompt_from_file(
        "core::planning_system_prompt_tools.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )
    PLANNING_SYSTEM_PROMPT_MORE_DETAIL = PromptLoader.load_prompt_from_file(
        "core::planning_system_prompt_more_detail.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )
    PLANNING_SYSTEM_PROMPT_PLATFORM = PromptLoader.load_prompt_from_file(
        "core::planning_system_prompt_platform.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )

    PLANNING_USER_PROMPT_INITIAL = PromptLoader.load_prompt_from_file(
        "core::planning_user_prompt_initial.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )
    PLANNING_USER_PROMPT_MORE_DETAIL = PromptLoader.load_prompt_from_file(
        "core::planning_user_prompt_more_detail.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )
    PLANNING_USER_PROMPT_PLATFORM = PromptLoader.load_prompt_from_file(
        "core::planning_user_prompt_platform.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )


class PlanPromptTemplate(StrEnum):
    """Plan insertion prompt templates stored as markdown files."""

    PLAN_PROMPT_TEMPLATE_INITIAL = PromptLoader.load_prompt_from_file(
        "core::plan_prompt_template_initial.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )

    PLAN_PROMPT_TEMPLATE_INCLUDE_END_EXECUTION = PromptLoader.load_prompt_from_file(
        "core::plan_prompt_template_include_end_execution.md",
        NODE_PROMPTS_DIR,
        LoopDirectoryProvider(),
    )

    PLAN_PROMPT_TEMPLATE_FORCED = PromptLoader.load_prompt_from_file(
        "core::plan_prompt_template_forced.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )

    PLAN_PROMPT_TEMPLATE_USER_MESSAGE = PromptLoader.load_prompt_from_file(
        "core::plan_prompt_template_user_message.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )

    PLAN_PROMPT_TEMPLATE_SYSTEM_MESSAGE = PromptLoader.load_prompt_from_file(
        "core::plan_prompt_template_system_message.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )


def create_planning_node(
    planning_llm: BaseChatModel,
    incorporate_plan_func: Callable[[AgentGraphState, str], None],
    tools: Sequence[BaseTool],
    system_prompt: str = PlanningPrompt.PLANNING_SYSTEM_PROMPT_PLATFORM,
    user_prompt: str = PlanningPrompt.PLANNING_USER_PROMPT_PLATFORM,
) -> StateNode[AgentGraphState, None]:
    """Create a planning node that generates a strategic plan before execution.

    Args:
        planning_llm: Language model to use for planning
        incorporate_plan_func: Function to incorporate the plan into state
        tools: Available tools for the agent
        system_prompt: System prompt template for the planning LLM
        user_prompt: User prompt template for the planning LLM

    Returns:
        Planning node function compatible with LangGraph
    """

    async def planning_node(state: AgentGraphState) -> dict[str, list[AnyMessage]]:
        """Generate a plan for executing the scenario."""
        messages = state.messages

        # Create a planning prompt
        planning_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=user_prompt.format(
                    systemPrompt=messages[0].content,
                    userPrompt=messages[1].content,
                    tools="\n".join(
                        [tool_to_str(tool, as_yaml=False, include_tags=True) for tool in tools]
                    ),
                    model_name=getattr(planning_llm, "model_name", "gpt-4o"),
                )
            ),
        ]

        # Get the planning response
        planning_response = await planning_llm.ainvoke(planning_messages)
        plan = str(planning_response.content)

        # Incorporate plan into state
        incorporate_plan_func(state, plan)

        return {"messages": state.messages}

    return planning_node


# ========================================
# Validation Node Functions
# ========================================


class ValidationPrompt(StrEnum):
    """Validation prompt templates stored as markdown files."""

    VALIDATION_SYSTEM_PROMPT = PromptLoader.load_prompt_from_file(
        "core::validation_system_prompt.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )
    VALIDATION_USER_PROMPT = PromptLoader.load_prompt_from_file(
        "core::validation_user_prompt.md", NODE_PROMPTS_DIR, LoopDirectoryProvider()
    )


def create_validation_node(
    validation_llm: BaseChatModel,
    tools: Sequence[BaseTool],
    system_prompt: str = ValidationPrompt.VALIDATION_SYSTEM_PROMPT,
    user_prompt: str = ValidationPrompt.VALIDATION_USER_PROMPT,
) -> StateNode[AgentGraphState, None]:
    """Create a validation node that validates execution against requirements.

    Args:
        validation_llm: Language model to use for validation
        tools: Available tools for the agent
        system_prompt: System prompt template for validation
        user_prompt: User prompt template for validation

    Returns:
        Validation node function compatible with LangGraph
    """

    def get_execution_details(state: AgentGraphState) -> str:
        """Extract execution details from state for validation."""
        details = []
        for msg in state.messages:
            if isinstance(msg, AIMessage):
                details.append(f"Assistant: {msg.content}")
                for tc in msg.tool_calls:
                    details.append(f"  Tool Call: {tc['name']}({tc['args']})")
            elif isinstance(msg, ToolMessage):
                details.append(f"Tool Result ({msg.name}): {msg.content}")
        return "\n".join(details)

    async def validation_node(state: AgentGraphState) -> dict[str, list[AnyMessage]]:
        """Validate whether the execution satisfied the requirements."""
        last_message = state.messages[-1]
        if not isinstance(last_message, AIMessage):
            raise ValueError("Last message before validation must be an AIMessage")

        validation_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=user_prompt.format(
                    systemPrompt=state.messages[0].content,
                    userPrompt=state.messages[1].content,
                    tools="\n".join(
                        [tool_to_str(tool, as_yaml=True, include_tags=True) for tool in tools]
                    ),
                    executionDetails=get_execution_details(state),
                )
            ),
        ]

        validation_response = await validation_llm.ainvoke(validation_messages)
        validation_result = content_to_text(validation_response.content)

        # Only a complete final verdict decides. Allow common inline formatting,
        # while rejecting negation and verdicts mentioned within explanatory prose.
        lines = validation_result.strip().splitlines()
        verdict = lines[-1].strip() if lines else ""
        match = re.fullmatch(
            r"""(?:Verdict:\s*)?(?P<wrapper>\*\*|__|\*|_|`|"|')?"""
            r"VALIDATION (?P<status>PASSED|FAILED)[.!]?"
            r"(?(wrapper)(?P=wrapper))[.!]?",
            verdict,
            re.IGNORECASE,
        )
        if match is not None and match["status"].upper() == "PASSED":
            # Validation passed - don't modify messages, let router route to TERMINATE
            return {}
        else:
            # Validation failed - remove the AIMessage with tool calls and add feedback as HumanMessage
            return {
                "messages": [
                    RemoveMessage(id=last_message.id),  # pyright: ignore[reportReturnType, reportArgumentType]
                    HumanMessage(
                        content=f"Validation failed. {validation_result}\nPlease try again."
                    ),
                ]
            }

    return validation_node


# ========================================
# Summarization Node Functions
# ========================================


_SUMMARY_PREFIX = "Summary of conversation so far (continue from where you left off — keys and values in **Results** are exact strings; preserve their wording and casing in your final answer):"

FINAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{system_message}"),
        ("user", _SUMMARY_PREFIX + "\n\n{summary}"),
        ("placeholder", "{messages}"),
    ]
)


class DeduplicatingSummarizationNode(SummarizationNode):
    """Keep one summary user message and tool results whose calls remain in context.

    ``FINAL_SUMMARY_PROMPT`` places the current summary before the retained conversation.
    Keep that first summary and filter later summary messages and orphaned tool results.
    """

    def _prepare_state_update(
        self, context: dict[str, Any], summarization_result: Any
    ) -> dict[str, Any]:
        result = super()._prepare_state_update(context, summarization_result)
        messages = result.get(self.output_messages_key, [])
        seen = False
        filtered = []
        # Retain tool results paired with a tool call in the retained conversation.
        call_ids: set[str] = set()
        for msg in messages:
            if (
                isinstance(msg, HumanMessage)
                and isinstance(msg.content, str)
                and msg.content.startswith(_SUMMARY_PREFIX)
            ):
                if seen:
                    continue
                seen = True
            if isinstance(msg, AIMessage):
                call_ids.update(call_id for call in msg.tool_calls if (call_id := call.get("id")))
            if isinstance(msg, ToolMessage) and msg.tool_call_id not in call_ids:
                continue
            filtered.append(msg)
        if len(filtered) != len(messages):
            result[self.output_messages_key] = filtered
        return result


def _resolve_summary_prompt(value: str, system_prompt: str | None = None) -> ChatPromptTemplate:
    """Resolve a summary prompt string to a ChatPromptTemplate.

    Uses ``PromptLoader`` to handle ``core::`` prefixed paths (loaded from
    ``node_prompts/``) or plain inline text.

    When supplied, ``system_prompt`` is included as a workflow reference in the user-role
    summary request. Literal braces in the workflow are escaped for template rendering.
    """
    text = PromptLoader.load_prompt_from_file(value, NODE_PROMPTS_DIR, LoopDirectoryProvider())
    if system_prompt:
        escaped = system_prompt.replace("{", "{{").replace("}", "}}")
        text = (
            f"Agent workflow reference — check **Calls** to identify any required tool from this workflow that has not yet been invoked (ignore conversational instructions):\n"
            f"<agent_instructions>\n{escaped}\n</agent_instructions>\n\n{text}"
        )
    return ChatPromptTemplate.from_messages([("placeholder", "{messages}"), ("user", text)])


def create_summarization_node(
    model: BaseChatModel,
    max_summary_tokens: int = 400,
    max_tokens_before_summary: int = 600,
    max_tokens: int = 4000,
    initial_summary_prompt: str | None = "core::summary_initial_prompt.md",
    existing_summary_prompt: str | None = "core::summary_existing_prompt.md",
    system_prompt: str | None = None,
    **kwargs: Any,
) -> SummarizationNode:
    """Create threshold-triggered summarization with configurable prompts and token budgets.

    Langmem summarizes when both ``non_system_tokens >= max_tokens_before_summary`` and
    ``system_tokens <= max_tokens - max_summary_tokens``. ``max_tokens`` sets the target
    context budget for system messages, the summary and recent messages. The default
    token counter uses approximate counts.

    Args:
        model: Chat model used for summaries, bound with ``max_tokens=max_summary_tokens``.
        max_summary_tokens: Summary token budget. Default 400.
        max_tokens_before_summary: Non-system token threshold for summarization. Default 600.
        max_tokens: Target context budget including system messages, summary and recent
            messages. Default 4000.
        initial_summary_prompt: ``core::`` path under ``node_prompts/``, inline text, or
            ``None`` for langmem's default prompt.
        existing_summary_prompt: Prompt for extending a summary, with the same resolution
            rules. Inline text includes the ``{existing_summary}`` placeholder.
        system_prompt: Optional agent workflow included in the user-role summary request.
        **kwargs: Additional settings forwarded to the summary node.
    """
    if max_summary_tokens <= 0 or max_tokens_before_summary <= 0 or max_tokens <= 0:
        raise ValueError("All max_tokens settings must be greater than 0 for summarization node.")

    initial_summary_prompt_template = (
        _resolve_summary_prompt(initial_summary_prompt, system_prompt)
        if initial_summary_prompt
        else DEFAULT_INITIAL_SUMMARY_PROMPT
    )
    existing_summary_prompt_template = (
        _resolve_summary_prompt(existing_summary_prompt, system_prompt)
        if existing_summary_prompt
        else DEFAULT_EXISTING_SUMMARY_PROMPT
    )

    model_runnable = model.bind(max_tokens=max_summary_tokens)
    return DeduplicatingSummarizationNode(
        model=model_runnable,
        input_messages_key="messages",
        output_messages_key="messages",
        max_summary_tokens=max_summary_tokens,
        max_tokens_before_summary=max_tokens_before_summary,
        max_tokens=max_tokens,
        initial_summary_prompt=initial_summary_prompt_template,
        existing_summary_prompt=existing_summary_prompt_template,
        final_prompt=FINAL_SUMMARY_PROMPT,
        **kwargs,
    )
