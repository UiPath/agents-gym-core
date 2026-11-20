"""Utility functions for agent loops."""

import functools
from collections.abc import Callable, Sequence
from typing import Any

from langchain.tools import BaseTool as LangchainBaseTool
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool
from pydantic import create_model
from uipath.agent.react import END_EXECUTION_TOOL
from uipath_langchain.agent.react import AgentGraphState
from uipath_langchain.agent.tools.tool_node import ToolWrapperMixin
from uipath_langchain.chat.handlers import get_payload_handler

from agents_gym_core.models.chat.clients import (
    AnthropicBedrockModels,
    AnthropicVertexModels,
    LangchainAnthropicModels,
)
from agents_gym_core.models.utils import get_model_name

TOOL_CALL_ERROR_TEMPLATE = "Error: {error}"

# Registries whose text-only assistant tails are normalized before model calls.
# HuggingFace turn alternation is handled by models/chat/huggingface.py.
_ANTHROPIC_REGISTRIES = (
    AnthropicBedrockModels,
    AnthropicVertexModels,
    LangchainAnthropicModels,
)

# Neutral user turn for provider turn alternation. The tool-forcing directive is applied
# separately by ``thinking_messages_limit`` in nodes.py.
_ALTERNATION_BREAKER = HumanMessage(content="Please continue.")


def _is_anthropic(model: BaseChatModel) -> bool:
    name = get_model_name(model)
    for prefix in ("eu.", "us.", "apac."):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return any(name in reg for reg in _ANTHROPIC_REGISTRIES)


def _has_text_only_ai_tail(messages: Sequence[AnyMessage]) -> bool:
    """True when the last message is an ``AIMessage`` with non-empty content
    and no tool calls — i.e. a "thinking" turn that ends the conversation in
    an assistant role.

    AI messages carrying tool calls route to TOOLS rather than back to AGENT.
    """
    if not messages:
        return False
    last = messages[-1]
    return isinstance(last, AIMessage) and not last.tool_calls and bool(last.content)


def break_consecutive_ai_turns(
    messages: Sequence[AnyMessage], model: BaseChatModel
) -> list[AnyMessage]:
    """Insert a neutral user turn when the tail is a text-only ``AIMessage``
    and ``model`` is from an Anthropic registry. No-op for any other model.

    This preserves the provider's turn structure. Tool forcing is controlled separately by
    ``thinking_messages_limit`` in ``nodes.py``.
    """
    if _is_anthropic(model) and _has_text_only_ai_tail(messages):
        return list(messages) + [_ALTERNATION_BREAKER]
    return list(messages)


def _error_handling_wrapper(
    tool: BaseTool, call: ToolCall, state: AgentGraphState
) -> dict[str, Any] | None:
    """Wrapper that catches tool errors and returns them as error messages.

    The agent receives the exception details and can choose its next action.
    """
    try:
        return tool.invoke(call["args"])
    except Exception as e:
        return {"error": TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))}


async def _async_error_handling_wrapper(
    tool: BaseTool, call: ToolCall, state: AgentGraphState
) -> dict[str, Any] | None:
    """Async wrapper that catches tool errors and returns them as error messages."""
    try:
        return await tool.ainvoke(call["args"])
    except Exception as e:
        return {"error": TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))}


def _with_error_handling(wrapper: Callable | None) -> Callable | None:
    """``wrapper`` with the same error handling; its signature is kept for the tool node."""
    if wrapper is None:
        return None

    @functools.wraps(wrapper)
    def handled(*args: Any) -> Any:
        try:
            return wrapper(*args)
        except Exception as e:
            return {"error": TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))}

    return handled


def _with_async_error_handling(awrapper: Callable | None) -> Callable | None:
    if awrapper is None:
        return None

    @functools.wraps(awrapper)
    async def handled(*args: Any) -> Any:
        try:
            return await awrapper(*args)
        except Exception as e:
            return {"error": TOOL_CALL_ERROR_TEMPLATE.format(error=repr(e))}

    return handled


def get_tool_wrappers(tool: BaseTool) -> tuple[Callable | None, Callable | None]:
    """Return the sync/async wrappers to apply to ``tool`` when tool errors are handled.

    A raised exception becomes an ``{"error": ...}`` payload the agent can read instead of
    crashing the loop. A tool's own ``wrapper``/``awrapper`` (via :class:`ToolWrapperMixin`)
    still runs, inside the same error handling; a path it has no wrapper for gets the plain one.
    """
    if isinstance(tool, ToolWrapperMixin) and (tool.wrapper or tool.awrapper):
        return (
            _with_error_handling(tool.wrapper) or _error_handling_wrapper,
            _with_async_error_handling(tool.awrapper) or _async_error_handling_wrapper,
        )
    return _error_handling_wrapper, _async_error_handling_wrapper


def apply_tool_embeddings(
    tools: Sequence[BaseTool], embeddings: Embeddings | None
) -> list[BaseTool]:
    """Inject the experiment's embeddings model into tools that consume embeddings.

    Convention: a tool opts in by declaring ``embeddings`` and ``embeddings_model_name``
    fields defaulting to None (CoreVectorDatabaseTool, SimulationTool).
    The ``embeddings`` field is filled here unless already set or the tool declares an
    explicit ``embeddings_model_name`` in YAML — then the tool builds its own embedder.
    """
    if embeddings is None:
        return list(tools)
    main_tools: list[BaseTool] = []
    for tool in tools:
        if (
            "embeddings" in type(tool).model_fields
            and getattr(tool, "embeddings", None) is None
            and getattr(tool, "embeddings_model_name", None) is None
        ):
            tool = tool.model_copy(update={"embeddings": embeddings})
        main_tools.append(tool)
    return main_tools


_FORCE_END_PROMPT = (
    "You have used all available iterations. You MUST call end_execution now "
    "with your best answer based on the information gathered so far. "
    "Do NOT call any other tool."
)


async def force_end_invoke(
    model: BaseChatModel,
    messages: Sequence[AnyMessage],
    tools: Sequence[LangchainBaseTool],
    parallel_tool_calls: bool | None,
    strict: bool | None,
) -> AIMessage:
    """Filter tools to ``end_execution`` only, then invoke with ``tool_choice="any"``.

    Used by the graceful-termination valves of ``create_custom_llm_node``.
    """
    end_tools = [t for t in tools if t.name == END_EXECUTION_TOOL.name]
    invoke_messages = list(messages) + [HumanMessage(content=_FORCE_END_PROMPT)]
    bind_kwargs = get_payload_handler(model).get_tool_binding_kwargs(
        end_tools,
        tool_choice="any",
        parallel_tool_calls=parallel_tool_calls,
        strict_mode=strict,
    )

    llm = model.bind_tools(end_tools, **bind_kwargs)
    return await llm.ainvoke(invoke_messages)


# ---------------------------------------------------------------------------
# DSPy helpers for dspy_program_loop
# ---------------------------------------------------------------------------


def build_dspy_state_model(scenario: Any) -> type:
    """Create a Pydantic state model combining input + output fields.

    Used as the ``state_schema`` of a LangGraph ``StateGraph`` so that both
    input values and output placeholders live in the same flat state dict.

    Args:
        scenario: The gym ``Agent`` definition.

    Returns:
        A dynamically created Pydantic ``BaseModel`` subclass with all fields.
    """
    fields: dict[str, Any] = {}

    for name, field_info in scenario.input_schema.model_fields.items():
        fields[name] = (
            field_info.annotation or str,
            ... if field_info.is_required() else field_info.default,
        )

    for name, field_info in scenario.output_schema.model_fields.items():
        fields[name] = (field_info.annotation or str, None)

    return create_model("DspyState", **fields)


def build_dspy_output_model(scenario: Any) -> type:
    """Create a Pydantic model with only the output fields.

    Used as the ``output_schema`` of a LangGraph ``StateGraph`` so that
    ``graph.invoke()`` returns only output fields, not the full state.

    Args:
        scenario: The gym ``Agent`` definition.

    Returns:
        A dynamically created Pydantic ``BaseModel`` subclass with output fields only.
    """
    fields: dict[str, Any] = {}
    for name, field_info in scenario.output_schema.model_fields.items():
        fields[name] = (field_info.annotation or str, None)
    return create_model("DspyOutput", **fields)
