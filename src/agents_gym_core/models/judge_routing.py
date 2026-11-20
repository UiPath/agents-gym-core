"""Route the UiPath SDK's LLM-judge evaluators through this package's model registry.

``LLMJudgeMixin`` calls its own LLM service, which needs UiPath platform credentials. After
:func:`route_llm_judges_through_registry`, judge calls resolve through
:func:`agents_gym_core.models.chat.clients.get_chat_model` instead, so they use the same
backend and credentials as the agent and optimizer (``AGENTS_GYM_LLM_BACKEND``). Each judge uses the model, temperature and max_tokens set in its
evaluator JSON (max_tokens unset: 8000 for Claude Haiku and Sonnet 4.5, else 4096), whatever
``UIPATH_AGENTS_GYM_LLM_JUDGE_TEMPERATURE`` and ``_MAX_TOKENS`` say.
"""

import json
import time
import uuid
from functools import partial
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from uipath.eval.evaluators.llm_as_judge_evaluator import LLMJudgeMixin
from uipath.platform.chat.llm_gateway import (
    ChatCompletion,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatMessage,
    RequiredToolChoice,
    SpecificToolChoice,
    ToolCall,
    ToolChoice,
    ToolDefinition,
)


def route_llm_judges_through_registry() -> None:
    """Patch ``LLMJudgeMixin._get_llm_service`` so judge calls flow through ``get_chat_model``."""
    LLMJudgeMixin._get_llm_service = lambda self: partial(routed_llm_service, is_judge=True)  # type: ignore[method-assign]


async def routed_llm_service(
    messages: list[dict[str, str]],
    model: str = "gpt-4o-2024-11-20",
    is_judge: bool = False,
    max_tokens: int = 4096,
    temperature: float = 0,
    tools: list[ToolDefinition] | None = None,
    tool_choice: ToolChoice | None = None,
    response_format: dict[str, Any] | type[BaseModel] | None = None,
    model_kwargs: dict[str, Any] | None = None,
    *,
    chat_model: BaseChatModel | None = None,
) -> ChatCompletion:
    """Use an explicit chat model or the registry and return the SDK response shape.

    An explicit model retains its generation settings and bypasses registry loading.
    Supports tool/function calling and ``response_format`` structured output.
    """
    lc_messages = [
        SystemMessage(content=m["content"])
        if m["role"] == "system"
        else HumanMessage(content=m["content"])
        for m in messages
    ]

    model_kwargs = {
        **(model_kwargs or {}),
        "is_judge": is_judge,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if chat_model is None:
        from agents_gym_core.models.chat.clients import get_chat_model

        chat_model = get_chat_model(model, **model_kwargs)

    if response_format:
        if isinstance(response_format, type) and issubclass(response_format, BaseModel):
            structured = chat_model.with_structured_output(response_format)
            result = await structured.ainvoke(lc_messages)
            content = (
                result.model_dump_json()  # pyright: ignore[reportAttributeAccessIssue]
                if hasattr(result, "model_dump_json")
                else json.dumps(result)
            )
        elif isinstance(response_format, dict) and response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema", {})
            structured = chat_model.with_structured_output(schema, method="json_schema")
            result = await structured.ainvoke(lc_messages)
            content = json.dumps(result) if isinstance(result, dict) else str(result)
        else:
            result = await chat_model.ainvoke(lc_messages)
            content = str(result.content) if result.content else ""
        return _completion(model, ChatMessage(role="assistant", content=content, tool_calls=None))

    if tools:
        # ToolDefinition is pydantic; the OpenAI API wants 'required' omitted rather than None.
        lc_tools = [t.model_dump(exclude_none=True) for t in tools]
        bound_model = chat_model.bind_tools(
            lc_tools, tool_choice=_langchain_tool_choice(tool_choice)
        )
    else:
        bound_model = chat_model

    result = await bound_model.ainvoke(lc_messages)
    tool_calls: list[ToolCall] | None = None
    if getattr(result, "tool_calls", None):
        tool_calls = [
            ToolCall(
                id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                name=tc.get("name") or "",
                arguments=tc.get("args") or {},
            )
            for tc in result.tool_calls  # type: ignore[attr-defined]
        ]
    message = ChatMessage(
        role="assistant",
        content=str(result.content) if result.content else None,
        tool_calls=tool_calls,
    )
    return _completion(model, message, finish_reason="tool_calls" if tool_calls else "stop")


def _langchain_tool_choice(tool_choice: ToolChoice | None) -> str:
    """Map the SDK's ToolChoice onto ``bind_tools``: auto, none, any (required) or a tool name."""
    if tool_choice is None:
        return "auto"
    if isinstance(tool_choice, str):  # Literal["auto", "none"]
        return tool_choice
    if isinstance(tool_choice, SpecificToolChoice):
        return tool_choice.name
    if isinstance(tool_choice, RequiredToolChoice):
        return "any"
    return "auto"


def _completion(model: str, message: ChatMessage, finish_reason: str = "stop") -> ChatCompletion:
    return ChatCompletion(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[ChatCompletionChoice(index=0, message=message, finish_reason=finish_reason)],
        usage=ChatCompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
    )
