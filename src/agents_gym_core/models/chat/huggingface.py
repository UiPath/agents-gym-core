"""ChatHuggingFace integration for pipeline tool calling.

For pipeline models, this subclass passes tools to the chat template, serializes ToolMessage
objects and parses <tool_call> blocks from generated text. Endpoint and TGI models use the
parent implementation.
"""

import json
import re
import uuid
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult, LLMResult
from langchain_core.runnables.config import run_in_executor

try:
    from langchain_huggingface import ChatHuggingFace  # pyright: ignore[reportMissingImports]
    from langchain_huggingface.llms.huggingface_pipeline import (  # pyright: ignore[reportMissingImports]
        HuggingFacePipeline,
    )
except ImportError:
    raise ImportError(
        "langchain-huggingface is required for HuggingFace models. "
        "From a source checkout, run: uv sync --locked --extra huggingface"
    )

from agents_gym_core.utils.gym_settings_core import gym_settings_core

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


class PatchedChatHuggingFace(ChatHuggingFace):  # pyright: ignore[reportUntypedBaseClass]
    """ChatHuggingFace with tool serialization and parsing for local pipelines."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        if not isinstance(self.llm, HuggingFacePipeline):
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, stream=stream, **kwargs
            )

        messages_dicts = self._to_message_dicts(messages)
        # Chat templates require alternating user/assistant turns.
        # Append a user turn when continuing after an assistant message.
        if messages_dicts and messages_dicts[-1]["role"] == "assistant":
            messages_dicts.append({"role": "user", "content": "You must call a tool now."})

        tools = kwargs.get("tools")
        prompt: str = self.tokenizer.apply_chat_template(
            messages_dicts,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        llm_result: LLMResult = self.llm._generate(
            prompts=[prompt],
            stop=stop,
            run_manager=run_manager,
        )
        raw_text: str = llm_result.generations[0][0].text

        tool_names = {t["function"]["name"] for t in tools} if tools else set()
        tool_calls = self._extract_tool_calls(raw_text, tool_names) if tools else []
        # Respect sequential execution by retaining the first parsed tool call.
        # Local pipelines read this setting directly rather than through bind_tools kwargs.
        if not gym_settings_core.UIPATH_AGENTS_GYM_PARALLEL_TOOL_CALLS and len(tool_calls) > 1:
            tool_calls = tool_calls[:1]
        content = _TOOL_CALL_RE.sub("", raw_text).strip() if tool_calls else raw_text

        message = AIMessage(content=content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        if not isinstance(self.llm, HuggingFacePipeline):
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, stream=stream, **kwargs
            )
        return await run_in_executor(
            None,
            self._generate,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            stream=stream,
            **kwargs,
        )

    @staticmethod
    def _to_message_dicts(messages: list[BaseMessage]) -> list[dict]:
        """Convert LangChain messages to ChatML dicts (handles ToolMessage)."""
        result = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": str(msg.content)})
            elif isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                d: dict = {"role": "assistant", "content": str(msg.content) if msg.content else ""}
                if msg.tool_calls:
                    d["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"], ensure_ascii=False),
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                result.append(d)
            elif isinstance(msg, ToolMessage):
                result.append(
                    {
                        "role": "tool",
                        "content": str(msg.content),
                        "tool_call_id": msg.tool_call_id,
                    }
                )
        return result

    @staticmethod
    def _extract_tool_calls(text: str, tool_names: set[str] | None = None) -> list[ToolCall]:
        """Parse <tool_call>...</tool_call> blocks from raw model output."""
        calls = []
        for match in _TOOL_CALL_RE.finditer(text):
            try:
                data = json.loads(match.group(1))
                calls.append(
                    ToolCall(
                        id=str(uuid.uuid4()),
                        name=data["name"],
                        args=data.get("arguments", data.get("args", {})),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
        # Fallback: model sometimes outputs bare JSON tool calls without <tool_call> tags.
        # Only accept if the name matches an actual tool to avoid false positives.
        if not calls and tool_names:
            try:
                data = json.loads(text.strip())
                if isinstance(data, dict) and data.get("name") in tool_names:
                    calls.append(
                        ToolCall(
                            id=str(uuid.uuid4()),
                            name=data["name"],
                            args=data.get("arguments", data.get("args", {})),
                        )
                    )
            except (json.JSONDecodeError, KeyError):
                pass
        return calls
