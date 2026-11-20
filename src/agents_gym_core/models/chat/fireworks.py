"""ChatFireworks with two fixes from langchain-ai/langchain PR #34343, applied by subclassing.

- bind_tools: honour the `strict` kwarg and forward it to convert_to_openai_tool.
- _create_chat_result: extract reasoning_content into additional_kwargs.

Imported lazily through the ModelSpec string path
"agents_gym_core.models.chat.fireworks:PatchedChatFireworks", so the rest of the package
works without langchain-fireworks installed.
"""

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_fireworks import ChatFireworks


class PatchedChatFireworks(ChatFireworks):
    """ChatFireworks with the fixes from langchain PR #34343."""

    def _create_chat_result(self, response: dict | Any) -> ChatResult:
        result = super()._create_chat_result(response)
        if not isinstance(response, dict):
            response = response.model_dump()
        for gen, res in zip(result.generations, response["choices"]):
            if isinstance(gen.message, AIMessage):
                if reasoning_content := res["message"].get("reasoning_content"):
                    gen.message.additional_kwargs.setdefault("reasoning_content", reasoning_content)
        return result

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any) -> Any:
        strict = kwargs.pop("strict", None)
        formatted_tools = [convert_to_openai_tool(tool, strict=strict) for tool in tools]
        if tool_choice is not None and tool_choice:
            if isinstance(tool_choice, str) and tool_choice not in ("auto", "any", "none"):
                tool_choice = {"type": "function", "function": {"name": tool_choice}}
            if isinstance(tool_choice, bool):
                if len(tools) > 1:
                    raise ValueError(
                        f"tool_choice can only be True when there is one tool. Received {len(tools)} tools."
                    )
                tool_choice = {
                    "type": "function",
                    "function": {"name": formatted_tools[0]["function"]["name"]},
                }
            kwargs["tool_choice"] = tool_choice
        return super(ChatFireworks, self).bind(tools=formatted_tools, **kwargs)
