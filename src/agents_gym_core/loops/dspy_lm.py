"""LangChain-to-DSPy LM adapter.

Wraps any LangChain ``BaseChatModel`` so it can be used as a ``dspy.BaseLM``.
This lets DSPy modules (ChainOfThought, ReAct, etc.) run through the gym's
model infrastructure — Azure direct, LLM Gateway, Bedrock, vLLM, etc.
"""

import time
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

try:
    from dspy import BaseLM  # pyright: ignore[reportMissingImports]
except ImportError as e:
    raise ImportError(
        "dspy is required for DSPy loops. "
        "From a source checkout, run: uv sync --locked --extra dspy"
    ) from e


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_langchain_messages(messages: list[dict[str, str]]) -> list:
    lc: list = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            lc.append(SystemMessage(content=content))
        elif role == "assistant":
            lc.append(AIMessage(content=content))
        else:
            lc.append(HumanMessage(content=content))
    return lc


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class GymLM(BaseLM):  # pyright: ignore[reportUntypedBaseClass]
    """Wraps a LangChain ``BaseChatModel`` as a ``dspy.BaseLM``.

    The wrapped model's own settings (eval temperature, reasoning, max tokens) apply. Per-call
    DSPy options are ignored, and the ``temperature=0.0`` and ``max_tokens=1000`` defaults that
    ``BaseLM`` only stores in ``lm.kwargs`` never reach the model.

    Usage::

        from agents_gym_core.models.chat.clients import get_chat_model
        chat_model = get_chat_model("gpt-4.1-mini-2025-04-14")
        lm = GymLM(chat_model, model_name="gpt-4.1-mini-2025-04-14")
        dspy.configure(lm=lm)
    """

    def __init__(
        self,
        langchain_model: BaseChatModel,
        model_name: str = "gym-model",
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model_name, **kwargs)
        self.langchain_model = langchain_model
        self._model_name = model_name

    # -- core contract --

    @staticmethod
    def _to_chat_completion(result: Any, model_name: str) -> Any:
        """Wrap a LangChain AIMessage into an OpenAI ChatCompletion."""
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage

        usage_meta = getattr(result, "usage_metadata", None) or {}
        prompt_tokens = usage_meta.get("input_tokens", 0) if isinstance(usage_meta, dict) else 0
        completion_tokens = (
            usage_meta.get("output_tokens", 0) if isinstance(usage_meta, dict) else 0
        )
        # Handle reasoning/thinking models (e.g. GPT-5) that return a list of
        # content blocks instead of a plain string.  Extract only the text blocks.
        raw_content = result.content
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            text_parts = []
            for block in raw_content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = "\n".join(text_parts) if text_parts else str(raw_content)
        else:
            content = str(raw_content)

        return ChatCompletion(
            id=f"chatcmpl-gym-{id(result)}",
            created=int(time.time()),
            model=model_name,
            object="chat.completion",
            choices=[
                Choice(
                    index=0,
                    finish_reason="stop",
                    message=ChatCompletionMessage(role="assistant", content=content),
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Call the underlying LangChain model, return OpenAI-shaped response."""
        msgs = messages or [{"role": "user", "content": prompt or ""}]
        result = self.langchain_model.invoke(_to_langchain_messages(msgs))
        return self._to_chat_completion(result, self._model_name)

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Async call the underlying LangChain model, return OpenAI-shaped response."""
        msgs = messages or [{"role": "user", "content": prompt or ""}]
        result = await self.langchain_model.ainvoke(_to_langchain_messages(msgs))
        return self._to_chat_completion(result, self._model_name)
