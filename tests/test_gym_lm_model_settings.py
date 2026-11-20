"""GymLM ignores DSPy's per-call options, so the wrapped model's settings apply.

DSPy's ``BaseLM`` sends ``temperature=0.0`` and ``max_tokens=1000`` with every call. Passed
through, they would override the eval temperature and truncate long reasoning (AIME answers run
to thousands of tokens).
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agents_gym_core.loops.dspy_lm import GymLM


class _Recorder(BaseChatModel):
    calls: list[dict[str, Any]] = []

    @property
    def _llm_type(self) -> str:
        return "recorder"

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any):
        self.calls.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="ok"))])


def test_dspy_call_options_do_not_reach_the_model() -> None:
    model = _Recorder(calls=[])
    lm = GymLM(model, model_name="m")

    lm(messages=[{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=1000)

    assert model.calls == [{}]
