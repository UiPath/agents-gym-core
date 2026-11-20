"""Synthetic DSPy cache-policy checks; GymLM already bypasses DSPy's cache."""

import os
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from packaging.requirements import Requirement
from pydantic import Field

if os.environ.get("DSPY_SECURITY_REQUIRED") == "1":
    import dspy
else:
    dspy = pytest.importorskip("dspy")

from diskcache import FanoutCache  # noqa: E402
from dspy.clients.cache import Cache  # noqa: E402

from agents_gym_core.loops.dspy_lm import GymLM  # noqa: E402


class _Recorder(BaseChatModel):
    calls: list[tuple[list[BaseMessage], dict[str, Any]]] = Field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 8000
    content: Any = "[[ ## answer ## ]]\n42\n[[ ## completed ## ]]"

    @property
    def _llm_type(self) -> str:
        return "synthetic-cache-fixture"

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.calls.append((messages, kwargs))
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=self.content,
                        usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
                    )
                )
            ]
        )


@pytest.fixture
def process_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    cache = Cache(
        enable_disk_cache=True,
        enable_memory_cache=True,
        disk_cache_dir=str(tmp_path / "cache"),
        memory_max_entries=7,
    )
    disk_cache = cache.disk_cache
    assert isinstance(disk_cache, FanoutCache)
    monkeypatch.setattr(dspy, "cache", cache)
    try:
        yield cache
    finally:
        disk_cache.close()


def test_published_dspy_extra_retains_core_litellm_security_range() -> None:
    config = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    requirements = [
        Requirement(value) for value in config["project"]["optional-dependencies"]["dspy"]
    ]
    requirement = next(value for value in requirements if value.name == "litellm")
    assert requirement.marker is None
    for accepted in ("1.109.0", "1.109.1"):
        assert requirement.specifier.contains(accepted)
    for refused in ("1.82.7", "1.82.8", "1.96.1", "1.105.1", "1.107.0", "1.108.11", "1.110.0"):
        assert not requirement.specifier.contains(refused)


@pytest.mark.parametrize("gym_first", [True, False], ids=["gym-first", "dspy-first"])
def test_default_mixed_lms_preserve_process_cache(
    gym_first: bool, process_cache: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    import dspy.clients.lm as sdk_lm
    from litellm import ModelResponse

    completion = Mock()

    def synthetic_completion(**kwargs: Any) -> Any:
        completion()
        return ModelResponse(
            choices=[
                {"message": {"role": "assistant", "content": "fixture"}, "finish_reason": "stop"}
            ]
        )

    monkeypatch.setattr(sdk_lm, "litellm_completion", synthetic_completion)
    configure = Mock(side_effect=AssertionError("construction must not configure DSPy caching"))
    monkeypatch.setattr(dspy, "configure_cache", configure)
    disk_cache = process_cache.disk_cache
    close = Mock(wraps=disk_cache.close)
    monkeypatch.setattr(disk_cache, "close", close)
    memory_cache = process_cache.memory_cache
    request = {"fixture": "existing entry"}
    process_cache.put(request, "keep")
    if gym_first:
        GymLM(_Recorder())
    sdk = dspy.LM("openai/synthetic-model", cache=True)
    sdk.forward(prompt="Return the fixture")
    GymLM(_Recorder())
    GymLM(_Recorder())
    sdk.forward(prompt="Return the fixture")

    assert dspy.cache is process_cache
    assert process_cache.memory_cache is memory_cache
    assert memory_cache.maxsize == 7
    assert process_cache.get(request) == "keep"
    assert process_cache.enable_memory_cache
    assert process_cache.enable_disk_cache
    assert process_cache.disk_cache is disk_cache
    configure.assert_not_called()
    close.assert_not_called()
    assert completion.call_count == 1
    # The ordinary DSPy LM can still resume from disk after a memory miss.
    process_cache.reset_memory_cache()
    sdk.forward(prompt="Return the fixture")
    assert completion.call_count == 1


def test_construction_leaves_disabled_cache_untouched(
    process_cache: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    process_cache.enable_disk_cache = False
    process_cache.memory_cache["existing"] = "keep"
    memory_cache = process_cache.memory_cache
    disk_cache = process_cache.disk_cache
    close = Mock(wraps=disk_cache.close)
    monkeypatch.setattr(disk_cache, "close", close)
    original_state = vars(process_cache).copy()

    GymLM(_Recorder())

    assert dspy.cache is process_cache
    assert vars(process_cache) == original_state
    assert process_cache.memory_cache is memory_cache
    assert memory_cache["existing"] == "keep"
    assert process_cache.disk_cache is disk_cache
    close.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_real_predict_bypasses_cache_get_and_put(
    asynchronous: bool, process_cache: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _Recorder()
    lm = GymLM(model, model_name="synthetic-model")
    get = Mock(side_effect=AssertionError("GymLM must not read DSPy's cache"))
    put = Mock(side_effect=AssertionError("GymLM must not write DSPy's cache"))
    monkeypatch.setattr(dspy.cache, "get", get)
    monkeypatch.setattr(dspy.cache, "put", put)

    with dspy.context(lm=lm):
        predictor = dspy.Predict("question -> answer")
        for _ in range(2):
            if asynchronous:
                prediction = await predictor.acall(question="Return the synthetic fixture")
            else:
                prediction = predictor(question="Return the synthetic fixture")
            assert prediction.answer == "42"

    assert len(model.calls) == 2
    assert all(kwargs == {} for _, kwargs in model.calls)
    assert dspy.cache is process_cache
    assert dspy.cache.enable_disk_cache
    get.assert_not_called()
    put.assert_not_called()


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.asyncio
async def test_adapter_preserves_core_content_usage_and_model_settings(asynchronous: bool) -> None:
    model = _Recorder(
        content=[
            {"type": "reasoning", "summary": "not answer text"},
            {"type": "text", "text": "first"},
            "second",
        ]
    )
    lm = GymLM(model, model_name="synthetic-model")
    messages = [
        {"role": "system", "content": "Synthetic system"},
        {"role": "user", "content": "Synthetic question"},
    ]
    if asynchronous:
        result = await lm.aforward(messages=messages, temperature=0.0, max_tokens=1000)
    else:
        result = lm.forward(messages=messages, temperature=0.0, max_tokens=1000)
    assert result.model == "synthetic-model"
    assert result.choices[0].message.content == "first\nsecond"
    assert result.usage.prompt_tokens == 3
    assert result.usage.completion_tokens == 2
    assert result.usage.total_tokens == 5
    messages, kwargs = model.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert kwargs == {}
    assert model.temperature == 0.7 and model.max_tokens == 8000
