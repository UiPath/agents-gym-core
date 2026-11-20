"""Offline checks for the GEPA runners' path and model plumbing."""

import importlib.util
from pathlib import Path

import dspy
import pytest

from agents_gym_core.loops.dspy_lm import GymLM
from agents_gym_core.models.chat.clients import get_chat_model
from agents_gym_core.models.model_spec import ModelSpec
from agents_gym_core.models.utils import LLM_BACKEND_ENV
from agents_gym_core.prompt_optimization.gepa.benchmark import ensure_splits
from agents_gym_core.prompt_optimization.gepa_dspy.lm import make_dspy_lm
from agents_gym_core.utils.directory_provider import CoreDirectoryProvider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        LLM_BACKEND_ENV,
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
    ):
        monkeypatch.delenv(name, raising=False)
    for cached in (get_chat_model, ModelSpec.get_chat_model):
        cached.cache_clear()  # type: ignore[attr-defined]


def test_ensure_splits_runs_init_only_when_a_split_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded: list[str] = []
    load = importlib.util.spec_from_file_location

    def recording_load(name: str, location: Path) -> object:
        loaded.append(Path(location).name)
        return load(name, location)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", recording_load)
    agents = CoreDirectoryProvider().get_agents_dir()
    for agent in ("sql", "claim_history", "meeting_management"):
        eval_sets = agents / agent / "evaluations" / "eval-sets"
        assert (eval_sets / f"evaluation-set-{agent}_train.json").is_file()
        assert (eval_sets / f"evaluation-set-{agent}_test.json").is_file()
        ensure_splits(agent)
    assert loaded == []

    toy = tmp_path / "toy"
    (toy / "evaluations" / "eval-sets").mkdir(parents=True)
    (toy / "evaluations" / "eval-sets" / "evaluation-set-toy_train.json").write_text("{}")
    (toy / "init.py").write_text(
        "def init(agent_dir, params):\n    (agent_dir / 'ran').write_text(repr(params))\n"
    )
    monkeypatch.setattr(CoreDirectoryProvider, "get_agents_dir", lambda self: tmp_path)
    ensure_splits("toy")
    assert loaded == ["init.py"]
    assert (toy / "ran").read_text() == "{}"


def test_reflection_lm_uses_litellm_on_the_public_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LLM_BACKEND_ENV, "langchain")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

    lm = make_dspy_lm("gpt-5-2025-08-07", temperature=1.0, max_tokens=16384)

    assert isinstance(lm, dspy.LM) and lm.model == "openai/gpt-5-2025-08-07"
    assert lm.kwargs["api_base"] == "http://127.0.0.1:8000/v1"
    assert lm.kwargs["temperature"] == 1.0
    # dspy renames max_tokens for reasoning models
    assert lm.kwargs.get("max_tokens", lm.kwargs.get("max_completion_tokens")) == 16384

    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    assert make_dspy_lm("gpt-5-2025-08-07").model == "azure/gpt-5-2025-08-07"


def test_reflection_lm_wraps_the_registry_model_on_the_gateway(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agents_gym_core.prompt_optimization.gepa_dspy import lm as lm_module

    calls: dict[str, object] = {}
    sentinel = object()

    def fake_get_chat_model(name: str, **kwargs: object) -> object:
        calls.update(name=name, **kwargs)
        return sentinel

    monkeypatch.setattr(lm_module, "get_chat_model", fake_get_chat_model)
    monkeypatch.setenv(LLM_BACKEND_ENV, "uipath")

    lm = make_dspy_lm("gpt-4o-2024-11-20", temperature=0.5, max_tokens=1000)

    assert isinstance(lm, GymLM) and lm.langchain_model is sentinel
    assert calls == {"name": "gpt-4o-2024-11-20", "temperature": 0.5, "max_tokens": 1000}
