"""Public ("langchain") backend: every model role must resolve without UiPath gateway credentials.

Offline: clients are constructed with dummy keys and inspected; no request is sent.
"""

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
from uipath.platform.chat.llm_gateway import (
    RequiredToolChoice,
    SpecificToolChoice,
    ToolDefinition,
    ToolFunctionDefinition,
    ToolParametersDefinition,
)

from agents_gym_core.models import judge_routing
from agents_gym_core.models.chat import clients as chat_clients
from agents_gym_core.models.chat.clients import VllmChatModels, get_chat_model
from agents_gym_core.models.embeddings import clients as embedding_clients
from agents_gym_core.models.embeddings.clients import get_embedding_model
from agents_gym_core.models.model_spec import ModelSpec
from agents_gym_core.models.utils import (
    LLM_BACKEND_ENV,
    ModelBackend,
    default_backend,
    get_model_name,
)
from agents_gym_core.testbed_core import TestbedRunnerCore

PROVIDER_ENV = (
    LLM_BACKEND_ENV,
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_VERSION",
    "AGENTS_GYM_MODEL_BASE_URL",
    "AGENTS_GYM_MODEL_API_KEY",
)


def _clear_client_caches() -> None:
    for cached in (
        get_chat_model,
        get_embedding_model,
        ModelSpec.get_chat_model,
        ModelSpec.get_embedding_model,
    ):
        cached.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[pytest.MonkeyPatch]:
    for name in PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    _clear_client_caches()
    yield monkeypatch
    _clear_client_caches()


@pytest.fixture
def no_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if anything reaches the UiPath gateway's client factories."""

    def gateway(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"UiPath gateway factory called: {args} {kwargs}")

    monkeypatch.setattr(chat_clients, "get_chat_model_factory", gateway)
    monkeypatch.setattr(embedding_clients, "get_embedding_model_factory", gateway)


def test_default_backend_comes_from_the_environment(isolated_env: pytest.MonkeyPatch) -> None:
    assert default_backend() is ModelBackend.LANGCHAIN
    isolated_env.setenv(LLM_BACKEND_ENV, "uipath")
    assert default_backend() is ModelBackend.UIPATH
    isolated_env.setenv(LLM_BACKEND_ENV, "bogus")
    with pytest.raises(ValueError):
        default_backend()


def test_openai_api_with_optional_compatible_base_url(
    isolated_env: pytest.MonkeyPatch, no_gateway: None
) -> None:
    isolated_env.setenv(LLM_BACKEND_ENV, "langchain")
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

    model = get_chat_model("gpt-4.1-mini-2025-04-14")

    assert type(model) is ChatOpenAI
    assert model.model_name == "gpt-4.1-mini-2025-04-14"
    assert str(model.root_client.base_url) == "http://127.0.0.1:8000/v1/"


def test_azure_openai_when_its_credentials_are_set(
    isolated_env: pytest.MonkeyPatch, no_gateway: None
) -> None:
    isolated_env.setenv("AZURE_OPENAI_API_KEY", "azure-test")
    isolated_env.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    isolated_env.setenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    model = get_chat_model("gpt-4o-2024-11-20", backend="langchain")

    assert isinstance(model, AzureChatOpenAI)
    assert model.openai_api_version == "2024-10-21"


def test_unregistered_model_stays_off_the_gateway(
    isolated_env: pytest.MonkeyPatch, no_gateway: None
) -> None:
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    isolated_env.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")

    model = get_chat_model("my-org/served-by-vllm", backend="langchain")
    embeddings = get_embedding_model("my-org/embedder", backend="langchain")

    assert type(model) is ChatOpenAI and model.model_name == "my-org/served-by-vllm"
    assert str(model.root_client.base_url) == "http://127.0.0.1:8000/v1/"
    assert type(embeddings) is OpenAIEmbeddings and embeddings.dimensions is None


def test_a_self_hosted_model_goes_alone_to_its_endpoint(
    isolated_env: pytest.MonkeyPatch, no_gateway: None
) -> None:
    isolated_env.setenv("AZURE_OPENAI_API_KEY", "azure-test")
    isolated_env.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    isolated_env.setenv("AGENTS_GYM_MODEL_BASE_URL", "http://127.0.0.1:8002/v1")

    model = get_chat_model("my-qwen3.5-122b", backend="langchain")
    judge = get_chat_model("gpt-4o-2024-11-20", backend="langchain")
    embeddings = get_embedding_model("text-embedding-3-large", backend="langchain")

    assert type(model) is ChatOpenAI and model.model_name == "my-qwen3.5-122b"
    assert str(model.root_client.base_url) == "http://127.0.0.1:8002/v1/"
    assert isinstance(judge, AzureChatOpenAI)
    assert isinstance(embeddings, AzureOpenAIEmbeddings)


def test_embeddings_without_a_key_fail_only_when_used(
    isolated_env: pytest.MonkeyPatch, no_gateway: None
) -> None:
    embeddings = get_embedding_model("text-embedding-3-large", backend="langchain")

    assert get_model_name(embeddings) == "text-embedding-3-large"
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        embeddings.embed_query("a query")


def test_embedding_dimensions_follow_the_model(
    isolated_env: pytest.MonkeyPatch, no_gateway: None
) -> None:
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    large = get_embedding_model("text-embedding-3-large", backend="langchain")
    ada = get_embedding_model("text-embedding-ada-002", backend="langchain")

    assert type(large) is OpenAIEmbeddings and large.dimensions == 3072
    assert type(ada) is OpenAIEmbeddings and ada.dimensions is None  # ada-002 takes none

    isolated_env.setenv("AZURE_OPENAI_API_KEY", "azure-test")
    isolated_env.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    _clear_client_caches()
    assert isinstance(
        get_embedding_model("text-embedding-3-small", backend="langchain"), AzureOpenAIEmbeddings
    )


async def test_testbed_flag_becomes_the_process_default(isolated_env: pytest.MonkeyPatch) -> None:
    """--model_backend alone (no env var) must reach judges and tool-internal models too."""
    from agents_gym_core import testbed_core

    class _StopError(Exception):
        pass

    seen: dict[str, object] = {}

    def capture_orchestrator(*args: object, **kwargs: object) -> None:
        seen["default_backend"] = default_backend()
        seen["env"] = os.environ.get(LLM_BACKEND_ENV)
        raise _StopError

    isolated_env.setattr(testbed_core, "ExperimentOrchestrator", capture_orchestrator)
    isolated_env.setattr(
        sys, "argv", ["testbed", "--agents", "calculator", "--model_backend", "langchain"]
    )

    with pytest.raises(_StopError):
        await TestbedRunnerCore().run()

    assert seen == {"default_backend": ModelBackend.LANGCHAIN, "env": "langchain"}


class _FakeChat:
    def __init__(self, calls: dict[str, Any]) -> None:
        self.calls = calls

    def bind_tools(self, tools: list[dict[str, Any]], tool_choice: str) -> "_FakeChat":
        self.calls["tool_choice"] = tool_choice
        return self

    async def ainvoke(self, messages: object) -> AIMessage:
        return AIMessage(content="ok")


def _tool() -> ToolDefinition:
    return ToolDefinition(
        function=ToolFunctionDefinition(
            name="score", parameters=ToolParametersDefinition(type="object", properties={})
        )
    )


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("none", "none"),
        (RequiredToolChoice(), "any"),
        (SpecificToolChoice(name="score"), "score"),
    ],
)
async def test_judge_routing_keeps_tool_choice(
    isolated_env: pytest.MonkeyPatch, choice: Any, expected: str
) -> None:
    calls: dict[str, Any] = {}

    def fake_get_chat_model(model: str, **kwargs: Any) -> _FakeChat:
        calls.update(model=model, kwargs=kwargs)
        return _FakeChat(calls)

    isolated_env.setattr(chat_clients, "get_chat_model", fake_get_chat_model)

    await judge_routing.routed_llm_service(
        [{"role": "user", "content": "x"}], is_judge=True, tools=[_tool()], tool_choice=choice
    )

    assert calls["tool_choice"] == expected
    assert calls["model"] == "gpt-4o-2024-11-20"  # the evaluator's own model
    assert "backend" not in calls["kwargs"]  # judges follow the process default backend


def test_sql_tool_model_follows_backend(isolated_env: pytest.MonkeyPatch, no_gateway: None) -> None:
    from agents_gym_core.agents.sql import tools as sql_tools

    isolated_env.setenv(LLM_BACKEND_ENV, "langchain")
    isolated_env.setenv("OPENAI_API_KEY", "sk-test")
    model = sql_tools._get_default_llm()
    assert type(model) is ChatOpenAI and model.model_name == "gpt-4o-2024-11-20"


def test_qwen_on_vllm_is_registered_with_thinking_off(isolated_env: pytest.MonkeyPatch) -> None:
    spec = VllmChatModels["Qwen/Qwen3.5-9B"]
    assert spec.default_config["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    model = get_chat_model("Qwen/Qwen3.5-9B", backend="langchain")
    assert isinstance(model, ChatOpenAI)
    assert model.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


# Run in a fresh interpreter: the specs read the environment when the registry is imported.
_OPEN_WEIGHT_SPEC_PROBE = """
import json, sys
from agents_gym_core.models.chat.clients import VllmChatModels
spec = VllmChatModels[sys.argv[1]]
model = spec.get_chat_model()
print(json.dumps({
    "model": model.model_name,
    "base_url": spec.default_config["base_url"],
    "extra_body": spec.default_config["extra_body"],
    "key": model.openai_api_key.get_secret_value(),
    "key_in_repr": "sk-endpoint-test" in repr(spec),
}))
"""
_QWEN = "Qwen/Qwen3.5-9B"
_DEEPSEEK = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
_HOSTED = "https://api.fireworks.ai/inference/v1"


@pytest.mark.parametrize(
    ("alias", "env", "expected"),
    [
        (
            _QWEN,
            {"QWEN_9B_VLLM_API_KEY": ""},  # an empty key in .env falls back to EMPTY
            {
                "model": _QWEN,
                "base_url": "http://localhost:8002/v1",
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                "key": "EMPTY",
                "key_in_repr": False,
            },
        ),
        (
            _QWEN,
            {
                "QWEN_9B_VLLM_BASE_URL": _HOSTED,
                "QWEN_9B_VLLM_MODEL": "accounts/example/models/qwen3p5-9b",
                "QWEN_9B_VLLM_API_KEY": "sk-endpoint-test",
            },
            {
                "model": "accounts/example/models/qwen3p5-9b",
                "base_url": _HOSTED,
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                "key": "sk-endpoint-test",
                "key_in_repr": False,
            },
        ),
        (
            _DEEPSEEK,
            {"DEEPSEEK_V41_FLASH_VLLM_API_KEY": ""},
            {
                "model": _DEEPSEEK,
                "base_url": "http://localhost:8003/v1",
                "extra_body": {"chat_template_kwargs": {"thinking": True}},
                "key": "EMPTY",
                "key_in_repr": False,
            },
        ),
        (
            _DEEPSEEK,
            {
                "DEEPSEEK_V41_FLASH_VLLM_BASE_URL": _HOSTED,
                "DEEPSEEK_V41_FLASH_VLLM_MODEL": "served-deepseek",
                "DEEPSEEK_V41_FLASH_VLLM_API_KEY": "sk-endpoint-test",
                "DEEPSEEK_V41_FLASH_VLLM_THINKING": "0",
            },
            {
                "model": "served-deepseek",
                "base_url": _HOSTED,
                "extra_body": {"chat_template_kwargs": {"thinking": False}},
                "key": "sk-endpoint-test",
                "key_in_repr": False,
            },
        ),
    ],
    ids=["qwen-defaults", "qwen-hosted", "deepseek-defaults", "deepseek-hosted"],
)
def test_open_weight_models_read_their_endpoint_from_the_environment(
    alias: str, env: dict[str, str], expected: dict[str, object]
) -> None:
    prefixes = ("QWEN_9B_VLLM_", "DEEPSEEK_V41_FLASH_VLLM_")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(prefixes)}
    done = subprocess.run(
        [sys.executable, "-c", _OPEN_WEIGHT_SPEC_PROBE, alias],
        env={**clean, **env},
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    assert json.loads(done.stdout.strip().splitlines()[-1]) == expected
