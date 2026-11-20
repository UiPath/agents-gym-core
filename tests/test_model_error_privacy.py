"""Model construction errors identify the model without dumping its configuration."""

import socket
from typing import Any, NoReturn

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_openai import AzureChatOpenAI
from pydantic import ValidationError

from agents_gym_core.models.model_spec import ModelSpec
from agents_gym_core.utils.general_utils import FrozenDict


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("model error regressions must stay offline")

    monkeypatch.setattr(socket.socket, "connect", reject)


@pytest.mark.parametrize("factory_method", [None, "from_config"])
@pytest.mark.parametrize("shape", ["dict", "frozen_dict", "list"])
def test_failed_model_construction_does_not_dump_nested_config(
    shape: str, factory_method: str | None
) -> None:
    secret = f"synthetic-private-marker-{shape}"
    failure = RuntimeError("synthetic provider initialization failure")
    calls: list[dict[str, Any]] = []

    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            raise failure

        @classmethod
        def from_config(cls, **kwargs: Any) -> NoReturn:
            calls.append(kwargs)
            raise failure

    nested: Any
    if shape == "dict":
        nested = {"connection": {"token": secret}}
    elif shape == "frozen_dict":
        nested = FrozenDict({"connection": FrozenDict({"token": secret})})
    else:
        nested = [{"connection": {"token": secret}}]
    spec = ModelSpec(
        "synthetic-model-identity",
        FailingChatModel,
        max_concurrency=None,
        factory_method=factory_method,
    )
    config = {"model": spec.model_name, "private_options": nested}
    with pytest.raises(ValueError, match="Failed to instantiate") as error:
        spec._instantiate(config)

    assert secret not in str(error.value)
    assert "private_options" not in str(error.value)
    assert spec.model_name in str(error.value)
    assert FailingChatModel.__name__ in str(error.value)
    assert error.value.__cause__ is failure
    assert calls == [config]


@pytest.mark.parametrize("factory_method", [None, "from_config"])
def test_public_chat_model_failure_does_not_dump_default_or_override_config(
    factory_method: str | None,
) -> None:
    failure = RuntimeError("synthetic constructor failure")

    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise failure

        @classmethod
        def from_config(cls, **kwargs: Any) -> NoReturn:
            raise failure

    default_secret = "synthetic-default-private-marker"
    override_secret = "synthetic-override-private-marker"
    spec = ModelSpec(
        "synthetic-public-model",
        FailingChatModel,
        default_config={"custom_defaults": {"token": default_secret}},
        max_concurrency=None,
        factory_method=factory_method,
    )
    with pytest.raises(ValueError) as error:
        spec.get_chat_model(custom_overrides=[FrozenDict({"token": override_secret})])

    assert default_secret not in str(error.value)
    assert override_secret not in str(error.value)
    assert spec.model_name in str(error.value)
    assert FailingChatModel.__name__ in str(error.value)
    assert error.value.__cause__ is failure


@pytest.mark.parametrize(
    "value",
    [
        "synthetic-private-marker",
        {"token": "synthetic-private-marker"},
        ["synthetic-private-marker"],
        True,
        float("nan"),
        float("inf"),
        10**30,
    ],
    ids=["string", "dict", "list", "boolean", "nan", "infinity", "large-integer"],
)
def test_unsafe_diagnostic_values_are_not_formatted(value: Any) -> None:
    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("synthetic failure")

    spec = ModelSpec("bounded-diagnostics", FailingChatModel, max_concurrency=None)
    with pytest.raises(ValueError) as error:
        spec._instantiate({"max_tokens": value})
    assert "max_tokens" not in str(error.value)
    assert "synthetic-private-marker" not in str(error.value)


def test_diagnostics_never_call_arbitrary_repr() -> None:
    class PrivateValue:
        def __repr__(self) -> str:
            raise AssertionError("private configuration repr must not run")

    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("synthetic failure")

    spec = ModelSpec("bounded-diagnostics", FailingChatModel, max_concurrency=None)
    with pytest.raises(ValueError) as error:
        spec._instantiate({"temperature": PrivateValue(), "private_options": PrivateValue()})
    assert "settings {}" in str(error.value)


def test_diagnostics_do_not_format_custom_mapping_keys() -> None:
    class PrivateKey(str):
        def __repr__(self) -> str:
            raise AssertionError("private key repr must not run")

    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("synthetic failure")

    spec = ModelSpec("bounded-diagnostics", FailingChatModel, max_concurrency=None)
    with pytest.raises(ValueError) as error:
        spec._instantiate({PrivateKey("max_tokens"): 5})
    assert "settings {}" in str(error.value)


def test_safe_numeric_diagnostics_are_preserved() -> None:
    failure = RuntimeError("synthetic failure")

    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise failure

    spec = ModelSpec("bounded-diagnostics", FailingChatModel, max_concurrency=None)
    with pytest.raises(ValueError) as error:
        spec._instantiate(
            {
                "temperature": 0.5,
                "max_tokens": 5,
                "max_retries": 2,
                "request_timeout": 300,
                "headers": {"Authorization": "synthetic-private-marker"},
            }
        )
    message = str(error.value)
    assert "'max_tokens': 5" in message
    assert "'temperature': 0.5" in message
    assert "'max_retries': 2" in message
    assert "'request_timeout': 300" in message
    assert "synthetic-private-marker" not in message
    assert "headers" not in message
    assert error.value.__cause__ is failure


@pytest.mark.parametrize(
    ("model_param", "version_key", "deployment_key", "endpoint_key"),
    [
        ("model", "api_version", "azure_deployment", "azure_endpoint"),
        ("model_name", "openai_api_version", "deployment_name", "azure_endpoint"),
        ("model", "api_version", "azure_deployment", "base_url"),
        ("model_name", "openai_api_version", "deployment_name", "openai_api_base"),
    ],
)
def test_failed_azure_constructor_keeps_final_aliases_and_only_endpoint_host(
    model_param: str,
    version_key: str,
    deployment_key: str,
    endpoint_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_client(*args: Any, **kwargs: Any) -> NoReturn:
        pytest.fail("invalid Azure configuration must fail before creating a provider client")

    monkeypatch.setattr("openai.AzureOpenAI", reject_client)
    monkeypatch.setattr("openai.AsyncAzureOpenAI", reject_client)
    spec = ModelSpec(
        "gpt-4o-2024-11-20",
        AzureChatOpenAI,
        default_config={
            "api_version": "old-version",
            "azure_deployment": "old-deployment",
            "azure_endpoint": "https://old.invalid/",
        },
        max_concurrency=None,
        model_param=model_param,
        kwarg_map={
            "api_version": version_key,
            "azure_deployment": deployment_key,
            "azure_endpoint": endpoint_key,
        },
    )
    with pytest.raises(ValueError, match="Failed to instantiate AzureChatOpenAI") as error:
        spec.get_chat_model(
            n=0,  # Azure's local validator rejects this before constructing any API client.
            api_version="2025-03-01-preview",
            azure_deployment="wrong-deployment-west",
            azure_endpoint=(
                "https://synthetic-user:synthetic-password@wrong.openai.azure.com:8443/"
                "private-path?api_key=synthetic-query#synthetic-fragment"
            ),
            api_key="synthetic-api-key",
        )

    message = str(error.value)
    assert f"'{model_param}': 'gpt-4o-2024-11-20'" in message
    assert f"'{version_key}': '2025-03-01-preview'" in message
    assert f"'{deployment_key}': 'wrong-deployment-west'" in message
    assert f"'{endpoint_key}_host': 'wrong.openai.azure.com'" in message
    for private in (
        "synthetic-user",
        "synthetic-password",
        "private-path",
        "synthetic-query",
        "synthetic-fragment",
        "synthetic-api-key",
        "8443",
        "https://",
        "api_key",
        "old-version",
        "old-deployment",
        "old.invalid",
    ):
        assert private not in message
    assert isinstance(error.value.__cause__, ValidationError)
    assert "n must be at least 1" in error.value.__cause__.errors()[0]["msg"]


@pytest.mark.parametrize(
    "key",
    [
        "model",
        "model_name",
        "model_id",
        "api_version",
        "openai_api_version",
        "deployment",
        "deployment_name",
        "deployment_id",
        "azure_deployment",
        "azure_deployment_name",
    ],
)
def test_failed_constructor_preserves_allowlisted_operational_identifiers(key: str) -> None:
    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("synthetic failure")

    spec = ModelSpec("bounded-diagnostics", FailingChatModel, max_concurrency=None)
    identifier = "org/Qwen3.5-9B:v1"
    with pytest.raises(ValueError) as error:
        spec._instantiate({key: identifier})
    assert f"'{key}': '{identifier}'" in str(error.value)


@pytest.mark.parametrize(
    "key", ["azure_endpoint", "base_url", "openai_api_base", "api_base", "endpoint", "endpoint_url"]
)
@pytest.mark.parametrize(
    ("endpoint", "host"),
    [
        (
            "https://user:password@deployment.example.invalid:8443/private?token=private#private",
            "deployment.example.invalid",
        ),
        ("http://localhost:8002/v1", "localhost"),
        ("http://127.0.0.1:8002/v1", "127.0.0.1"),
        ("https://[::1]:8443/private", "::1"),
    ],
)
def test_failed_constructor_reports_only_structured_endpoint_host(
    key: str, endpoint: str, host: str
) -> None:
    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("synthetic failure")

    spec = ModelSpec("bounded-diagnostics", FailingChatModel, max_concurrency=None)
    with pytest.raises(ValueError) as error:
        spec._instantiate({key: endpoint})
    assert str(error.value).endswith(f"settings {{'{key}_host': '{host}'}}")


@pytest.mark.parametrize(
    "value",
    [
        "",
        "a" * 201,
        "identifier with spaces",
        "identifier\nprivate",
        "identifier\tprivate",
        "identifier?token=private",
        "identifier#private",
        "user:password@host",
        "https://private/path",
        "non-ascii-\u00e9",
        7,
        0.5,
        True,
        None,
        {"token": "private"},
        FrozenDict({"token": "private"}),
        ["private"],
    ],
)
def test_failed_constructor_omits_unsafe_identifier_values(value: Any) -> None:
    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("synthetic failure")

    spec = ModelSpec(value, FailingChatModel, max_concurrency=None)
    with pytest.raises(ValueError) as error:
        spec._instantiate({"model": value, "api_version": value, "deployment_name": value})
    assert str(error.value).endswith("with model <redacted> and settings {}")


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "not-a-url",
        "//user:password@host/path",
        "file:///private/path",
        "https:///private",
        "https://[malformed/private",
        "https://[::1",
        "https://host\n.private/path",
        "https://host\t.private/path",
        "https://host%40private/path",
        "https://non-ascii-\u00e9.invalid/path",
        "https://" + "a" * 254,
        "https://host/" + "a" * 4096,
    ],
)
def test_failed_constructor_omits_malformed_or_unbounded_endpoints(endpoint: str) -> None:
    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("synthetic failure")

    spec = ModelSpec("bounded-diagnostics", FailingChatModel, max_concurrency=None)
    with pytest.raises(ValueError) as error:
        spec._instantiate({"base_url": endpoint})
    assert str(error.value).endswith("settings {}")


@pytest.mark.parametrize(
    "credential_key",
    ["api_key", "openai_api_key", "azure_ad_token", "client_secret", "Authorization", "password"],
)
def test_identifiers_matching_direct_credential_values_are_omitted(credential_key: str) -> None:
    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("synthetic failure")

    secret = "synthetic-credential.invalid"
    spec = ModelSpec(secret, FailingChatModel, max_concurrency=None)
    with pytest.raises(ValueError) as error:
        spec._instantiate(
            {
                credential_key: secret,
                "model": secret,
                "model_name": secret,
                "deployment_name": secret,
                "api_version": secret,
                "base_url": f"https://{secret}/private",
            }
        )
    assert secret not in str(error.value)
    assert str(error.value).endswith("with model <redacted> and settings {}")


@pytest.mark.parametrize("header_key", ["headers", "default_headers", "extra_headers"])
def test_identifier_matching_plain_header_value_is_omitted(header_key: str) -> None:
    class FailingChatModel(FakeListChatModel):
        def __init__(self, **kwargs: Any) -> None:
            raise RuntimeError("synthetic failure")

    secret = "synthetic-header-credential"
    spec = ModelSpec("bounded-diagnostics", FailingChatModel, max_concurrency=None)
    with pytest.raises(ValueError) as error:
        spec._instantiate({header_key: {"Authorization": secret}, "deployment_name": secret})
    message = str(error.value)
    assert secret not in message
    assert "Authorization" not in message
    assert header_key not in message
    assert message.endswith("settings {}")


@pytest.mark.parametrize("base", [object, str, int, float])
def test_failed_constructor_never_formats_objects_or_primitive_subclasses(base: type) -> None:
    class HostileFormatting:
        def __str__(self) -> NoReturn:
            raise AssertionError("custom str must not execute")

        def __repr__(self) -> NoReturn:
            raise AssertionError("custom repr must not execute")

        def __format__(self, spec: str) -> NoReturn:
            raise AssertionError("custom format must not execute")

    class HostileMeta(type(FakeListChatModel)):
        def __str__(cls) -> NoReturn:
            raise AssertionError("class str must not execute")

        def __repr__(cls) -> NoReturn:
            raise AssertionError("class repr must not execute")

    failure = RuntimeError("synthetic failure")

    class FailingChatModel(FakeListChatModel, metaclass=HostileMeta):
        def __init__(self, **kwargs: Any) -> None:
            raise failure

    value: Any = type("HostileValue", (HostileFormatting, base), {})()
    spec = ModelSpec(value, FailingChatModel, max_concurrency=None)
    with pytest.raises(ValueError) as error:
        spec._instantiate(
            {
                "model": value,
                "api_version": value,
                "deployment_name": value,
                "base_url": value,
                "max_tokens": value,
                "api_key": value,
                "private_options": {"nested": [value]},
                "headers": {"Authorization": value},
            }
        )
    assert str(error.value).endswith("with model <redacted> and settings {}")
    assert error.value.__cause__ is failure
