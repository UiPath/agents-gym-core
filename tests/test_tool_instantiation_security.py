"""Synthetic, offline regressions for tool construction and config isolation."""

import socket
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import Mock

import pytest
from hydra.errors import InstantiationException
from langchain_core.tools import BaseTool, StructuredTool, tool
from omegaconf import DictConfig, ListConfig, OmegaConf

from agents_gym_core.hydra_agent import tool_instantiator
from agents_gym_core.hydra_agent.tool_instantiator import ToolInstantiator


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("tool-loader regressions must stay offline")

    monkeypatch.setattr(socket.socket, "connect", reject)


def _echo(value: str) -> str:
    """Echo a synthetic value."""
    return value


_shared_tool = tool(_echo)


def _factory_target(name: str = "synthetic") -> BaseTool:
    return StructuredTool.from_function(_echo, name=name)


class _NestedFactories:
    factory = staticmethod(_factory_target)


@dataclass
class _StructuredPayload:
    label: str = "fixture"


class _CallableTool(BaseTool):
    name: str = "callable_fixture"
    description: str = "Synthetic callable tool instance"

    def _run(self) -> str:
        return "ready"

    def __call__(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("loading must not execute an existing tool")


_callable_tool = _CallableTool()


def _config(target: str = "_factory_target", **values: Any) -> DictConfig:
    return DictConfig({"_target_": f"{__name__}.{target}", **values})


def _replace_factory(monkeypatch: pytest.MonkeyPatch, factory: Callable[..., Any]) -> None:
    monkeypatch.setattr(sys.modules[__name__], "_factory_target", factory)


@pytest.mark.parametrize("class_target", [False, True], ids=["factory", "class"])
def test_factory_or_class_executes_once(
    class_target: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def factory(name: str) -> BaseTool:
        calls.append(name)
        return StructuredTool.from_function(_echo, name=name)

    class SyntheticTool(_CallableTool):
        def __init__(self, name: str) -> None:
            calls.append(name)
            super().__init__(name=name)

    _replace_factory(monkeypatch, SyntheticTool if class_target else factory)
    result = ToolInstantiator._instantiate_single_tool(_config(name="configured"))
    assert result.name == "configured"
    assert calls == ["configured"]


@pytest.mark.parametrize("class_target", [False, True], ids=["factory", "constructor"])
def test_failure_is_never_retried(class_target: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    failure = ValueError("synthetic construction failure")

    def factory(name: str) -> BaseTool:
        calls.append(name)
        raise failure

    class FailingTool(_CallableTool):
        def __init__(self, name: str) -> None:
            calls.append(name)
            raise failure

    _replace_factory(monkeypatch, FailingTool if class_target else factory)
    with pytest.raises(InstantiationException, match="synthetic construction failure") as error:
        ToolInstantiator._instantiate_single_tool(_config(name="configured"))
    assert error.value.__cause__ is failure
    assert calls == ["configured"]


@pytest.mark.parametrize("target", ["_shared_tool", "_callable_tool"])
def test_existing_tools_are_not_executed(target: str, monkeypatch: pytest.MonkeyPatch) -> None:
    invoke = Mock(side_effect=AssertionError("loading must not invoke a tool"))
    monkeypatch.setattr(BaseTool, "invoke", invoke)
    result = ToolInstantiator._instantiate_single_tool(_config(target))
    assert result is getattr(sys.modules[__name__], target)
    invoke.assert_not_called()


@pytest.mark.parametrize("target", ["_shared_tool", "_callable_tool"])
def test_existing_tools_reject_unused_arguments_before_nested_execution(target: str) -> None:
    # Neither the missing resolver nor the nested constructor is needed for a tool instance.
    cfg = _config(
        target,
        unused={"_target_": f"{__name__}._factory_target", "name": "${missing_resolver:}"},
    )
    with pytest.raises(TypeError, match="not fields of tool .*unused"):
        ToolInstantiator._instantiate_single_tool(cfg)


def test_existing_tool_overrides_are_version_local_and_do_not_instantiate_fields() -> None:
    original = (_shared_tool.name, _shared_tool.description, _shared_tool.metadata)
    metadata = {"_target_": f"{__name__}._factory_target", "label": "kept-as-data"}
    result = ToolInstantiator._instantiate_single_tool(
        _config(
            "_shared_tool",
            name="version_name",
            description="Version description",
            metadata=metadata,
        )
    )
    assert result is not _shared_tool
    assert result.name == "version_name"
    assert result.description == "Version description"
    assert result.metadata == metadata
    assert (_shared_tool.name, _shared_tool.description, _shared_tool.metadata) == original
    assert ToolInstantiator._instantiate_single_tool(_config("_shared_tool")) is _shared_tool


def test_factory_returned_callable_is_wrapped_without_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def echo(value: str) -> str:
        """Echo only when explicitly invoked."""
        calls.append(value)
        return value

    def factory() -> Callable[[str], str]:
        calls.append("factory")
        return echo

    _replace_factory(monkeypatch, factory)
    result = ToolInstantiator._instantiate_single_tool(_config())
    assert isinstance(result, StructuredTool)
    assert calls == ["factory"]
    assert result.invoke({"value": "runtime"}) == "runtime"
    assert calls == ["factory", "runtime"]


def test_failed_callable_wrapping_does_not_execute_zero_argument_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def undocumented() -> BaseTool:
        calls.append("unexpected callable execution")
        return _shared_tool

    def factory() -> Callable[[], BaseTool]:
        calls.append("factory")
        return undocumented

    _replace_factory(monkeypatch, factory)
    with pytest.raises(ValueError, match="docstring"):
        ToolInstantiator._instantiate_single_tool(_config())
    assert calls == ["factory"]


def test_raw_function_is_not_reinterpreted_after_construction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _replace_factory(monkeypatch, _echo)
    with pytest.raises(InstantiationException, match="value") as error:
        ToolInstantiator._instantiate_single_tool(_config())
    assert isinstance(error.value.__cause__, TypeError)


@pytest.mark.parametrize("target", ["_shared_tool", "_callable_tool", "_factory_target"])
def test_readonly_interpolated_target_is_snapshotted_once(target: str) -> None:
    calls: list[str] = []

    def changing_target() -> str:
        calls.append(target)
        return f"{__name__}.{target}" if len(calls) == 1 else f"{__name__}._echo"

    OmegaConf.register_new_resolver("tool_loader_test_target", changing_target)
    try:
        root = DictConfig(
            {
                "label": "from-parent",
                "tools": [{"_target_": "${tool_loader_test_target:}", "name": "${label}"}],
            }
        )
        before = OmegaConf.to_container(root, resolve=False)
        OmegaConf.set_readonly(root, True)
        result = ToolInstantiator._instantiate_single_tool(root.tools[0])
        assert result.name == "from-parent"
        assert calls == [target]
        assert OmegaConf.is_readonly(root.tools[0])
        assert OmegaConf.to_container(root, resolve=False) == before
    finally:
        OmegaConf.clear_resolver("tool_loader_test_target")


def test_escaped_target_interpolation_is_rejected_before_hydra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = DictConfig(
        {
            "_target_": r"\${oc.select:.real,builtins.str}",
            "real": f"{__name__}._factory_target",
        }
    )
    before = OmegaConf.to_container(cfg, resolve=False)
    OmegaConf.set_readonly(cfg, True)
    instantiate = Mock(side_effect=AssertionError("escaped target must not reach Hydra"))
    monkeypatch.setattr(tool_instantiator, "instantiate", instantiate)
    with pytest.raises(InstantiationException, match="literal import path"):
        ToolInstantiator._instantiate_single_tool(cfg)
    instantiate.assert_not_called()
    assert OmegaConf.is_readonly(cfg)
    assert OmegaConf.to_container(cfg, resolve=False) == before


@pytest.mark.parametrize("partial_value", [True, "${deferred}"])
def test_deferred_factory_is_rejected_before_nested_arguments(
    partial_value: bool | str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[bool] = []

    def factory(**kwargs: Any) -> BaseTool:
        calls.append(True)
        raise AssertionError("deferred factory must not execute")

    _replace_factory(monkeypatch, factory)
    root = DictConfig(
        {
            "deferred": True,
            "tools": [
                {
                    "_target_": f"{__name__}._factory_target",
                    "_partial_": partial_value,
                    "unused": {"_target_": f"{__name__}._factory_target"},
                }
            ],
        }
    )
    with pytest.raises(InstantiationException, match="do not support _partial_=True.*BaseTool"):
        ToolInstantiator._instantiate_single_tool(root.tools[0])
    assert calls == []


@pytest.mark.parametrize("policy", [None, "all", "none", "partial", "object"])
def test_factory_config_conversion_honors_policy(
    policy: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[Any] = []

    def factory(payload: Any) -> BaseTool:
        received.append(payload)
        return _shared_tool

    _replace_factory(monkeypatch, factory)
    cfg = _config(payload={"nested": [{"value": "fixture"}]})
    if policy is not None:
        cfg._convert_ = policy
    assert ToolInstantiator._instantiate_single_tool(cfg) is _shared_tool
    (payload,) = received
    if policy == "none":
        assert isinstance(payload, DictConfig)
        assert isinstance(payload.nested, ListConfig)
        assert isinstance(payload.nested[0], DictConfig)
    else:
        assert type(payload) is dict
        assert type(payload["nested"]) is list
        assert type(payload["nested"][0]) is dict
    assert payload == {"nested": [{"value": "fixture"}]}


@pytest.mark.parametrize("policy", [None, "all", "none", "partial", "object"])
def test_structured_factory_config_conversion_honors_policy(
    policy: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[Any] = []

    def factory(payload: Any) -> BaseTool:
        received.append(payload)
        return _shared_tool

    _replace_factory(monkeypatch, factory)
    cfg = _config(payload=OmegaConf.structured(_StructuredPayload))
    if policy is not None:
        cfg._convert_ = policy
    assert ToolInstantiator._instantiate_single_tool(cfg) is _shared_tool
    (payload,) = received
    if policy in {"none", "partial"}:
        assert isinstance(payload, DictConfig)
        assert OmegaConf.get_type(payload) is _StructuredPayload
        assert payload.label == "fixture"
    elif policy == "object":
        assert isinstance(payload, _StructuredPayload)
        assert payload.label == "fixture"
    else:
        assert type(payload) is dict
        assert payload == {"label": "fixture"}


@pytest.mark.parametrize("recursive", [False, True])
def test_nested_conversion_and_recursion_are_honored(
    recursive: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[Any] = []
    calls: list[str] = []

    def factory(name: str = "nested", payload: Any = None) -> BaseTool:
        calls.append(name)
        if name == "outer":
            received.append(payload)
        return _shared_tool

    _replace_factory(monkeypatch, factory)
    nested = {"_target_": f"{__name__}._factory_target", "name": "nested"}
    cfg = _config(name="outer", payload=nested, _recursive_=recursive)
    assert ToolInstantiator._instantiate_single_tool(cfg) is _shared_tool
    assert calls == (["nested", "outer"] if recursive else ["outer"])
    assert received == ([_shared_tool] if recursive else [nested])


def test_nested_factory_conversion_overrides_outer_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[Any] = []

    def factory(payload: Any) -> BaseTool:
        received.append(payload)
        return _shared_tool

    _replace_factory(monkeypatch, factory)
    cfg = _config(
        payload={
            "_target_": f"{__name__}._factory_target",
            "_convert_": "none",
            "payload": {"value": "fixture"},
        }
    )
    assert ToolInstantiator._instantiate_single_tool(cfg) is _shared_tool
    assert isinstance(received[0], DictConfig)
    assert received[0].value == "fixture"
    assert received[1] is _shared_tool


def test_nested_class_factory_import_path_remains_supported() -> None:
    assert (
        ToolInstantiator._instantiate_single_tool(
            _config("_NestedFactories.factory", name="nested")
        ).name
        == "nested"
    )


def test_hydra_rejection_is_not_retried_or_looked_up(monkeypatch: pytest.MonkeyPatch) -> None:
    from hydra._internal.instantiate import _instantiate2

    failure = InstantiationException("synthetic Hydra target restriction")
    resolve = Mock(side_effect=failure)
    lookup = Mock(side_effect=AssertionError("rejected target must not be looked up"))
    calls: list[bool] = []

    def factory() -> BaseTool:
        calls.append(True)
        raise AssertionError("rejected target must not execute")

    _replace_factory(monkeypatch, factory)
    monkeypatch.setattr(_instantiate2, "_resolve_target", resolve)
    monkeypatch.setattr(tool_instantiator, "get_object", lookup)
    with pytest.raises(InstantiationException, match="synthetic Hydra target restriction") as error:
        ToolInstantiator._instantiate_single_tool(_config())
    assert error.value is failure
    resolve.assert_called_once()
    lookup.assert_not_called()
    assert calls == []


def test_nested_hydra_rejection_never_falls_back_to_raw_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hydra._internal.instantiate import _instantiate2

    original_resolve = _instantiate2._resolve_target

    def resolve(target: Any, full_key: str) -> Any:
        if target == "synthetic_rejected.nested":
            raise InstantiationException("synthetic nested Hydra restriction")
        return original_resolve(target, full_key)

    calls: list[bool] = []

    def factory(payload: Any) -> BaseTool:
        calls.append(True)
        raise AssertionError("outer factory must not execute")

    _replace_factory(monkeypatch, factory)
    monkeypatch.setattr(_instantiate2, "_resolve_target", resolve)
    with pytest.raises(InstantiationException, match="synthetic nested Hydra restriction"):
        ToolInstantiator._instantiate_single_tool(
            _config(payload={"_target_": "synthetic_rejected.nested"})
        )
    assert calls == []


def test_instantiating_tool_list_keeps_order_and_empty_list_behavior() -> None:
    assert ToolInstantiator.instantiate_tools(DictConfig({})) == []
    assert ToolInstantiator.instantiate_tools(DictConfig({"tools": []})) == []
    cfg = DictConfig({"tools": [_config(name="first"), _config("_shared_tool", name="second")]})
    assert [item.name for item in ToolInstantiator.instantiate_tools(cfg)] == ["first", "second"]
