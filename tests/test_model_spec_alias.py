"""Tests for ModelSpec.alias as the registry key and result identity."""

from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from agents_gym_core.loops.types import LoopDetails
from agents_gym_core.models.model_spec import ModelRegistryMeta, ModelSpec
from agents_gym_core.models.utils import get_model_name

_FAKE_CONFIG = {"responses": ["ok"]}


def _fake_spec(model_name: str, alias: str = "") -> ModelSpec:
    return ModelSpec(
        model_name,
        FakeListChatModel,
        default_config=dict(_FAKE_CONFIG),
        max_concurrency=None,
        alias=alias,
    )


class _Reg(metaclass=ModelRegistryMeta):
    """Dummy registry — no network, no gateway."""

    plain = _fake_spec("vendor-model")
    tuned = _fake_spec("vendor-model", alias="vendor-model-high")


def test_variants_sharing_a_model_name_stay_distinct() -> None:
    assert _Reg.plain.model_name == _Reg.tuned.model_name
    assert "vendor-model" in _Reg
    assert "vendor-model-high" in _Reg
    assert _Reg["vendor-model"] is _Reg.plain
    assert _Reg["vendor-model-high"] is _Reg.tuned
    assert hash(_Reg.plain) != hash(_Reg.tuned)


def test_register_indexes_by_alias() -> None:
    class _Local(metaclass=ModelRegistryMeta):
        pass

    spec = _Local.register(_fake_spec("vendor-model", alias="vendor-model-low"))
    assert _Local._index == {"vendor-model-low": spec}
    assert _Local["vendor-model-low"] is spec


def test_replacing_a_class_attribute_drops_the_old_alias_key() -> None:
    class _Local(metaclass=ModelRegistryMeta):
        entry = _fake_spec("vendor-model", alias="first")

    _Local.entry = _fake_spec("vendor-model", alias="second")

    assert "first" not in _Local
    assert _Local._index == {"second": _Local.entry}


def test_loop_key_separates_variants_of_one_deployment() -> None:
    plain = _Reg.plain.get_chat_model()
    tuned = _Reg.tuned.get_chat_model()

    plain_key = LoopDetails(loop_name="basic", models={"model": get_model_name(plain)}).loop_key
    tuned_key = LoopDetails(loop_name="basic", models={"model": get_model_name(tuned)}).loop_key

    assert plain_key != tuned_key
    assert tuned_key.endswith("model:vendor-model-high")
