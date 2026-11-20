"""deep_freeze turns nested arguments into hashable values that still behave like their originals."""

import copy
import json
import pickle
from typing import cast

import pytest

from agents_gym_core.utils.general_utils import FrozenDict, deep_freeze, hashable_lru_cache


def test_equal_mappings_freeze_to_equal_hashes_whatever_the_order() -> None:
    a = deep_freeze({"x": 1, "y": {"z": [1, 2]}})
    b = deep_freeze({"y": {"z": [1, 2]}, "x": 1})

    assert a == b and hash(a) == hash(b)
    assert a != deep_freeze({"x": 1, "y": {"z": [2, 1]}})


def test_a_frozen_mapping_is_still_a_read_only_dict() -> None:
    frozen = deep_freeze({"b": 2, "a": {"c": [3]}})

    assert isinstance(frozen, dict) and isinstance(frozen["a"], FrozenDict)
    assert dict(**frozen) == {"a": {"c": (3,)}, "b": 2}
    assert json.loads(json.dumps(frozen)) == {"a": {"c": [3]}, "b": 2}
    for change in (
        lambda: frozen.__setitem__("d", 4),
        lambda: frozen.__delitem__("b"),
        lambda: frozen.update(d=4),
        lambda: frozen.pop("b"),
        lambda: frozen.setdefault("d", 4),
        frozen.popitem,
        frozen.clear,
    ):
        with pytest.raises(TypeError):
            change()
    assert frozen == {"a": {"c": (3,)}, "b": 2}


def test_sequences_and_sets_freeze_to_tuples_and_unhashables_are_rejected() -> None:
    assert deep_freeze([1, [2, {3}]]) == (1, (2, (3,)))
    with pytest.raises(TypeError):
        deep_freeze(bytearray(b"x"))


def test_the_cache_reuses_results_for_equal_dict_arguments() -> None:
    calls: list[dict[str, int]] = []

    @hashable_lru_cache
    def build(config: dict[str, int], *, retries: int = 0) -> int:
        calls.append(config)
        return sum(config.values()) + retries

    assert build({"a": 1, "b": 2}) == 3
    assert build({"b": 2, "a": 1}) == 3
    assert build({"a": 1, "b": 2}, retries=1) == 4

    assert len(calls) == 2
    assert all(isinstance(c, dict) for c in calls)


def test_a_frozen_mapping_copies_and_merges_like_frozendict() -> None:
    frozen = cast(FrozenDict, deep_freeze({"a": {"b": 1}}))

    for clone in (copy.copy(frozen), copy.deepcopy(frozen), pickle.loads(pickle.dumps(frozen))):
        assert clone == frozen and isinstance(clone, FrozenDict) and hash(clone) == hash(frozen)
    merged = frozen | {"c": 2}
    assert isinstance(merged, FrozenDict) and merged == {"a": {"b": 1}, "c": 2}

    original = frozen
    frozen |= {"c": 2}
    assert frozen == merged and original == {"a": {"b": 1}}
