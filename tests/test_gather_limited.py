"""gather_limited runs at most N calls at once, reads its input lazily, and keeps or drops failures."""

import asyncio
from collections.abc import Coroutine, Generator
from typing import Any, TypeVar

import pytest

from agents_gym_core.utils.general_utils import gather_limited

T = TypeVar("T")


def _run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def test_never_runs_more_than_max_concurrency_calls() -> None:
    active = peak = 0

    async def work(i: int) -> int:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return i

    results = _run(gather_limited(3, [(i,) for i in range(20)], work, show_progress=False))

    assert sorted(results) == list(range(20))
    assert peak == 3


def test_a_generator_is_read_only_as_slots_free_up() -> None:
    pulled: list[int] = []
    started: list[tuple[int, int]] = []  # (item, items pulled when it started)

    def items() -> Generator[tuple[int], None, None]:
        for i in range(6):
            pulled.append(i)
            yield (i,)

    async def work(i: int) -> int:
        started.append((i, len(pulled)))
        await asyncio.sleep(0.01)
        return i

    _run(gather_limited(2, items(), work, show_progress=False))

    assert [n for _, n in started[:2]] == [2, 2]
    assert all(n <= i + 2 for i, n in started)


def test_ordered_results_follow_the_input_and_unordered_follow_completion() -> None:
    async def work(i: int) -> int:
        await asyncio.sleep(0.01 * (3 - i))
        return i

    items = [(i,) for i in range(3)]

    assert _run(gather_limited(3, items, work, ordered=True, show_progress=False)) == [0, 1, 2]
    assert _run(gather_limited(3, items, work, ordered=False, show_progress=False)) == [2, 1, 0]


def test_fault_tolerant_mode_drops_failed_calls() -> None:
    async def work(i: int) -> int:
        if i % 2:
            raise ValueError(f"bad {i}")
        return i

    results = _run(gather_limited(2, [(i,) for i in range(6)], work, show_progress=False))

    assert sorted(results) == [0, 2, 4]


def test_otherwise_the_first_failure_propagates_and_running_calls_are_cancelled() -> None:
    cancelled: list[int] = []

    async def work(i: int) -> int:
        if i == 0:
            await asyncio.sleep(0.01)
            raise ValueError("boom")
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.append(i)
            raise
        return i

    with pytest.raises(ValueError, match="boom"):
        _run(
            gather_limited(
                3, [(i,) for i in range(5)], work, fault_tolerant=False, show_progress=False
            )
        )

    assert sorted(cancelled) == [1, 2]


def test_the_progress_bar_path_returns_the_same_results() -> None:
    async def work(i: int) -> int:
        return i * i

    results = _run(gather_limited(4, [(i,) for i in range(5)], work, ordered=True))

    assert results == [0, 1, 4, 9, 16]


def test_empty_input_returns_no_results() -> None:
    async def work(i: int) -> int:
        return i

    assert _run(gather_limited(4, [], work)) == []
