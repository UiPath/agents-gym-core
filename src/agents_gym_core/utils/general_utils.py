import ast
import asyncio
import re
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Hashable,
    Iterable,
    Mapping,
    Sequence,
)
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Literal, NoReturn, TypeVar, overload

from tqdm.std import tqdm
from uipath.eval.helpers import EvalHelpers
from uipath.eval.models.evaluation_set import EvaluationSet

from agents_gym_core.utils.trace_utils import error, warning

T = TypeVar("T")
R = TypeVar("R")


class _Failed:
    """Sentinel returned by _wrapper when a fault-tolerant task fails."""


def get_evaluation_set(path: Path) -> EvaluationSet:
    eval_set, _ = EvalHelpers.load_eval_set(str(path))
    return eval_set


async def _bounded_map(
    items: Iterable[T], func: Callable[[T], Awaitable[R]], limit: int, ordered: bool
) -> AsyncIterator[R]:
    """Yield ``func(item)`` for every item, running at most ``limit`` calls at a time.

    An item is taken from ``items`` only when a call slot is free, so a generator is read
    lazily. With ``ordered`` the results come in input order, otherwise as the calls finish.
    If a call raises, the calls still running are cancelled and the exception propagates.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")
    iterator = iter(items)
    running: dict[asyncio.Future[R], int] = {}
    finished: dict[int, R] = {}
    next_index = submitted = 0
    exhausted = False
    try:
        while True:
            while not exhausted and len(running) < limit:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                running[asyncio.ensure_future(func(item))] = submitted
                submitted += 1
            if not running:
                return
            done, _ = await asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)
            for task in sorted(done, key=running.__getitem__):
                index = running.pop(task)
                result = task.result()
                if ordered:
                    finished[index] = result
                else:
                    yield result
            while next_index in finished:
                yield finished.pop(next_index)
                next_index += 1
    finally:
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)


async def gather_limited(
    max_concurrency: int,
    items: Generator[T, None, None] | Sequence[T],
    func: Callable[..., Awaitable[R]],
    *,
    total: int | None = None,
    initial: int = 0,
    show_progress: bool = True,
    ordered: bool = False,
    tqdm_description: str | None = None,
    fault_tolerant: bool = True,
) -> list[R]:
    """Await ``func(*item)`` for each item, at most ``max_concurrency`` at a time.

    ``total`` and ``initial`` set the tqdm bar; ``total`` is inferred for a sequence, and
    without it no bar is shown. ``ordered`` returns results in input order. With
    ``fault_tolerant``, a failing task is logged and left out of the results; otherwise the
    first exception propagates and in-flight results are lost.
    """

    _sentinel = _Failed()

    async def _wrapper(item: Any) -> R | _Failed:
        nonlocal _task_idx
        idx = _task_idx
        _task_idx += 1
        if fault_tolerant:
            try:
                return await func(*item)
            except Exception as exc:
                warning(f"gather_limited: task {idx} failed: {exc}", task_index=idx)
                _errors.append((idx, exc))
                return _sentinel
        return await func(*item)

    _task_idx = 0
    _errors: list[tuple[int, Exception]] = []

    if isinstance(items, Sequence) and total is None:
        total = len(items)

    results: list[R | _Failed] = []

    mapped = _bounded_map(items, _wrapper, max_concurrency, ordered)

    if show_progress and total is not None:
        pbar = tqdm(total=total, initial=initial, smoothing=0.02, desc=tqdm_description)
        try:
            async for x in mapped:
                results.append(x)
                pbar.update(1)
        finally:
            pbar.close()
    else:
        results = [x async for x in mapped]

    succeeded = [x for x in results if not isinstance(x, _Failed)]
    if _errors:
        total_tasks = len(results)
        error(
            f"gather_limited: {len(_errors)} of {total_tasks} tasks failed and were excluded from results",
            failed=len(_errors),
            total=total_tasks,
            succeeded=len(succeeded),
            failed_indices=[idx for idx, _ in _errors],
            errors=[str(exc) for _, exc in _errors],
        )

    return succeeded


def _literal_values(inner: str) -> tuple[Any, ...]:
    """The values of ``Literal[<inner>]``, e.g. ``"'a', \"don't\""`` gives ``('a', "don't")``."""
    try:
        values = ast.literal_eval(f"({inner},)")
    except (ValueError, SyntaxError):
        return tuple(re.findall(r"['\"]([^'\"]+)['\"]", inner))
    return tuple(values) if isinstance(values, tuple) else ()


def get_python_type(type_str: str) -> Any:
    """Convert a type string to a Python type; unknown names become ``str``.

    Supports "str", "int", "float", "bool", "list", "dict", unions such as
    "str | int | None", "Literal['a', 'b']", "list[str]" and "list[Literal['a', 'b']]"
    (``List`` is also accepted).
    """
    list_simple_match = re.fullmatch(r"[Ll]ist\[(\w+)\]", type_str.strip())
    if list_simple_match:
        inner_type_str = list_simple_match.group(1)
        inner_type = {"str": str, "int": int, "float": float, "bool": bool, "dict": dict}.get(
            inner_type_str, str
        )
        return list[inner_type]

    list_literal_match = re.fullmatch(r"[Ll]ist\[Literal\[(.+)\]\]", type_str.strip())
    if list_literal_match:
        values = _literal_values(list_literal_match.group(1))
        if values:
            return list[Literal[values]]

    literal_match = re.fullmatch(r"Literal\[(.+)\]", type_str.strip())
    if literal_match:
        values = _literal_values(literal_match.group(1))
        if values:
            return Literal[values]

    type_mapping: dict[str, type] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
    }

    if "|" in type_str:
        parts = [p.strip() for p in type_str.split("|")]
        has_none = "None" in parts
        type_parts = [get_python_type(p) for p in parts if p != "None"]

        if not type_parts:
            return str | None  # type: ignore[return-value]

        result_type = type_parts[0]
        for t in type_parts[1:]:
            result_type = result_type | t  # type: ignore[assignment]

        if has_none:
            result_type = result_type | None  # type: ignore[assignment]

        return result_type  # type: ignore[return-value]

    return type_mapping.get(type_str, str)


class FrozenDict(dict[Any, Any]):
    """A read-only, hashable dict: the frozen form of a mapping in ``deep_freeze``.

    It stays a ``dict``, so it unpacks with ``**``, serialises to JSON and passes
    ``isinstance(x, dict)``; every method that would change it raises ``TypeError``.
    """

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __reduce__(self) -> tuple[type["FrozenDict"], tuple[dict[Any, Any]]]:
        # Rebuild from a plain dict, so copy, deepcopy and pickle never call __setitem__.
        return (FrozenDict, (dict(self),))

    def __or__(self, other: Any) -> "FrozenDict":  # type: ignore[override]
        return FrozenDict({**self, **other})

    # Like frozendict: ``d |= other`` rebinds d to a new FrozenDict instead of changing it.
    __ior__ = __or__  # type: ignore[assignment]

    def _read_only(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise TypeError("FrozenDict is read-only")

    __setitem__ = __delitem__ = _read_only
    clear = pop = popitem = setdefault = update = _read_only  # type: ignore[assignment]


def deep_freeze(x: Any) -> Hashable:
    if isinstance(x, Mapping):
        return FrozenDict(sorted((k, deep_freeze(v)) for k, v in x.items()))
    elif isinstance(x, (list, tuple, set, frozenset)):
        return tuple(deep_freeze(i) for i in x)
    elif isinstance(x, Hashable):
        return x
    else:
        raise TypeError(f"Unfreezable type: {type(x)}")


F = TypeVar("F", bound=Callable[..., Any])


@overload
def hashable_lru_cache(func: F) -> F: ...
@overload
def hashable_lru_cache(*, maxsize: int | None = 128, typed: bool = False) -> Callable[[F], F]: ...


def hashable_lru_cache(
    func: F | None = None,
    *,
    maxsize: int | None = 128,
    typed: bool = False,
) -> F | Callable[[F], F]:
    """``functools.lru_cache`` that deep-freezes dict, list and set arguments into cache keys.

    Usable bare (``@hashable_lru_cache``) or with arguments (``@hashable_lru_cache(maxsize=256)``).
    """

    def decorator(inner_func: F) -> F:
        @lru_cache(maxsize=maxsize, typed=typed)
        def cached_call(frozen_args: Any, frozen_kwargs: Any) -> Any:
            return inner_func(*frozen_args, **frozen_kwargs)

        @wraps(inner_func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return cached_call(deep_freeze(args), deep_freeze(kwargs))

        wrapper.cache_info = cached_call.cache_info  # type: ignore[attr-defined]
        wrapper.cache_clear = cached_call.cache_clear  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator
