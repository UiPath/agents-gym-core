import json
import re
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from langchain_community.cache import SQLiteCache
from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.outputs import Generation

_current_run_number: ContextVar[int] = ContextVar("current_run_number", default=0)
_embedding_cache_dir: ContextVar[str | None] = ContextVar("embedding_cache_dir", default=None)


def get_embedding_cache_dir() -> str | None:
    """Return the embedding cache directory set by ``with_caching``, or *None*."""
    return _embedding_cache_dir.get(None)


@contextmanager
def caching_run_context(run_number: int) -> Iterator[None]:
    """Set the run number that partitions the LLM cache.

    Runs 0 and 1 use the plain cache key; run N > 1 appends ``:run_N`` to ``llm_string``, so
    repeated runs get distinct responses that replay identically from the cache.
    """
    token = _current_run_number.set(run_number)
    try:
        yield
    finally:
        _current_run_number.reset(token)


class _RunAwareSQLiteCache(SQLiteCache):
    """SQLiteCache that partitions entries by run number and counts hits and misses.

    Counts are taken in ``lookup``, which ``alookup`` delegates to, so async calls are
    included. Writes take a lock: concurrent writes corrupt the database ("database disk
    image is malformed").
    """

    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self.hits: int = 0
        self.misses: int = 0
        self._write_lock = threading.Lock()

    # Stripped before hashing: they differ between a hit and a miss (LangChain adds
    # ``total_cost: 0`` to ``usage_metadata`` on hits), so prior AI messages in a multi-turn
    # history would otherwise miss the cache.
    _VOLATILE_FIELDS: set[str] = {"usage_metadata", "response_metadata", "additional_kwargs"}

    @staticmethod
    def _keyed_llm_string(llm_string: str) -> str:
        run = _current_run_number.get()
        if run > 1:
            return f"{llm_string}:run_{run}"
        return llm_string

    @staticmethod
    def _strip_volatile(obj: Any) -> Any:
        """Recursively strip volatile fields from a deserialized prompt."""
        if isinstance(obj, dict):
            return {
                k: _RunAwareSQLiteCache._strip_volatile(v)
                for k, v in obj.items()
                if k not in _RunAwareSQLiteCache._VOLATILE_FIELDS
            }
        if isinstance(obj, list):
            return [_RunAwareSQLiteCache._strip_volatile(item) for item in obj]
        return obj

    @staticmethod
    def _normalize_prompt(prompt: str) -> str:
        """Strip volatile fields from a ``langchain_core.load.dumps`` prompt; non-JSON is kept."""
        try:
            data = json.loads(prompt)
            data = _RunAwareSQLiteCache._strip_volatile(data)
            return json.dumps(data, separators=(",", ":"), sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            return prompt

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_requests * 100 if self.total_requests else 0.0

    def log_stats(self) -> None:
        print(f"Cache stats: {self.hits}/{self.total_requests} hits ({self.hit_rate:.1f}%)")

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        result = super().lookup(self._normalize_prompt(prompt), self._keyed_llm_string(llm_string))
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        with self._write_lock:
            super().update(
                self._normalize_prompt(prompt), self._keyed_llm_string(llm_string), return_val
            )


class _MultiModelCache(BaseCache):
    """Cache that routes each request to a per-model SQLite database in ``cache_dir``.

    One ``.db`` per model avoids write contention between models. The first model whose name
    is a substring of ``llm_string`` is used; other requests (e.g. LLM-judge calls) go to
    ``_default.db``.
    """

    def __init__(self, cache_dir: str, model_names: list[str]):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._model_names = model_names
        self._caches: dict[str, _RunAwareSQLiteCache] = {}

        for model_name in model_names:
            if model_name in self._caches:
                continue
            # A name such as "Qwen/Qwen3.5-9B" must not name a subdirectory (or leave cache_dir).
            file_name = re.sub(r"[/\\\x00]", "_", model_name)
            db_path = self._cache_dir / f"{file_name}.db"
            self._caches[model_name] = self._create_cache(str(db_path))

        default_path = self._cache_dir / "_default.db"
        self._default_cache = self._create_cache(str(default_path))

    @staticmethod
    def _create_cache(db_path: str) -> _RunAwareSQLiteCache:
        try:
            return _RunAwareSQLiteCache(database_path=db_path)
        except Exception:
            print(f"Cache corrupted or unreadable, deleting and recreating: {db_path}")
            path = Path(db_path)
            if path.exists():
                path.unlink()
            return _RunAwareSQLiteCache(database_path=db_path)

    def _resolve_cache(self, llm_string: str) -> _RunAwareSQLiteCache:
        """Find the per-model cache whose name appears in ``llm_string``."""
        for model_name in self._model_names:
            if model_name in llm_string:
                return self._caches[model_name]
        return self._default_cache

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        return self._resolve_cache(llm_string).lookup(prompt, llm_string)

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        self._resolve_cache(llm_string).update(prompt, llm_string, return_val)

    def clear(self, **kwargs: Any) -> None:
        for cache in self._caches.values():
            cache.clear(**kwargs)
        self._default_cache.clear(**kwargs)

    @property
    def hits(self) -> int:
        return sum(c.hits for c in self._caches.values()) + self._default_cache.hits

    @property
    def misses(self) -> int:
        return sum(c.misses for c in self._caches.values()) + self._default_cache.misses

    @property
    def total_requests(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_requests * 100 if self.total_requests else 0.0

    def log_stats(self) -> None:
        print(
            f"Cache stats (aggregate): {self.hits}/{self.total_requests} hits ({self.hit_rate:.1f}%)"
        )
        for model_name, cache in self._caches.items():
            if cache.total_requests > 0:
                cache_label = f"  {model_name}"
                print(
                    f"{cache_label}: {cache.hits}/{cache.total_requests} hits ({cache.hit_rate:.1f}%)"
                )
        if self._default_cache.total_requests > 0:
            print(
                f"  _default: {self._default_cache.hits}/{self._default_cache.total_requests} "
                f"hits ({self._default_cache.hit_rate:.1f}%)"
            )


def setup_llm_cache(
    cache_db_name: str,
    model_names: list[str] | None = None,
) -> _RunAwareSQLiteCache | _MultiModelCache:
    """Install and return a SQLite LLM cache that counts hits and misses.

    With ``model_names``, there is one ``.db`` per model in ``<stem of cache_db_name>_caches/``
    instead of the single file ``cache_db_name``.
    """
    if model_names:
        cache_dir = str(Path(cache_db_name).stem) + "_caches"
        cache = _MultiModelCache(cache_dir=cache_dir, model_names=model_names)
        set_llm_cache(cache)
        print(f"LLM multi-model cache initialized: {cache_dir}/ ({len(model_names)} models)")
        return cache

    cache_path = Path(cache_db_name)
    try:
        cache = _RunAwareSQLiteCache(database_path=str(cache_path))
    except Exception:
        print(f"Cache corrupted or unreadable, deleting and recreating: {cache_path}")
        if cache_path.exists():
            cache_path.unlink()
        cache = _RunAwareSQLiteCache(database_path=str(cache_path))
    set_llm_cache(cache)
    print(f"LLM cache initialized: {cache_path}")
    return cache


@contextmanager
def with_caching(
    cache_db_name: str,
    embedding_cache_dir: str | None = None,
    model_names: list[str] | None = None,
) -> Iterator[_RunAwareSQLiteCache | _MultiModelCache]:
    """Install the LLM cache (see :func:`setup_llm_cache`) for the block; print stats on exit.

    The embedding cache directory, ``<stem of cache_db_name>_embeddings`` by default, is read
    through :func:`get_embedding_cache_dir`.
    """
    cache = setup_llm_cache(cache_db_name, model_names=model_names)
    emb_dir = (
        embedding_cache_dir
        if embedding_cache_dir is not None
        else str(Path(cache_db_name).stem) + "_embeddings"
    )
    token = _embedding_cache_dir.set(emb_dir)
    try:
        yield cache
    finally:
        _embedding_cache_dir.reset(token)
        set_llm_cache(None)
        cache.log_stats()
