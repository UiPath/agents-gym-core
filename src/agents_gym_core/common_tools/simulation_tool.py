"""Simulated tools for external dependencies (APIs, humans, services) during evaluation."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import logging
import os
from enum import StrEnum
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr, field_validator
from typing_extensions import override

from agents_gym_core.models.embeddings import get_embedding_model
from agents_gym_core.models.utils import get_model_name

logger = logging.getLogger(__name__)

INPUT_MOCKING_PROMPT_CONFIG_KEY = "input_mocking_prompt"

DEFAULT_SIMULATION_SYSTEM_PROMPT = (
    "You are simulating an external dependency during evaluation. "
    "Use the provided context and examples to generate a realistic "
    "response to the given request. Be concise and consistent with "
    "the examples."
)


class GenerationStrategy(StrEnum):
    """Strategy for generating responses when no exact match exists."""

    RANDOM_FEW_SHOT = "random_few_shot"
    RETRIEVAL_FEW_SHOT = "retrieval_few_shot"
    DIRECT_RETRIEVAL = "direct_retrieval"


def dict_to_cache_key(d: dict[str, Any]) -> str:
    """Serialize a dict to sorted JSON, with spaces removed from keys, for use as a cache key."""
    return json.dumps({k.replace(" ", ""): v for k, v in d.items()}, sort_keys=True)


def build_few_shot_messages(
    tool_name: str,
    examples: list[tuple[str, Any]],
) -> list[AIMessage | ToolMessage]:
    """Turn ``(args_json_str, result)`` examples into alternating AIMessage / ToolMessage pairs."""
    messages: list[AIMessage | ToolMessage] = []
    for i, (args_str, result) in enumerate(examples):
        args = json.loads(args_str)
        content = result if isinstance(result, str) else json.dumps(result)
        messages.append(
            AIMessage(
                content="",
                tool_calls=[ToolCall(name=tool_name, args=args, id=f"example_{i}")],
            )
        )
        messages.append(ToolMessage(content=content, tool_call_id=f"example_{i}"))
    return messages


class RetrievalHandler:
    """Example selection over a dict of recorded ``{args_json: result}`` pairs.

    - ``random_few_shot(k)``: the first *k* examples (deterministic).
    - ``retrieval_few_shot(query, k)``: the *k* most similar examples (FAISS).
    - ``direct_retrieval(query)``: the result of the most similar example.

    The FAISS index is built on first use, saved under ``vector_store_dir`` and
    cached at class level by directory and embedding model, so instances of a tool
    on the same data file share the index.
    """

    # Unlocked: safe under asyncio because the synchronous FAISS build never yields.
    # Threaded execution would need a lock around the build-and-save block.
    _store_cache: dict[str, Any] = {}

    def __init__(
        self,
        data: dict[str, Any],
        tool_name: str,
        vector_store_dir: str | None = None,
        embeddings: Any | None = None,
    ):
        self._data = data
        self._tool_name = tool_name
        self._vector_store_dir = vector_store_dir
        self._vector_store: Any | None = None
        self._embeddings: Any | None = embeddings

    @property
    def embeddings(self) -> Any:
        if self._embeddings is None:
            from agents_gym_core.models.embeddings import (
                OpenAIEmbeddingModels,
                get_embedding_model,
            )

            self._embeddings = get_embedding_model(
                OpenAIEmbeddingModels.text_embedding_ada_002.name,
                max_retries=5,
                chunk_size=50,
            )
        return self._embeddings

    @property
    def vector_store(self) -> Any | None:
        """The FAISS store: from the class cache, from disk, or built from the data."""
        if self._vector_store is not None:
            return self._vector_store
        if not self._data:
            return None

        from langchain_community.vectorstores import FAISS
        from langchain_core.documents import Document

        vsd = self._vector_store_dir
        emb_id = get_model_name(self.embeddings) if self.embeddings is not None else ""
        cache_key = f"{vsd}::{emb_id}" if vsd else ""
        if vsd:
            if cache_key in self._store_cache:
                self._vector_store = self._store_cache[cache_key]
                return self._vector_store

            if os.path.exists(vsd):
                try:
                    loaded = FAISS.load_local(
                        vsd, self.embeddings, allow_dangerous_deserialization=True
                    )
                    # Rebuild if the index was made with an embedding of another size.
                    test_vec = self.embeddings.embed_query("test")
                    if loaded.index.d != len(test_vec):
                        raise ValueError("Stale FAISS cache")
                    self._vector_store = loaded
                    self._store_cache[cache_key] = self._vector_store
                    return self._vector_store
                except Exception:
                    pass

        documents = [
            Document(page_content=args_str, metadata={"args": args_str, "result": result})
            for args_str, result in self._data.items()
        ]
        self._vector_store = FAISS.from_documents(documents=documents, embedding=self.embeddings)

        # Persist the optional vector-store cache.
        if vsd:
            try:
                os.makedirs(os.path.dirname(vsd), exist_ok=True)
                self._vector_store.save_local(vsd)
            except Exception:
                pass
            if cache_key:
                self._store_cache[cache_key] = self._vector_store

        return self._vector_store

    async def direct_retrieval(self, query: str) -> str | None:
        if self.vector_store is None:
            return None
        docs = await self.vector_store.asimilarity_search(query, k=1)
        return docs[0].metadata["result"] if docs else None

    async def retrieval_few_shot(self, query: str, k: int) -> list[tuple[str, str]]:
        if self.vector_store is None:
            return []
        docs = await self.vector_store.asimilarity_search(query, k=k)
        return [(doc.metadata["args"], doc.metadata["result"]) for doc in docs]

    def random_few_shot(self, k: int) -> list[tuple[str, str]]:
        return list(self._data.items())[:k]


class SimulationTool(BaseTool):
    """Base tool that simulates an external dependency during evaluation.

    Responses come from three sources:

    1. Recorded data: exact-match lookup in a ``{cache_key: response}`` dict
       returned by ``_load_data()``, which subclasses override for their own source.
    2. Input mocking prompt: per-datapoint context that the execution manager puts in
       ``RunnableConfig.configurable`` from the datapoint's
       ``inputMockingStrategy.prompt``.
    3. LLM generation: on a cache miss with ``generation_enabled``, an LLM answers
       from few-shot examples (random or retrieved) and the mocking prompt.

    Configuration in ``agent.yaml``::

        - _target_: agents_gym_core.common_tools.SimulationTool
          name: "contact_customer"
          description: "Contact the customer to request data or ask questions."
          simulation_system_prompt: "You are a UiPath customer..."
          generation_enabled: true
          generation_strategy: "retrieval_few_shot"
          few_shot_examples: 5

    Parameters:
        no_data_response: Returned on a cache miss when there is no mocking prompt.
            If empty, the tool generates from the few-shot examples alone (when
            ``generation_enabled``), which suits pure replay of recorded data.
    """

    name: str = "simulation_tool"
    description: str = ""

    simulation_system_prompt: str = DEFAULT_SIMULATION_SYSTEM_PROMPT
    no_data_response: str = (
        "No simulation data available. Proceed with the information currently available."
    )
    model_name: str = "gpt-5-mini-2025-08-07"
    generation_enabled: bool = False
    generation_strategy: str = GenerationStrategy.RANDOM_FEW_SHOT
    few_shot_examples: int = 5

    # JSON file of recorded pairs, either {cache_key: output, ...} or
    # [{"input": {...}, "output": ...}, ...]. Without it the tool relies on the
    # mocking prompt and the LLM.
    data_path: str | None = None

    # Key of the mocking prompt in RunnableConfig.configurable.
    config_key: str = INPUT_MOCKING_PROMPT_CONFIG_KEY

    # Exact-match replay of ``data``. Suits replay tools whose data holds recorded
    # outputs keyed by request kwargs. Subclasses whose data is few-shot material set
    # it to False, so a query that happens to match an entry never returns that
    # unrelated reply.
    exact_match_recorded_data: bool = True

    # Memoize generated replies within a run. The key combines the tool args with the
    # per-datapoint mocking prompt (``_generated_cache_key``), so a reply is reused
    # only in an identical simulation context. Off by default because most
    # simulators depend on per-datapoint context.
    cache_generated_replies: bool = False

    _data: dict[str, Any] | None = PrivateAttr(default=None)
    # Memo of generated (and direct-retrieval) replies, kept apart from ``data`` so
    # they never become replay or few-shot material.
    _generated_cache: dict[str, Any] = PrivateAttr(default_factory=dict)
    # Embeddings for few-shot retrieval, in order of precedence:
    # 1. embeddings_model_name from the YAML (the tool builds the model);
    # 2. embeddings injected by the loop from the experiment's --models spec;
    # 3. the RetrievalHandler default.
    embeddings_model_name: str | None = None
    embeddings: Any | None = None

    _retrieval_handler: RetrievalHandler | None = PrivateAttr(default=None)

    @field_validator("generation_strategy")
    @classmethod
    def validate_generation_strategy(cls, v: str) -> str:
        valid = [s.value for s in GenerationStrategy]
        if v not in valid:
            raise ValueError(f"Invalid generation_strategy '{v}'. Must be one of: {valid}")
        return v

    def _load_data(self) -> dict[str, Any]:
        """Return recorded ``{cache_key: response}`` pairs; override for other sources.

        The base implementation reads ``data_path``: a dict is returned as is, and a
        list of ``{"input": ..., "output": ...}`` items is keyed with ``dict_to_cache_key``.
        """
        if not self.data_path:
            return {}

        path = Path(self.data_path)
        if not path.exists():
            logger.warning("SimulationTool data_path does not exist: %s", self.data_path)
            return {}

        try:
            raw = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to load SimulationTool data from %s: %s", self.data_path, e)
            return {}

        if isinstance(raw, dict):
            return raw
        if isinstance(raw, list):
            return {
                dict_to_cache_key(item["input"]): item["output"]
                for item in raw
                if isinstance(item, dict) and "input" in item and "output" in item
            }
        logger.warning("Unexpected data format in %s: %s", self.data_path, type(raw))
        return {}

    @property
    def data(self) -> dict[str, Any]:
        if self._data is None:
            self._data = self._load_data()
        return self._data

    def _get_vector_store_dir(self) -> str | None:
        """Directory for the FAISS index, next to ``data_path``.

        The name includes the data file's name, so two data files in one folder get their own
        index, and the embedding model, so different embedders do not overwrite each other's
        index. Delete the folder after changing the data, as for any saved index.
        """
        if self.data_path:
            data_path = Path(self.data_path)
            emb_id = self.embeddings_model_name or (
                get_model_name(self.embeddings) if self.embeddings else ""
            )
            suffix = f"_{emb_id.replace('/', '--')}" if emb_id else ""
            name = f"vector_stores_{self.name}_{data_path.stem}{suffix}"
            return str(data_path.parent / name)
        return None

    @property
    def retrieval_handler(self) -> RetrievalHandler:
        if self._retrieval_handler is None:
            embeddings = self.embeddings
            if self.embeddings_model_name is not None:
                embeddings = get_embedding_model(self.embeddings_model_name)
            self._retrieval_handler = RetrievalHandler(
                data=self.data,
                tool_name=self.name,
                vector_store_dir=self._get_vector_store_dir(),
                embeddings=embeddings,
            )
        return self._retrieval_handler

    def _get_input_mocking_prompt(self) -> str:
        """Read the per-datapoint mocking prompt from the RunnableConfig."""
        config = var_child_runnable_config.get(None)
        if config is None:
            return ""
        return config.get("configurable", {}).get(self.config_key, "") or ""

    @staticmethod
    def _as_str(result: Any) -> str:
        """Return a tool result as a string (JSON-serializing non-str values)."""
        return result if isinstance(result, str) else json.dumps(result)

    def _generated_cache_key(self, kwargs: dict[str, Any], mocking_prompt: str) -> str:
        """Memo key for generated replies: the tool args plus the mocking prompt.

        The mocking prompt goes into the generation system prompt, so it is part of
        the key. Without one the key is the args key alone, and a reply is reused
        across datapoints with the same args.
        """
        args_key = dict_to_cache_key(kwargs)
        if not mocking_prompt:
            return args_key
        return dict_to_cache_key({"args": args_key, "mocking_prompt": mocking_prompt})

    def _get_llm(self) -> BaseChatModel:
        from agents_gym_core.models.chat import get_chat_model

        return get_chat_model(self.model_name)

    async def _select_few_shots(self, query: str) -> list[tuple[str, Any]]:
        """Select few-shot examples based on the generation strategy."""
        strategy = GenerationStrategy(self.generation_strategy)

        if strategy == GenerationStrategy.RETRIEVAL_FEW_SHOT:
            return await self.retrieval_handler.retrieval_few_shot(query, k=self.few_shot_examples)
        # RANDOM_FEW_SHOT, or DIRECT_RETRIEVAL after its lookup found nothing.
        return self.retrieval_handler.random_few_shot(k=self.few_shot_examples)

    async def _generate_response(
        self, query: str, kwargs: dict[str, Any], mocking_prompt: str
    ) -> str:
        """Generate a reply from few-shot examples and the mocking prompt.

        ``query`` is the serialized request (JSON of ``kwargs``, or the single argument
        as text); ``kwargs`` builds the memo key. DIRECT_RETRIEVAL returns the nearest
        recorded reply, when there is one, without calling the LLM.
        """
        if GenerationStrategy(self.generation_strategy) == GenerationStrategy.DIRECT_RETRIEVAL:
            result = await self.retrieval_handler.direct_retrieval(query)
            if result is not None:
                if self.cache_generated_replies:
                    self._generated_cache[self._generated_cache_key(kwargs, mocking_prompt)] = (
                        result
                    )
                return self._as_str(result)

        examples = await self._select_few_shots(query)

        system_parts = [self.simulation_system_prompt]
        if mocking_prompt:
            system_parts.append(f"\n## Simulation context\n{mocking_prompt}")

        few_shot_msgs = build_few_shot_messages(self.name, examples) if examples else []
        messages = [
            SystemMessage(content="\n".join(system_parts)),
            *few_shot_msgs,
            HumanMessage(content=query),
        ]

        llm = self._get_llm()
        response = await llm.ainvoke(messages)
        content = str(response.content)

        if self.cache_generated_replies:
            self._generated_cache[self._generated_cache_key(kwargs, mocking_prompt)] = content
        return content

    @override
    async def _arun(self, **kwargs: Any) -> str:
        hashed = dict_to_cache_key(kwargs)

        # 1. Exact-match replay of recorded data.
        if self.exact_match_recorded_data and hashed in self.data:
            return self._as_str(self.data[hashed])

        mocking_prompt = self._get_input_mocking_prompt()

        # 2. Memoized generated reply.
        if self.cache_generated_replies:
            gen_key = self._generated_cache_key(kwargs, mocking_prompt)
            if gen_key in self._generated_cache:
                return self._as_str(self._generated_cache[gen_key])

        has_mocking_prompt = bool(mocking_prompt.strip())

        query = json.dumps(kwargs) if len(kwargs) != 1 else str(next(iter(kwargs.values())))

        # 3. With a mocking prompt: generate, or return the prompt itself.
        if has_mocking_prompt:
            if self.generation_enabled:
                return await self._generate_response(query, kwargs, mocking_prompt)
            return mocking_prompt

        # 4. Without one: the placeholder, or generation from few-shot examples alone.
        if self.no_data_response:
            return self.no_data_response

        if self.generation_enabled:
            return await self._generate_response(query, kwargs, mocking_prompt)

        raise ValueError(
            f"Tool '{self.name}': no cached data for input {kwargs}, no input "
            f"mocking prompt available, and generation is disabled."
        )

    @override
    def _run(self, **kwargs: Any) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._arun(**kwargs))
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Run in a copy of this context so the RunnableConfig (and its mocking
                # prompt) reaches the worker thread.
                future = executor.submit(
                    contextvars.copy_context().run, asyncio.run, self._arun(**kwargs)
                )
                return future.result()
