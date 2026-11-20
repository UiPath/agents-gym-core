import base64
import io
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import tiktoken
from langchain_chroma.vectorstores import Chroma
from langchain_classic.embeddings.cache import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
)
from langchain_core.documents import Document
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pandas import read_csv, read_excel
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from typing_extensions import override

from agents_gym_core.common_tools.vector_database.bounded_excel_loader import (
    ExcelReadLimits,
    _read_snapshot,
)
from agents_gym_core.common_tools.vector_database.uipath_ecs_vector_store import ECSVectorStore
from agents_gym_core.models.caching import get_embedding_cache_dir
from agents_gym_core.models.embeddings import OpenAIEmbeddingModels, get_embedding_model
from agents_gym_core.models.utils import get_model_name
from agents_gym_core.utils.gym_settings_core import gym_settings_core
from agents_gym_core.utils.trace_utils import warning

DEFAULT_EMBEDDINGS_MODEL: str = OpenAIEmbeddingModels.text_embedding_3_large.name

META_DESCRIPTION: str = """
This is a vector search tool that retrieves information from a knowledge base based on semantic similarity.
To use this tool effectively, generate concise search queries using specific keywords and phrases related to the topic.
Avoid full questions or conversational language. Focus on core concepts, entities, or terms most likely to match relevant information.

<knowledge-base-description>
{description}
</knowledge-base-description>

When formulating queries for this tool:
1. Generate brief, specific search terms aligned with the above description and current topic.
2. Think in terms of key search terms, not full questions.
3. Aim for terms likely to be semantically similar to the desired information.
4. Avoid full questions or conversational language
"""


META_DESCRIPTION_UPDATED = """
This context grounding index search tool retrieves information from a knowledge base using semantic similarity.

<description>
{description}
</description>

Query guidelines:
- Create concise queries with specific keywords related to your topic
- Focus on core concepts and terms, not questions or conversational language
- Choose terms with semantic similarity to desired information
"""


class VectorDatabaseInput(BaseModel):
    query: str


class VectorDatabaseResult(BaseModel):
    content: str
    source: str
    page: int
    similarity_score: float  # final combined score (used for ranking)
    semantic_score: float = 0.0  # vector-only cosine/L2 score
    keyword_score: float = 0.0  # BM25 score (normalized 0-1)
    image_base64: str | None = None  # base64-encoded page image (PNG)


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[int, float]]],
    k: int = 60,
) -> list[tuple[int, float]]:
    """Merge ranked lists with Reciprocal Rank Fusion.

    Each input list holds ``(doc_index, score)`` tuples sorted by descending score.
    Returns ``(doc_index, rrf_score)`` tuples sorted by descending RRF score.
    """
    scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (doc_idx, _score) in enumerate(ranked_list):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _build_bm25_index(documents: list[Document]) -> Any:
    """Build a BM25Okapi index from a list of LangChain Documents."""
    from rank_bm25 import BM25Okapi

    tokenized = [doc.page_content.lower().split() for doc in documents]
    return BM25Okapi(tokenized)


class VectorStoreType(StrEnum):
    CHROMA = "chroma"
    FAISS = "faiss"
    UIPATH_ECS = "uipath_ecs"


@runtime_checkable
class ImageStore(Protocol):
    """Protocol for providing page images to the vector database tool.

    Implementations map a (source_path, page_number) pair to a PIL Image.
    """

    def get_image(self, source: str, page: int) -> Any:
        """Return a PIL Image for the given source file and page, or None."""
        ...


class _LegacyExcelSnapshotLoader(UnstructuredExcelLoader):
    """Unstructured's BIFF partitioner over the sniffed snapshot; never reopens the caller's path."""

    def __init__(self, file_path: str, snapshot: bytes, modified: str, *, mode: str) -> None:
        self._snapshot = snapshot
        self._modified = modified
        super().__init__(file_path, mode=mode)

    def _get_elements(self) -> list[Any]:
        from unstructured.partition.xlsx import partition_xlsx

        return partition_xlsx(
            file=io.BytesIO(self._snapshot),
            metadata_filename=str(self.file_path),
            metadata_last_modified=self._modified,
        )


def _load_xlsx(
    file_path: str, single: bool, *, limits: ExcelReadLimits = ExcelReadLimits()
) -> list[Document]:
    """Validate once, then load immutable canonical bytes with pandas/openpyxl.

    Single mode combines sheets as CSV; otherwise rows become JSON documents
    with consecutive, zero-based page indices across all sheets.
    Genuine BIFF uses Unstructured's Office partitioner and its metadata.
    """
    snapshot, modified = _read_snapshot(
        file_path, limits, legacy_xls=Path(file_path).suffix.lower() == ".xls"
    )
    if not snapshot.startswith(b"PK\x03\x04"):
        return _LegacyExcelSnapshotLoader(
            file_path, snapshot, modified, mode="single" if single else "elements"
        ).load()
    sheets = read_excel(io.BytesIO(snapshot), sheet_name=None, engine="openpyxl")
    if single:
        text_content = "\n\n".join(df.to_csv(index=False) for df in sheets.values())
        return [Document(page_content=text_content, metadata={"source": file_path, "page": 0})]
    return [
        Document(
            page_content=json.dumps(row.to_dict(), indent=4, default=str),
            metadata={"source": file_path, "page": i},
        )
        for i, row in enumerate(row for df in sheets.values() for _, row in df.iterrows())
    ]


def _load_json(file_path: str) -> list[Document]:
    """Load a JSON list: a string item is a page of text, any other item a record (as JSON)."""
    with open(file_path, encoding="utf-8") as f:
        items: list[Any] = json.load(f)
    return [
        Document(
            page_content=item if isinstance(item, str) else json.dumps(item, indent=4),
            metadata={"source": file_path, "page": i},
        )
        for i, item in enumerate(items)
    ]


class CoreVectorDatabaseTool(BaseTool):
    """Retrieval tool over local documents (FAISS or Chroma) or a UiPath Context Grounding index.

    ``description`` is inserted into ``meta_description``. Local stores are built from
    ``documents`` or ``file_paths`` (resolved relative to ``agents_gym_core/agents/``) and
    saved under ``persist_directory``. ``search_strategy="hybrid"`` merges vector and BM25
    rankings with Reciprocal Rank Fusion.

    ``excel_limits`` configures OOXML resource budgets for new file ingestion,
    not reuse of existing stores. It is trusted configuration, not a tool input.

    For ``vector_store_type="uipath_ecs"``, pass ``ecs_index_name`` and ``folder_path``
    or ``folder_key``, and set ``UIPATH_URL`` (e.g. https://cloud.uipath.com/org/tenant)
    and ``UIPATH_ACCESS_TOKEN`` (a token with Context Grounding permissions)::

        CoreVectorDatabaseTool(
            name="my_vector_db",
            description="Search through my documents.",
            vector_store_type="uipath_ecs",
            ecs_index_name="My_Index",
            folder_path="Shared",
        )
    """

    model_config = ConfigDict(populate_by_name=True)

    # Caches keyed by persist_directory. Agents run as async tasks in one process, so
    # each store is loaded or built once and shared across tool instances.
    _store_cache: ClassVar[dict[str, VectorStore]] = {}
    # Parallel cache for BM25 indexes and their backing document lists.
    _bm25_cache: ClassVar[dict[str, tuple[Any, list[Document]]]] = {}

    name: str = "vector_database_tool"
    description: str = "Use this tool to search through documents and find relevant information. Input should be a search query string."
    meta_description: str = META_DESCRIPTION
    vector_store: Any = None
    vector_store_type: str = VectorStoreType.FAISS
    persist_directory: str | None = None
    file_paths: list[str] = []
    excel_limits: ExcelReadLimits = Field(default_factory=ExcelReadLimits)
    embeddings_model_name: str | None = Field(default=None, alias="model_name")
    embeddings: Any | None = None
    distance_metric: str = "l2"
    similarity_threshold: float | None = None
    max_items: int = 3
    include_metadata: bool = False
    splitting_method: str = "page"
    splitting_chunk_size: int = 1000
    splitting_chunk_overlap: int = 200
    embedding_chunk_size: int = 5  # Number of texts per embedding API call
    ecs_index_name: str | None = None
    folder_path: str | None = None
    folder_key: str | None = None
    documents: list[Document] | None = None  # Pre-loaded documents (bypass file loading)
    image_store: Any | None = None  # Optional ImageStore for page images
    output_mode: Literal["text", "image", "hybrid"] = "text"
    search_strategy: Literal["vector", "hybrid"] = "vector"
    page_window: int = 0  # Expand each hit with ±N adjacent pages from the same document
    _persist_dir_namespaced: bool = PrivateAttr(default=False)  # Guard against double-namespacing

    @field_validator("excel_limits", mode="before")
    @classmethod
    def _validate_excel_limits(cls, value: Any) -> ExcelReadLimits:
        if isinstance(value, ExcelReadLimits):
            value = asdict(value)
        if not isinstance(value, Mapping):
            raise ValueError("excel_limits must be a mapping or ExcelReadLimits instance")
        try:
            # Construct before Pydantic can coerce strings, floats or booleans to ints.
            return ExcelReadLimits(**value)
        except TypeError as exc:
            raise ValueError(f"Invalid excel_limits: {exc}") from exc

    def model_post_init(self, context: Any, /) -> None:
        self.description = self.meta_description.format(description=self.description)

        # uipath_ecs needs no file paths or embeddings; the store is created on first query.
        if self.vector_store_type == "uipath_ecs":
            self.args_schema = VectorDatabaseInput
            return super().model_post_init(context)

        # Resolve file paths relative to agents_gym_core/agents/
        agents_base_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "agents"
        )
        resolved_paths: list[str] = []
        for file_path in self.file_paths:
            full_path = os.path.join(agents_base_dir, file_path)
            if os.path.isdir(full_path):
                for entry in sorted(os.listdir(full_path)):
                    entry_path = os.path.join(full_path, entry)
                    if os.path.isfile(entry_path):
                        resolved_paths.append(entry_path)
            else:
                resolved_paths.append(full_path)
        self.file_paths = resolved_paths

        # Embeddings and the vector store are created on first query (_ensure_initialized),
        # so constructing the tool, e.g. during `gym uipath generate`, stays cheap.
        self.args_schema = VectorDatabaseInput

        try:
            return super().model_post_init(context)
        except Exception as e:
            warning(str(e))
            raise Exception(e)

    def _ensure_initialized(self) -> None:
        """Create the embeddings and the vector store on first use."""
        if self.vector_store is not None:
            return

        # ECS stores embed on the platform side, so no local embedding model is needed
        # (and none may be configured).
        if self.vector_store_type != "uipath_ecs" and self.embeddings is None:
            model_name: str = (
                self.embeddings_model_name
                if self.embeddings_model_name is not None
                else DEFAULT_EMBEDDINGS_MODEL
            )
            self.embeddings = get_embedding_model(
                model_name=model_name,
                max_retries=gym_settings_core.UIPATH_AGENTS_GYM_MAX_RETRIES,
                chunk_size=self.embedding_chunk_size,
            )

        # Namespace persist_directory by embedding model name to avoid dimension conflicts
        # when the same agent is run with different embedding models.
        if (
            self.persist_directory is not None
            and self.embeddings is not None
            and not self._persist_dir_namespaced
        ):
            emb_name = get_model_name(self.embeddings).replace("/", "--")
            self.persist_directory = os.path.join(self.persist_directory, emb_name)
            self._persist_dir_namespaced = True

        # Under with_caching(), cache document and query embeddings on disk: the embedding
        # API can return slightly different vectors for the same input, which would make
        # FAISS results vary across runs.
        emb_cache_dir = get_embedding_cache_dir()
        if (
            emb_cache_dir is not None
            and self.embeddings is not None
            and not isinstance(self.embeddings, CacheBackedEmbeddings)
        ):
            store = LocalFileStore(emb_cache_dir)
            self.embeddings = CacheBackedEmbeddings.from_bytes_store(
                underlying_embeddings=self.embeddings,
                document_embedding_cache=store,
                query_embedding_cache=True,
                namespace=get_model_name(self.embeddings),
                key_encoder="sha256",
            )

        self.vector_store = self.initialize_vector_store()

    @override
    async def _arun(self, query: str) -> str:
        """Async run method required by BaseTool"""
        results = await self.aquery(query)
        return self._format_results(results)  # type: ignore[return-value]

    @override
    def _run(self, query: str) -> str:
        """Synchronous run method required by BaseTool"""
        results = self.query(query)
        return self._format_results(results)  # type: ignore[return-value]

    def _format_results(self, results: list[VectorDatabaseResult]) -> str | list[dict[str, Any]]:
        """Format query results based on ``output_mode``.

        - ``"text"``: Returns a stringified list of page contents (default).
        - ``"image"``: Returns LangChain content blocks with page images.
          Falls back to text if no images are available.
        - ``"hybrid"``: Returns content blocks with both images and text.
          Falls back to text if no images are available.
        """
        if self.output_mode in ("image", "hybrid") and any(r.image_base64 for r in results):
            content_blocks: list[dict[str, Any]] = []
            for result in results:
                has_image = result.image_base64 is not None

                # In image-only mode, skip pages without images
                if self.output_mode == "image" and not has_image:
                    continue

                doc_id = Path(result.source).stem
                header = f"--- Page {result.page} ({doc_id}) ---"
                if self.include_metadata:
                    header += f" (relevance: {round(result.similarity_score, 2)})"
                content_blocks.append({"type": "text", "text": header})

                if has_image:
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{result.image_base64}"},
                        }
                    )

                if self.output_mode == "hybrid":
                    content_blocks.append({"type": "text", "text": result.content})

            if content_blocks:
                return content_blocks
            # Fall through to text mode if no content blocks produced

        # Text mode (default): a stringified list, for API compatibility.
        contents = [
            f"{result.content}; relevance score: {round(result.similarity_score, 2)}"
            if self.include_metadata
            else result.content
            for result in results
        ]
        return str(contents)

    def initialize_vector_store(
        self,
    ) -> VectorStore:
        """Initialize the vector store with document contents.

        Uses a class-level cache so that multiple tool instances pointing to
        the same persist_directory share a single in-memory store.
        """
        # uipath_ecs is a remote service and is not cached.
        if self.vector_store_type == "uipath_ecs":
            if self.ecs_index_name is None:
                raise ValueError(
                    "ecs_index_name must be provided if vector_store_type is 'uipath_ecs'"
                )
            if self.folder_path is None and self.folder_key is None:
                raise ValueError(
                    "At least one of folder_path or folder_key must be provided if vector_store_type is 'uipath_ecs'"
                )
            return ECSVectorStore(
                ecs_index_name=self.ecs_index_name,
                folder_path=self.folder_path,
                folder_key=self.folder_key,
            )

        if self.embeddings is None or self.persist_directory is None:
            raise ValueError(
                "embeddings and persist_directory must be provided if vector_store_type is 'chroma' or 'faiss'"
            )

        if not self.documents and not self.file_paths:
            raise ValueError(
                "Either documents or file_paths must be provided if vector_store_type is 'chroma' or 'faiss'"
            )

        if self.persist_directory in self._store_cache:
            warning(
                f"[CoreVectorDatabaseTool] Using in-memory cached store for '{self.persist_directory}'"
            )
            if self.search_strategy == "hybrid" and self.persist_directory not in self._bm25_cache:
                bm25_docs = self._get_bm25_documents()
                if not bm25_docs:
                    raise ValueError(
                        f"search_strategy='hybrid' requires documents for BM25 index, "
                        f"but no documents are available. Either pass documents= explicitly "
                        f"or use search_strategy='vector'. "
                        f"persist_directory='{self.persist_directory}'"
                    )
                self._bm25_cache[self.persist_directory] = (
                    _build_bm25_index(bm25_docs),
                    bm25_docs,
                )
            return self._store_cache[self.persist_directory]

        if os.path.exists(self.persist_directory):
            try:
                if self.vector_store_type == "chroma":
                    store = Chroma(
                        persist_directory=self.persist_directory,
                        embedding_function=self.embeddings,
                        collection_metadata={"hnsw:space": self.distance_metric},
                        relevance_score_fn=l2_relevance_score_fn
                        if self.distance_metric == "l2"
                        else cosine_relevance_score_fn,
                    )
                    # A first build that failed while embedding leaves an empty collection.
                    if hasattr(store, "get") and not store.get(limit=1, include=[])["ids"]:
                        raise ValueError("the persisted collection is empty")
                else:
                    store = FAISS.load_local(
                        self.persist_directory,
                        embeddings=self.embeddings,
                        allow_dangerous_deserialization=True,
                        relevance_score_fn=l2_relevance_score_fn
                        if self.distance_metric == "l2"
                        else faiss_cosine_relevance_score_fn,
                    )
            except Exception as e:
                warning(
                    f"[CoreVectorDatabaseTool] Failed to load existing vector store "
                    f"from '{self.persist_directory}': {e}. Rebuilding from documents."
                )
            else:
                self._store_cache[self.persist_directory] = store
                if self.search_strategy == "hybrid":
                    bm25_docs = self._get_bm25_documents()
                    if not bm25_docs:
                        raise ValueError(
                            f"search_strategy='hybrid' requires documents for BM25 index, "
                            f"but no documents are available. Either pass documents= explicitly "
                            f"or use search_strategy='vector'. "
                            f"persist_directory='{self.persist_directory}'"
                        )
                    self._bm25_cache[self.persist_directory] = (
                        _build_bm25_index(bm25_docs),
                        bm25_docs,
                    )
                return store

        if self.documents:
            warning(
                f"[CoreVectorDatabaseTool] Building vector store from {len(self.documents)} pre-loaded document(s)"
            )
        else:
            warning(
                f"[CoreVectorDatabaseTool] Building vector store from {len(self.file_paths)} file(s)"
            )
        loaded_documents = self._load_documents()

        if self.vector_store_type == "chroma":
            if os.path.exists(self.persist_directory):
                # from_documents adds to an existing collection, so a rebuild starts from an
                # empty one.
                Chroma(persist_directory=self.persist_directory).delete_collection()
            db = Chroma.from_documents(
                documents=loaded_documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_metadata={"hnsw:space": self.distance_metric},
                relevance_score_fn=l2_relevance_score_fn
                if self.distance_metric == "l2"
                else cosine_relevance_score_fn,
            )
        else:
            db = FAISS.from_documents(
                documents=loaded_documents,
                embedding=self.embeddings,
                distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE
                if self.distance_metric == "l2"
                else DistanceStrategy.COSINE,
                relevance_score_fn=l2_relevance_score_fn
                if self.distance_metric == "l2"
                else faiss_cosine_relevance_score_fn,
            )
            db.save_local(self.persist_directory)

        self._store_cache[self.persist_directory] = db

        if self.search_strategy == "hybrid":
            self._bm25_cache[self.persist_directory] = (
                _build_bm25_index(loaded_documents),
                loaded_documents,
            )

        return db

    def _load_documents(self) -> list[Document]:
        """Return ``documents``, or the documents loaded from ``file_paths``, split for indexing.

        Pages with empty content are dropped.
        """
        loaded_documents: list[Document] = list(self.documents) if self.documents else []

        for file_path in self.file_paths if not self.documents else []:
            if not os.path.exists(file_path):
                warning(f"[CoreVectorDatabaseTool] File not found, skipping: {file_path}")
                continue

            file_ext = os.path.splitext(file_path)[1].lower()

            if file_ext == ".pdf":
                mode = "single" if "single" in self.splitting_method else "page"
                loader = PyPDFLoader(file_path, mode=mode)
                loaded_documents.extend(loader.load())
            elif file_ext in [".xlsx", ".xls"]:
                loaded_documents.extend(
                    _load_xlsx(
                        file_path,
                        single="single" in self.splitting_method,
                        limits=self.excel_limits,
                    )
                )
            elif file_ext in [".doc", ".docx"]:
                mode = "single" if "single" in self.splitting_method else "elements"
                loader = UnstructuredWordDocumentLoader(file_path, mode=mode)
                loaded_documents.extend(loader.load())
            elif file_ext == ".json":
                loaded_documents.extend(_load_json(file_path))
            elif file_ext == ".csv":
                df = read_csv(file_path)
                for i, row in df.iterrows():
                    row_dict = row.to_dict()
                    text_content = json.dumps(row_dict, indent=4)
                    metadata = {"source": file_path, "page": i}
                    loaded_documents.append(Document(page_content=text_content, metadata=metadata))
            elif file_ext == ".txt":
                with open(file_path) as f:
                    for i, line in enumerate(f.readlines()):
                        metadata = {"source": file_path, "page": i}
                        loaded_documents.append(Document(page_content=line, metadata=metadata))

        if not loaded_documents:
            raise ValueError(
                f"No documents found. Checked file_paths: {self.file_paths}, documents: {self.documents is not None}"
            )

        # Filter out documents with empty page_content (e.g. from failed PDF text extraction)
        empty_count = sum(1 for doc in loaded_documents if not doc.page_content.strip())
        if empty_count > 0:
            warning(
                f"[CoreVectorDatabaseTool] {empty_count}/{len(loaded_documents)} documents "
                f"have empty page_content. Sources: "
                f"{set(doc.metadata.get('source', '?') for doc in loaded_documents if not doc.page_content.strip())}"
            )
            loaded_documents = [doc for doc in loaded_documents if doc.page_content.strip()]
            if not loaded_documents:
                raise ValueError(
                    "All loaded documents have empty page_content. "
                    "The PDF/document text extraction failed. "
                    f"Check file_paths: {self.file_paths}"
                )

        text_splitter = None
        if "recursive" in self.splitting_method:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.splitting_chunk_size,
                chunk_overlap=self.splitting_chunk_overlap,
            )
        elif "char" in self.splitting_method:
            # Map the embedding model name to its tiktoken encoding name; cl100k_base for a
            # model tiktoken does not know (or an embedding without a model name).
            try:
                encoding_name = tiktoken.encoding_for_model(
                    str(getattr(self.embeddings, "model", ""))
                ).name
            except KeyError:
                encoding_name = "cl100k_base"
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=encoding_name,
                chunk_size=self.splitting_chunk_size,
                chunk_overlap=self.splitting_chunk_overlap,
            )

        if text_splitter:
            loaded_documents = [
                Document(page_content=chunk, metadata=doc.metadata)
                for doc in loaded_documents
                for chunk in text_splitter.split_text(doc.page_content)
            ]

        return loaded_documents

    def _get_bm25_documents(self) -> list[Document]:
        """Rebuild BM25 from the same filtered, split documents as the vector store."""
        return self._load_documents()

    def _resolve_images(self, results: list[VectorDatabaseResult]) -> None:
        """Attach base64 page images to results using the image_store."""
        if self.image_store is None or self.output_mode == "text":
            return
        for result in results:
            img = self.image_store.get_image(result.source, result.page)
            if img is not None:
                result.image_base64 = _pil_to_base64(img)

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """BM25 keyword search; returns the top ``k`` ``(doc_index, score)`` pairs."""
        if self.persist_directory is None or self.persist_directory not in self._bm25_cache:
            raise ValueError("BM25 index not available — hybrid search requires local vector store")

        bm25_index, _documents = self._bm25_cache[self.persist_directory]
        tokens = query.lower().split()
        scores = bm25_index.get_scores(tokens)

        # Only a document that contains a query word is a hit (BM25 can score such a document 0 or
        # less); ranking the others would give corpus order rank-fusion credit.
        hits = [
            (index, score)
            for index, score in enumerate(scores)
            if any(token in bm25_index.doc_freqs[index] for token in tokens)
        ]
        return sorted(hits, key=lambda x: x[1], reverse=True)[:k]

    def _vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Vector search; returns ``(doc_index, score)`` pairs indexed into the BM25 documents.

        Sharing the BM25 indices lets RRF merge both rankings. Hits are matched by
        (source, page, first 100 characters); unmatched hits are dropped.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        docs = self.vector_store.similarity_search_with_relevance_scores(query, k=k)

        if self.persist_directory is None or self.persist_directory not in self._bm25_cache:
            raise ValueError("BM25 cache required for hybrid search")

        _, bm25_docs = self._bm25_cache[self.persist_directory]

        doc_lookup: dict[tuple[str, int, str], int] = {}
        for idx, doc in enumerate(bm25_docs):
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", 0),
                doc.page_content[:100],
            )
            doc_lookup[key] = idx

        results: list[tuple[int, float]] = []
        for doc, score in docs:
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", 0),
                doc.page_content[:100],
            )
            idx = doc_lookup.get(key)
            if idx is not None:
                results.append((idx, score))

        return results

    def _hybrid_query(self, query_text: str) -> list[VectorDatabaseResult]:
        """Hybrid search: vector and BM25 rankings merged with RRF."""
        # Over-fetch so that RRF has more candidates to merge.
        fetch_k = self.max_items * 3

        vector_results = self._vector_search(query_text, k=fetch_k)
        bm25_results = self._bm25_search(query_text, k=fetch_k)

        vector_scores = dict(vector_results)
        bm25_scores = dict(bm25_results)

        max_bm25 = max((s for _, s in bm25_results), default=1.0) or 1.0

        merged = reciprocal_rank_fusion([vector_results, bm25_results])

        _, documents = self._bm25_cache[self.persist_directory]  # type: ignore[index]

        results: list[VectorDatabaseResult] = []
        for doc_idx, rrf_score in merged[: self.max_items]:
            doc = documents[doc_idx]
            semantic = vector_scores.get(doc_idx, 0.0)
            keyword = bm25_scores.get(doc_idx, 0.0)
            results.append(
                VectorDatabaseResult(
                    content=doc.page_content,
                    source=doc.metadata.get("source", ""),
                    page=doc.metadata.get("page", 0),
                    similarity_score=rrf_score,
                    semantic_score=semantic,
                    keyword_score=keyword / max_bm25 if max_bm25 else 0.0,
                )
            )

        # similarity_threshold is not applied: RRF scores (~0.01-0.03) are on another scale
        # than vector similarity (0-1), so a vector-mode threshold would drop every result.
        return results

    def _vector_only_query(self, query_text: str) -> list[VectorDatabaseResult]:
        """Vector similarity search, filtered by ``similarity_threshold``."""
        docs = self.vector_store.similarity_search_with_relevance_scores(
            query_text, k=self.max_items
        )
        return [
            VectorDatabaseResult(
                content=doc.page_content,
                source=doc.metadata.get("source", ""),
                page=doc.metadata.get("page", 0),
                similarity_score=score,
                semantic_score=score,
            )
            for doc, score in docs
            if self.similarity_threshold is None or score >= self.similarity_threshold
        ]

    async def _vector_only_aquery(self, query_text: str) -> list[VectorDatabaseResult]:
        """Async version of ``_vector_only_query``."""
        docs = await self.vector_store.asimilarity_search_with_relevance_scores(
            query_text, k=self.max_items
        )
        return [
            VectorDatabaseResult(
                content=doc.page_content,
                source=doc.metadata.get("source", ""),
                page=doc.metadata.get("page", 0),
                similarity_score=score,
                semantic_score=score,
            )
            for doc, score in docs
            if self.similarity_threshold is None or score >= self.similarity_threshold
        ]

    def _expand_page_window(
        self, results: list[VectorDatabaseResult]
    ) -> list[VectorDatabaseResult]:
        """Add the pages within ±``page_window`` of each hit, from the same document.

        Needs pre-loaded ``documents``. Neighbouring pages get half the hit's score;
        pages are deduplicated and sorted by (source, page).
        """
        if self.page_window <= 0 or not self.documents:
            return results

        page_lookup: dict[tuple[str, int], Document] = {}
        source_max_page: dict[str, int] = defaultdict(int)
        for doc in self.documents:
            src = doc.metadata.get("source", "")
            pg = doc.metadata.get("page", 0)
            page_lookup[(src, pg)] = doc
            if pg > source_max_page[src]:
                source_max_page[src] = pg

        seen: set[tuple[str, int]] = set()
        expanded: list[VectorDatabaseResult] = []

        for result in results:
            src = result.source
            pg = result.page
            max_pg = source_max_page.get(src, pg)

            for offset in range(-self.page_window, self.page_window + 1):
                target_page = pg + offset
                if target_page < 0 or target_page > max_pg:
                    continue
                key = (src, target_page)
                if key in seen:
                    continue
                seen.add(key)

                if offset == 0:
                    expanded.append(result)
                else:
                    neighbor_doc = page_lookup.get(key)
                    if neighbor_doc is not None:
                        expanded.append(
                            VectorDatabaseResult(
                                content=neighbor_doc.page_content,
                                source=src,
                                page=target_page,
                                similarity_score=result.similarity_score * 0.5,
                                semantic_score=0.0,
                                keyword_score=0.0,
                            )
                        )

        expanded.sort(key=lambda r: (r.source, r.page))
        return expanded

    def query(self, query_text: str) -> list[VectorDatabaseResult]:
        """Search with ``search_strategy``, then apply the page window and attach images."""
        self._ensure_initialized()

        if self.search_strategy == "hybrid":
            results = self._hybrid_query(query_text)
        else:
            results = self._vector_only_query(query_text)

        if self.page_window > 0:
            results = self._expand_page_window(results)

        self._resolve_images(results)
        return results

    async def aquery(self, query_text: str) -> list[VectorDatabaseResult]:
        """Async version of ``query``."""
        self._ensure_initialized()

        if self.search_strategy == "hybrid":
            # BM25 is CPU-bound, so the hybrid path runs synchronously.
            results = self._hybrid_query(query_text)
        else:
            results = await self._vector_only_aquery(query_text)

        if self.page_window > 0:
            results = self._expand_page_window(results)

        self._resolve_images(results)
        return results


def l2_relevance_score_fn(distance: float) -> float:
    """Map a squared L2 distance between unit vectors (range [0, 4]) to a [0, 1] relevance."""
    return max(0.0, 1 - math.sqrt(distance) / 2)


def cosine_relevance_score_fn(distance: float) -> float:
    """Map a cosine distance between unit vectors (range [0, 2]) to a [0, 1] relevance."""
    return max(0.0, 1 - distance / 2)


def faiss_cosine_relevance_score_fn(distance: float) -> float:
    """Map FAISS's squared L2 distance between unit vectors (range [0, 4]) to a [0, 1] relevance.

    Half of it is the cosine distance, so scores match ``cosine_relevance_score_fn`` on Chroma.
    """
    return max(0.0, 1 - distance / 4)


def _pil_to_base64(pil_image: Any, max_dimension: int = 2000) -> str:
    """Encode a PIL image as base64 PNG, downscaled to at most ``max_dimension`` px a side."""
    width, height = pil_image.size
    if width > max_dimension or height > max_dimension:
        scale = max_dimension / max(width, height)
        pil_image = pil_image.resize(
            (int(width * scale), int(height * scale)),
        )
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
