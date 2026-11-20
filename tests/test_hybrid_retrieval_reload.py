"""Hybrid retrieval must preserve its indexed chunks when a local store is reused."""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from agents_gym_core.common_tools.vector_database.core_vector_database_tool import (
    CoreVectorDatabaseTool,
)


class _OfflineEmbeddings(Embeddings):
    model = "hybrid-reload-test"

    def __init__(self) -> None:
        self.document_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "needle" in text else [0.0, 1.0]


@pytest.mark.parametrize("reload_from_disk", [False, True], ids=["cached-store", "disk"])
def test_hybrid_retrieval_preserves_chunks_when_reusing_a_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reload_from_disk: bool
) -> None:
    monkeypatch.setattr(CoreVectorDatabaseTool, "_store_cache", {})
    monkeypatch.setattr(CoreVectorDatabaseTool, "_bm25_cache", {})
    embeddings = _OfflineEmbeddings()
    document = Document(
        page_content="alpha " * 50 + "\n\n" + "needle " * 50,
        metadata={"source": "corpus", "page": 0},
    )

    def make_tool() -> CoreVectorDatabaseTool:
        return CoreVectorDatabaseTool(
            persist_directory=str(tmp_path / "index"),
            embeddings=embeddings,
            documents=[document],
            splitting_method="recursive",
            splitting_chunk_size=150,
            splitting_chunk_overlap=0,
            search_strategy="hybrid",
        )

    fresh = make_tool()
    expected = [result.model_dump() for result in fresh.query("needle")]
    assert any(result["semantic_score"] > 0 for result in expected)
    assert all(len(result["content"]) < len(document.page_content) for result in expected)
    embedding_calls = embeddings.document_calls
    assert embedding_calls > 0

    CoreVectorDatabaseTool._bm25_cache.clear()
    if reload_from_disk:
        CoreVectorDatabaseTool._store_cache.clear()
    reloaded = make_tool()

    assert [result.model_dump() for result in reloaded.query("needle")] == expected
    assert embeddings.document_calls == embedding_calls
