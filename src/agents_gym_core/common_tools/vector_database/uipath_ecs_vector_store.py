"""Read-only vector store backed by UiPath Context Grounding.

Requires the environment variables:
- UIPATH_URL="https://cloud.uipath.com/{ORG_ID}/{TENANT_ID}"
- UIPATH_ACCESS_TOKEN={BEARER_TOKEN_WITH_CONTEXT_GROUNDING_PERMISSIONS}
"""

from typing import Any

from langchain_core.documents import Document
from typing_extensions import override
from uipath.platform import UiPath
from uipath_langchain.vectorstores.context_grounding_vectorstore import ContextGroundingVectorStore


class ECSVectorStore(ContextGroundingVectorStore):
    """``ContextGroundingVectorStore`` that also passes ``folder_key`` to searches.

    Example:
        .. code-block:: python

            from agents_gym_core.common_tools.vector_database.uipath_ecs_vector_store import (
                ECSVectorStore,
            )

            vectorstore = ECSVectorStore(ecs_index_name="my_index", folder_path="Shared")
            docs_with_scores = vectorstore.similarity_search_with_score(
                "search query", k=5
            )
    """

    def __init__(
        self,
        ecs_index_name: str,
        folder_path: str | None = None,
        folder_key: str | None = None,
        uipath_sdk: UiPath | None = None,
    ):
        """Open the Context Grounding index ``ecs_index_name``; ``uipath_sdk`` defaults to
        a new ``UiPath()`` client.
        """
        sdk = uipath_sdk or UiPath()
        super().__init__(
            index_name=ecs_index_name,
            uipath_sdk=sdk,
            folder_path=folder_path,
        )
        self.folder_key = folder_key

    @override
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Return the ``k`` documents most similar to ``query`` with their scores."""
        results = self.sdk.context_grounding.search(
            name=self.index_name,
            query=query,
            number_of_results=k,
            folder_path=self.folder_path,
            folder_key=self.folder_key,
        )
        return self._convert_results_to_documents(results)

    @override
    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> list[tuple[Document, float]]:
        """Async version of ``similarity_search_with_score``."""
        results: list[Any] = await self.sdk.context_grounding.search_async(
            name=self.index_name,
            query=query,
            number_of_results=k,
            folder_path=self.folder_path,
            folder_key=self.folder_key,
        )
        return self._convert_results_to_documents(results)
