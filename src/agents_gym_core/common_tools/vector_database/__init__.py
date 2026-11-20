"""Vector database tools for RAG operations."""

from agents_gym_core.common_tools.vector_database.core_vector_database_tool import (
    DEFAULT_EMBEDDINGS_MODEL,
    META_DESCRIPTION,
    META_DESCRIPTION_UPDATED,
    CoreVectorDatabaseTool,
    ImageStore,
    VectorDatabaseInput,
    VectorDatabaseResult,
)

VectorDatabaseTool = CoreVectorDatabaseTool

__all__ = [
    "CoreVectorDatabaseTool",
    "DEFAULT_EMBEDDINGS_MODEL",
    "ImageStore",
    "VectorDatabaseTool",
    "VectorDatabaseInput",
    "VectorDatabaseResult",
    "META_DESCRIPTION",
    "META_DESCRIPTION_UPDATED",
]
