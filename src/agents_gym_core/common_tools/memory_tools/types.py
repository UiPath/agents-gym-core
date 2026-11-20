"""Memory schema types for episodic memory tools."""

from pydantic import BaseModel, Field


class Observation(BaseModel):
    """Structured observation of a single agent tool call.

    Each observation records exactly one tool invocation with its key arguments,
    making it easy for a termination checker to detect genuine repetition.
    """

    tool_name: str = Field(
        ...,
        description="Exact name of the tool that was called (e.g. 'sql_db_query', 'search_web').",
    )
    tool_arguments: str = Field(
        ...,
        description="The most important argument(s) passed to the tool — e.g. the SQL query text, the search term, the file path. Keep it short but specific enough to distinguish this call from others.",
    )
    result_summary: str = Field(
        ...,
        description="Brief factual summary of what the tool returned (1 sentence). Include key data values when present.",
    )
