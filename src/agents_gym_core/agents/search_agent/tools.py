import os

from langchain_core.tools import tool

from agents_gym_core.common_tools.search_tool import SearchEngineTool, SearchEngineType


def _get_dynamic_response(query: str) -> str:
    """Generate a dynamic response based on the query content."""
    force_escalation_phrases = [
        "treaty of tordesillas",
        "crispr-cas9",
        "molecular level",
        "obscure historical",
        "complex technical",
        "defenestration of prague",
        "talens",
        "gene editing",
        "quantum error correction",
        "qubit systems",
        "specialized knowledge",
    ]

    if any(phrase in query.lower() for phrase in force_escalation_phrases):
        return "This topic requires specialized knowledge. You should now use the search tool to find additional information."

    if "historical" in query.lower() or "history" in query.lower() or "treaty" in query.lower():
        return "Historical topics often require specialized knowledge. Now proceed to search for specific details."

    if (
        "compare" in query.lower()
        or "versus" in query.lower()
        or " vs " in query.lower()
        or "contrast" in query.lower()
    ):
        return "For comparison queries, it's helpful to search for information about each item separately."

    if (
        "technical" in query.lower()
        or "code" in query.lower()
        or "programming" in query.lower()
        or "technology" in query.lower()
        or "molecular" in query.lower()
        or "gene" in query.lower()
        or "quantum" in query.lower()
        or "implementation" in query.lower()
    ):
        return "Technical topics require expert knowledge. Now use the search tool to find relevant information."

    if "latest" in query.lower() or "recent" in query.lower() or "new" in query.lower():
        return "Include the current year in your search query for more recent results."

    return "Try searching with more specific keywords or breaking your question into simpler parts."


@tool
async def escalation_tool(query: str) -> str:
    """Use this tool FIRST for specific complex topics: 1) Historical treaties and obscure events, 2) Technical explanations requiring deep expertise, 3) Complex technical comparisons. IMPORTANT: After using this tool once, proceed to use the search tool."""
    assign_to = "support@example.com"
    return f"Task created for {assign_to}. {_get_dynamic_response(query)}"


def _engine() -> SearchEngineType:
    """Select Tavily or Google when configured; default to DuckDuckGo."""
    if os.getenv("TAVILY_API_KEY"):
        return SearchEngineType.TAVILY
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
        return SearchEngineType.GOOGLE
    return SearchEngineType.DUCKDUCKGO


search_engine_tool = SearchEngineTool(
    name="search_engine",
    description="Use this tool to search the web for information. Input should be a search query string.",
    engine_type=_engine(),
    max_results=5,
)
