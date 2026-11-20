"""A failing API engine falls back to DuckDuckGo; a failing DuckDuckGo search is reported."""

import asyncio

import pytest

from agents_gym_core.common_tools.search_tool import (
    SearchEngineTool,
    SearchEngineType,
    SearchResult,
)


def _fake_search(failing: set[SearchEngineType]):
    async def search(self: SearchEngineTool, query: str) -> list[SearchResult]:
        if self.engine_type in failing:
            raise RuntimeError("Error 401: Unauthorized")
        return [SearchResult(title=query, link="https://example.com", snippet="s", position=0)]

    return search


def test_a_rejected_api_key_falls_back_to_duckduckgo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SearchEngineTool, "_search", _fake_search({SearchEngineType.TAVILY}))
    tool = SearchEngineTool(engine_type=SearchEngineType.TAVILY, api_key="stale")

    result = asyncio.run(tool._arun("query"))

    assert "https://example.com" in result
    assert tool.engine_type == SearchEngineType.DUCKDUCKGO


def test_a_failed_duckduckgo_search_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SearchEngineTool, "_search", _fake_search({SearchEngineType.DUCKDUCKGO}))
    tool = SearchEngineTool(engine_type=SearchEngineType.DUCKDUCKGO)

    assert asyncio.run(tool._arun("query")) == "Search failed: Error 401: Unauthorized"
