import asyncio
import json
import logging
import os
from enum import StrEnum
from typing import Any, Literal, cast

from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.tools import BaseTool
from langchain_google_community import GoogleSearchAPIWrapper
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import override

logger = logging.getLogger(__name__)

META_DESCRIPTION: str = """
This is a search engine tool that retrieves information from the web based on search queries.
To use this tool effectively, create clear, specific search queries focused on the information you need.

When formulating queries for this tool:
1. Be specific and concise in your queries
2. Use relevant keywords that are likely to appear in the desired information
3. Avoid ambiguous terms or overly broad questions
4. Consider including specific terms that narrow the search scope
"""


class SearchResult(BaseModel):
    """Model for search results returned by the search tool."""

    title: str
    link: str
    snippet: str
    position: int


class SearchEngineType(StrEnum):
    """Enum for supported search engines."""

    GOOGLE = "google"
    DUCKDUCKGO = "duckduckgo"
    TAVILY = "tavily"


class SearchEngineTool(BaseTool):
    """Web search via Google, DuckDuckGo or Tavily.

    Google and Tavily fall back to DuckDuckGo when no API key is passed or found in the
    environment.
    """

    name: str = "search_engine_tool"
    description: str
    engine_type: SearchEngineType = SearchEngineType.DUCKDUCKGO
    api_key: str | None = None
    max_results: int = 5

    # Engine-specific settings
    google_params: dict[str, Any] = Field(
        default_factory=lambda: {"google_cse_id": os.getenv("GOOGLE_CSE_ID")}
    )
    duckduckgo_params: dict[str, Any] = Field(default_factory=dict)
    tavily_params: dict[str, Any] = Field(default_factory=dict)
    results_separator: str = "\n"
    output_format: Literal["string", "markdown", "json"] = "string"
    keys_to_include: set[str] | None = None
    _search_wrapper: Any = None

    def __init__(
        self,
        description: str = "Use this tool to search the web for information. Input should be a search query string.",
        engine_type: SearchEngineType = SearchEngineType.DUCKDUCKGO,
        api_key: str | None = None,
        **kwargs: Any,
    ):
        if not api_key:
            if (
                engine_type == SearchEngineType.GOOGLE
                and os.getenv("GOOGLE_API_KEY")
                and os.getenv("GOOGLE_CSE_ID")
            ):
                api_key = os.getenv("GOOGLE_API_KEY")
            elif engine_type == SearchEngineType.TAVILY and os.getenv("TAVILY_API_KEY"):
                api_key = os.getenv("TAVILY_API_KEY")
            elif engine_type != SearchEngineType.DUCKDUCKGO:
                logger.warning("No search API key or CSE id provided, defaulting to DuckDuckGo")
                engine_type = SearchEngineType.DUCKDUCKGO
        super().__init__(
            description=META_DESCRIPTION + description,
            api_key=api_key,
            engine_type=engine_type,
            **kwargs,
        )

    @override
    async def _arun(self, query: str) -> str:
        try:
            return self.format_results(await self._search(query))
        except Exception as e:
            if self.engine_type == SearchEngineType.DUCKDUCKGO:
                # Tell the model the search failed (DuckDuckGo rate-limits under load) instead
                # of returning an empty result it would read as "nothing found".
                logger.error(f"Error during search: {str(e)}")
                return f"Search failed: {e}"
            # A rejected or exhausted API key must not fail every search: use DuckDuckGo.
            logger.warning(f"{self.engine_type} search failed ({e}); falling back to DuckDuckGo")
            self.engine_type = SearchEngineType.DUCKDUCKGO
            self._search_wrapper = None
            return await self._arun(query)

    async def _search(self, query: str) -> list[SearchResult]:
        search_wrapper = self._get_search_wrapper()
        if self.engine_type == SearchEngineType.GOOGLE:
            raw = await asyncio.to_thread(
                cast(GoogleSearchAPIWrapper, search_wrapper).results,
                query,
                num_results=self.max_results,
            )
            results = [
                SearchResult(
                    title=r.get("title", ""),
                    link=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                    position=i,
                )
                for i, r in enumerate(raw)
            ]
        elif self.engine_type == SearchEngineType.DUCKDUCKGO:
            raw = await asyncio.to_thread(
                cast(DuckDuckGoSearchAPIWrapper, search_wrapper).results,
                query,
                max_results=self.max_results,
            )
            results = [
                SearchResult(
                    title=r.get("title", ""),
                    link=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                    position=i,
                )
                for i, r in enumerate(raw)
            ]
        elif self.engine_type == SearchEngineType.TAVILY:
            raw = await cast(TavilySearchAPIWrapper, search_wrapper).results_async(
                query, max_results=self.max_results
            )
            results = [
                SearchResult(
                    title=r.get("title", ""),
                    link=r.get("url", ""),
                    snippet=r.get("content", ""),
                    position=i,
                )
                for i, r in enumerate(raw)
            ]
        else:
            raise ValueError(f"Unsupported search engine type: {self.engine_type}")
        return results

    @override
    def _run(self, query: str) -> str:
        return asyncio.run(self._arun(query))

    def format_results(self, results: list[SearchResult]) -> str:
        """Render results as ``output_format`` (string, markdown or JSON)."""
        if self.output_format == "string":
            return self.results_separator.join(
                [
                    ", ".join(
                        [
                            f"{k}: {v}"
                            for k, v in d.model_dump().items()
                            if not self.keys_to_include or k in self.keys_to_include
                        ]
                    )
                    for d in results
                ]
            )
        elif self.output_format == "markdown":
            formatted_results = []
            for result in results:
                parts = []
                if not self.keys_to_include or "title" in self.keys_to_include:
                    parts.append(f"## {result.title}")
                if not self.keys_to_include or "link" in self.keys_to_include:
                    parts.append(f"[Link]({result.link})")
                if not self.keys_to_include or "snippet" in self.keys_to_include:
                    parts.append(f"\n{result.snippet}")
                formatted_results.append("\n".join(parts))
            return self.results_separator.join(formatted_results)
        elif self.output_format == "json":
            return json.dumps([d.model_dump(include=self.keys_to_include) for d in results])
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def _get_search_wrapper(
        self,
    ) -> DuckDuckGoSearchAPIWrapper | GoogleSearchAPIWrapper | TavilySearchAPIWrapper:
        """Return the cached search wrapper for ``engine_type``, creating it if needed."""
        if self._search_wrapper is not None:
            return self._search_wrapper

        if self.engine_type == SearchEngineType.GOOGLE:
            self._search_wrapper = GoogleSearchAPIWrapper(
                google_api_key=self.api_key, **self.google_params
            )
        elif self.engine_type == SearchEngineType.DUCKDUCKGO:
            self._search_wrapper = DuckDuckGoSearchAPIWrapper(**self.duckduckgo_params)
        elif self.engine_type == SearchEngineType.TAVILY:
            assert self.api_key is not None
            self._search_wrapper = TavilySearchAPIWrapper(
                tavily_api_key=SecretStr(self.api_key),
                **self.tavily_params,
            )
        else:
            raise ValueError(f"Unsupported search engine type: {self.engine_type}")

        return self._search_wrapper

    def set_engine(self, engine_type: SearchEngineType, api_key: str | None = None) -> None:
        """Switch the search engine; the wrapper is rebuilt on the next search."""
        self.engine_type = engine_type
        if api_key:
            self.api_key = api_key
        self._search_wrapper = None
