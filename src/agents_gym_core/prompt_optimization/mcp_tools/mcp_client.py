"""A persistent MCP session that yields the coach-light tools as LangChain tools (ReAPO).

The claude engine connects to the same server through the config ``main.py`` writes.
"""

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from langchain_core.tools import BaseTool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from uipath_langchain.agent.tools.mcp import load_mcp_tools

from agents_gym_core.project import ProjectDirectoryProvider
from agents_gym_core.prompt_optimization.mcp_tools.helpers import directory_provider

_MCP_SERVER_PATH = Path(__file__).resolve().parents[1] / "mcp_server.py"


def _build_connection() -> StdioServerParameters:
    """Build the stdio connection config for the coach-light MCP server.

    Forwards optimizer workspace settings so the server picks up the same
    output path, message limit, and immutable-round boundary.
    """
    # Start with the full current environment so the MCP server subprocess
    # inherits everything (LLMGW_*, UIPATH_*, PATH, HOME, etc.)
    env = {k: v for k, v in os.environ.items() if isinstance(v, str)}
    # Override/set optimizer-specific vars
    env["COACH_OUTPUT_DIR"] = os.environ.get("COACH_OUTPUT_DIR", "")
    env["COACH_MAX_CHARS"] = os.environ.get("COACH_MAX_CHARS", "20000")
    env["COACH_IMMUTABLE_ROUNDS_BEFORE"] = os.environ.get("COACH_IMMUTABLE_ROUNDS_BEFORE", "0")

    arguments = [str(_MCP_SERVER_PATH)]
    provider = directory_provider()
    if isinstance(provider, ProjectDirectoryProvider):
        arguments += ["--project", str(provider.get_project_root())]
    return StdioServerParameters(
        command=sys.executable,
        args=arguments,
        env=env,
    )


@asynccontextmanager
async def coach_light_session() -> AsyncIterator[list[BaseTool]]:
    """Async context manager that yields a persistent list of LangChain tools
    from the coach-light MCP server.

    Opens one subprocess for the entire duration.

    Usage:
        async with coach_light_session() as tools:
            agent = create_react_agent(model=llm, tools=tools)
            ...
    """
    async with stdio_client(_build_connection()) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            yield await load_mcp_tools(session)
