"""Factory for episodic memory tools.

Tool descriptions are loaded from markdown files under ``prompts/`` and can be
overridden with inline text.
"""

from pathlib import Path
from typing import Literal

from langchain_core.tools import BaseTool
from langmem import create_manage_memory_tool, create_search_memory_tool
from pydantic import BaseModel

from agents_gym_core.common_tools.memory_tools.types import Observation

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text()


def create_memory_tools(
    *,
    namespace: tuple[str, ...] = ("memories",),
    schema: type[BaseModel] = Observation,
    manage_instructions: str | None = None,
    search_instructions: str | None = None,
    actions_permitted: tuple[Literal["create", "update", "delete"], ...] = (
        "create",
        "update",
        "delete",
    ),
) -> list[BaseTool]:
    """Create the langmem ``[manage_memory, search_memory]`` tool pair.

    The default descriptions tell the agent to record each tool result and to search
    its notes before repeating a call.

    Args:
        schema: Pydantic model for memory content.
        manage_instructions: Description of the manage tool. Defaults to
            ``prompts/manage_memory_instructions.md``.
        search_instructions: Description of the search tool. Defaults to
            ``prompts/search_memory_instructions.md``.
        actions_permitted: Actions the manage tool may perform.
    """
    manage_tool = create_manage_memory_tool(
        namespace=namespace,
        schema=schema,
        actions_permitted=actions_permitted,
    )
    search_tool = create_search_memory_tool(
        namespace=namespace,
    )

    # Replace the descriptions: langmem prepends generic text that dilutes the instructions.
    manage_tool.description = manage_instructions or _load_prompt("manage_memory_instructions.md")
    search_tool.description = search_instructions or _load_prompt("search_memory_instructions.md")

    return [manage_tool, search_tool]
