from typing import Annotated

from langchain_core.tools import tool


@tool
async def add(a: Annotated[float, "First number"], b: Annotated[float, "Second number"]) -> float:
    """Add two numbers together."""
    return a + b


@tool
async def multiply(
    a: Annotated[float, "First number"], b: Annotated[float, "Second number"]
) -> float:
    """Multiply two numbers together."""
    return a * b
