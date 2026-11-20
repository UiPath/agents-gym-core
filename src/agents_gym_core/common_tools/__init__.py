"""Common tools for agents. Exports resolve lazily (PEP 562)."""

from typing import TYPE_CHECKING

_EXPORTS: dict[str, str] = {
    "EscalationTool": "agents_gym_core.common_tools.escalation_tool",
    "Observation": "agents_gym_core.common_tools.memory_tools",
    "create_memory_tools": "agents_gym_core.common_tools.memory_tools",
    "PlatformAnalyzeAttachmentsTool": "agents_gym_core.common_tools.platform_analyze_attachments_tool",
    "SearchEngineTool": "agents_gym_core.common_tools.search_tool",
    "SearchEngineType": "agents_gym_core.common_tools.search_tool",
    "GenerationStrategy": "agents_gym_core.common_tools.simulation_tool",
    "RetrievalHandler": "agents_gym_core.common_tools.simulation_tool",
    "SimulationTool": "agents_gym_core.common_tools.simulation_tool",
    "data_preview_tool": "agents_gym_core.common_tools.spreadsheet_tools",
    "describe_tool": "agents_gym_core.common_tools.spreadsheet_tools",
    "filter_csv_tool": "agents_gym_core.common_tools.spreadsheet_tools",
    "list_columns_tool": "agents_gym_core.common_tools.spreadsheet_tools",
    "CoreVectorDatabaseTool": "agents_gym_core.common_tools.vector_database",
    "VectorDatabaseTool": "agents_gym_core.common_tools.vector_database",
}

if TYPE_CHECKING:
    from agents_gym_core.common_tools.escalation_tool import EscalationTool as EscalationTool
    from agents_gym_core.common_tools.memory_tools import Observation as Observation
    from agents_gym_core.common_tools.memory_tools import (
        create_memory_tools as create_memory_tools,
    )
    from agents_gym_core.common_tools.platform_analyze_attachments_tool import (
        PlatformAnalyzeAttachmentsTool as PlatformAnalyzeAttachmentsTool,
    )
    from agents_gym_core.common_tools.search_tool import SearchEngineTool as SearchEngineTool
    from agents_gym_core.common_tools.search_tool import SearchEngineType as SearchEngineType
    from agents_gym_core.common_tools.simulation_tool import (
        GenerationStrategy as GenerationStrategy,
    )
    from agents_gym_core.common_tools.simulation_tool import (
        RetrievalHandler as RetrievalHandler,
    )
    from agents_gym_core.common_tools.simulation_tool import SimulationTool as SimulationTool
    from agents_gym_core.common_tools.spreadsheet_tools import (
        data_preview_tool as data_preview_tool,
    )
    from agents_gym_core.common_tools.spreadsheet_tools import describe_tool as describe_tool
    from agents_gym_core.common_tools.spreadsheet_tools import (
        filter_csv_tool as filter_csv_tool,
    )
    from agents_gym_core.common_tools.spreadsheet_tools import (
        list_columns_tool as list_columns_tool,
    )
    from agents_gym_core.common_tools.vector_database import (
        CoreVectorDatabaseTool as CoreVectorDatabaseTool,
    )
    from agents_gym_core.common_tools.vector_database import (
        VectorDatabaseTool as VectorDatabaseTool,
    )

__all__ = [
    "CoreVectorDatabaseTool",
    "EscalationTool",
    "GenerationStrategy",
    "Observation",
    "PlatformAnalyzeAttachmentsTool",
    "RetrievalHandler",
    "SearchEngineTool",
    "SearchEngineType",
    "SimulationTool",
    "VectorDatabaseTool",
    "create_memory_tools",
    "data_preview_tool",
    "describe_tool",
    "filter_csv_tool",
    "list_columns_tool",
]


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(target), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
