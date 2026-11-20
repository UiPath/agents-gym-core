"""Tests for CoreVectorDatabaseTool — documents field, image support, format modes."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from pydantic import BaseModel, ValidationError

from agents_gym_core.common_tools.vector_database.bounded_excel_loader import (
    ExcelReadLimits,
    ExcelResourceLimitError,
    profile_workbook,
)
from agents_gym_core.common_tools.vector_database.core_vector_database_tool import (
    CoreVectorDatabaseTool,
    VectorDatabaseResult,
    _load_json,
    _load_xlsx,
    _pil_to_base64,
)
from agents_gym_core.hydra_agent.tool_instantiator import ToolInstantiator


@pytest.fixture
def workbook_agent_config(tmp_path: Path) -> DictConfig:
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        """agent:
  tools:
    - _target_: agents_gym_core.common_tools.vector_database.core_vector_database_tool.CoreVectorDatabaseTool
      name: workbook_search
      description: Search workbook exports.
      vector_store_type: chroma
      persist_directory: ${index_path}
      file_paths:
        - ${workbook_path}
      splitting_method: single
      excel_limits:
        max_rows: 60000
"""
    )
    config = OmegaConf.load(agent_yaml)
    assert isinstance(config, DictConfig)
    config.index_path = str(tmp_path / "index")
    config.workbook_path = str(tmp_path / "export.xlsx")
    return config


def _configured_workbook_tool(config: DictConfig) -> CoreVectorDatabaseTool:
    (tool,) = ToolInstantiator.instantiate_tools(config.agent)
    assert isinstance(tool, CoreVectorDatabaseTool)
    return tool


def test_excel_limits_defaults_and_partial_configuration() -> None:
    defaults = CoreVectorDatabaseTool().excel_limits
    assert defaults == ExcelReadLimits()
    assert len(asdict(defaults)) == 13
    for field in fields(ExcelReadLimits):
        configured = CoreVectorDatabaseTool.model_validate({"excel_limits": {field.name: 123}})
        assert configured.excel_limits == replace(defaults, **{field.name: 123})
    limits = ExcelReadLimits(max_rows=60_000)
    tool = CoreVectorDatabaseTool(excel_limits=limits)
    assert tool.excel_limits == limits
    serialized = tool.model_dump_json(include={"excel_limits"})
    assert CoreVectorDatabaseTool.model_validate_json(serialized).excel_limits == limits
    assert CoreVectorDatabaseTool().excel_limits == defaults


@pytest.mark.parametrize("field", [item.name for item in fields(ExcelReadLimits)])
@pytest.mark.parametrize(
    "value", [0, -1, True, False, 1.0, 1.5, "1", "60000", None, float("inf"), float("nan")]
)
def test_excel_limits_reject_invalid_values_without_coercion(field: str, value: Any) -> None:
    with pytest.raises(ValidationError, match="positive integers"):
        CoreVectorDatabaseTool.model_validate({"excel_limits": {field: value}})
    with pytest.raises(ValidationError, match="positive integers"):
        CoreVectorDatabaseTool.model_validate_json(json.dumps({"excel_limits": {field: value}}))


@pytest.mark.parametrize("value", [None, True, 60_000, [], "60000"])
def test_excel_limits_reject_non_mapping_configuration(value: Any) -> None:
    with pytest.raises(ValidationError, match="mapping or ExcelReadLimits"):
        CoreVectorDatabaseTool.model_validate({"excel_limits": value})


def test_excel_limits_reject_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="unexpected keyword argument 'max_row'"):
        CoreVectorDatabaseTool.model_validate(
            {"excel_limits": {"max_rows": 60_000, "max_row": 60_000}}
        )


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_row", 60_000),
        ("max_rows", True),
        ("max_rows", 0),
        ("max_rows", -1),
        ("max_rows", "60000"),
        ("max_rows", 1.0),
    ],
)
def test_yaml_workbook_limits_reject_invalid_configuration(
    workbook_agent_config: DictConfig, key: str, value: Any
) -> None:
    from hydra.errors import InstantiationException

    workbook_agent_config.agent.tools[0].excel_limits = {key: value}
    with pytest.raises(InstantiationException, match="excel_limits"):
        _configured_workbook_tool(workbook_agent_config)


def test_yaml_workbook_limits_are_not_llm_inputs(workbook_agent_config: DictConfig) -> None:
    tool = _configured_workbook_tool(workbook_agent_config)
    assert tool.excel_limits == ExcelReadLimits(max_rows=60_000)
    input_schema = tool.get_input_schema()
    assert issubclass(input_schema, BaseModel)
    assert set(input_schema.model_json_schema()["properties"]) == {"query"}
    tool_call_schema = tool.tool_call_schema
    assert isinstance(tool_call_schema, type)
    assert issubclass(tool_call_schema, BaseModel)
    assert set(tool_call_schema.model_json_schema()["properties"]) == {"query"}


@pytest.mark.parametrize("suffix", [".xlsx", ".xls"])
@pytest.mark.parametrize("single", [False, True])
def test_yaml_configured_60000_row_export_ingests_but_defaults_reject(
    workbook_agent_config: DictConfig, monkeypatch: pytest.MonkeyPatch, suffix: str, single: bool
) -> None:
    from openpyxl import Workbook

    from agents_gym_core.common_tools.vector_database import core_vector_database_tool as module

    path = Path(workbook_agent_config.workbook_path).with_suffix(suffix)
    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet()
    sheet.append(["value"])
    for value in range(59_999):
        sheet.append([value])
    workbook.save(path)
    workbook.close()
    workbook_agent_config.workbook_path = str(path)
    workbook_agent_config.agent.tools[0].splitting_method = "single" if single else "page"
    create = Mock(return_value=object())
    monkeypatch.setattr(module.Chroma, "from_documents", create)
    read_excel = module.read_excel
    parser = Mock(side_effect=AssertionError("Default row budget must reject before pandas"))
    monkeypatch.setattr(module, "read_excel", parser)

    default_config = OmegaConf.create(OmegaConf.to_container(workbook_agent_config, resolve=True))
    assert isinstance(default_config, DictConfig)
    del default_config.agent.tools[0].excel_limits
    default_tool = _configured_workbook_tool(default_config)
    default_tool.embeddings = object()
    with pytest.raises(ExcelResourceLimitError, match="row") as rejected:
        default_tool.initialize_vector_store()
    assert rejected.value.limit == 50_000
    assert rejected.value.observed > 50_000
    parser.assert_not_called()
    create.assert_not_called()
    assert workbook_agent_config.index_path not in default_tool._store_cache
    assert workbook_agent_config.index_path not in default_tool._bm25_cache

    tool = _configured_workbook_tool(workbook_agent_config)
    tool.embeddings = object()
    assert tool.excel_limits == replace(default_tool.excel_limits, max_rows=60_000)
    monkeypatch.setattr(module, "read_excel", read_excel)
    tool.initialize_vector_store()
    create.assert_called_once()
    documents = create.call_args.kwargs["documents"]
    if single:
        assert len(documents) == 1
        assert documents[0].page_content == "value\n" + "".join(
            f"{value}\n" for value in range(59_999)
        )
    else:
        assert len(documents) == 59_999
        assert [doc.page_content for doc in documents] == [
            json.dumps({"value": value}, indent=4) for value in range(59_999)
        ]
    assert [doc.metadata for doc in documents] == [
        {"source": str(path), "page": page} for page in range(len(documents))
    ]


def test_custom_profile_preserves_default_cli_budgets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from openpyxl import Workbook

    from agents_gym_core.common_tools.vector_database.profile_excel import main

    path = tmp_path / "sparse.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet["A60000"] = "synthetic value"
    workbook.save(path)
    workbook.close()
    limits = ExcelReadLimits(max_rows=60_000)
    report = profile_workbook(path, limits=limits)
    assert report["accepted"] is True
    assert report["limits"] == asdict(limits)
    assert "synthetic value" not in json.dumps(report)
    with pytest.raises(ExcelResourceLimitError) as rejected:
        profile_workbook(path)
    assert rejected.value.limit == 50_000
    assert main([str(path)]) == 1
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["accepted"] is False
    assert cli_report["limits"] == asdict(ExcelReadLimits())
    assert cli_report["error"]["limit"] == 50_000


@pytest.mark.parametrize("cached", [False, True], ids=["persisted", "in-memory"])
def test_workbook_budgets_do_not_change_existing_store_reuse(
    workbook_agent_config: DictConfig, monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    from agents_gym_core.common_tools.vector_database import core_vector_database_tool as module

    tool = _configured_workbook_tool(workbook_agent_config)
    tool.embeddings = object()
    persist_directory = tool.persist_directory
    assert persist_directory is not None
    store = object()
    if cached:
        monkeypatch.setattr(
            module.CoreVectorDatabaseTool,
            "_store_cache",
            {
                persist_directory: store,
            },
        )
    else:
        Path(persist_directory).mkdir()
        monkeypatch.setattr(module, "Chroma", Mock(return_value=store))
    loader = Mock(side_effect=AssertionError("Store reuse must not ingest workbooks"))
    monkeypatch.setattr(module, "_load_xlsx", loader)
    assert tool.initialize_vector_store() is store
    loader.assert_not_called()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(width: int = 100, height: int = 100) -> Image.Image:
    """Create a small solid-color test image."""
    return Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))


def _image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _sample_results(with_images: bool = False) -> list[VectorDatabaseResult]:
    b64 = _image_to_base64(_make_image()) if with_images else None
    return [
        VectorDatabaseResult(
            content="First page content about climate change.",
            source="/data/corpus/report_A.md",
            page=3,
            similarity_score=0.92,
            image_base64=b64,
        ),
        VectorDatabaseResult(
            content="Second page content about renewable energy.",
            source="/data/corpus/report_B.md",
            page=7,
            similarity_score=0.85,
            image_base64=b64,
        ),
    ]


def _sample_results_mixed() -> list[VectorDatabaseResult]:
    """First result has image, second does not."""
    b64 = _image_to_base64(_make_image())
    return [
        VectorDatabaseResult(
            content="First page content about climate change.",
            source="/data/corpus/report_A.md",
            page=3,
            similarity_score=0.92,
            image_base64=b64,
        ),
        VectorDatabaseResult(
            content="Second page content about renewable energy.",
            source="/data/corpus/report_B.md",
            page=7,
            similarity_score=0.85,
            image_base64=None,
        ),
    ]


class FakeImageStore:
    """ImageStore that returns a test image for any source/page."""

    def __init__(self) -> None:
        self._img = _make_image()

    def get_image(self, source: str, page: int) -> Image.Image:
        return self._img


class EmptyImageStore:
    """ImageStore that never has images."""

    def get_image(self, source: str, page: int) -> None:
        return None


def _make_tool(output_mode: str = "text", include_metadata: bool = False) -> CoreVectorDatabaseTool:
    """Create a tool instance with just the fields _format_results needs, bypassing heavy init."""
    return CoreVectorDatabaseTool.model_construct(
        output_mode=output_mode,
        include_metadata=include_metadata,
        # Required BaseTool fields
        name="test_tool",
        description="test",
    )


# ---------------------------------------------------------------------------
# Tests: _format_results — text mode (default)
# ---------------------------------------------------------------------------


class TestFormatResultsTextMode:
    def test_text_content_present(self) -> None:
        output = _make_tool()._format_results(_sample_results())
        assert "climate change" in output
        assert "renewable energy" in output

    def test_include_metadata_appends_score(self) -> None:
        output = _make_tool(include_metadata=True)._format_results(_sample_results())
        assert "relevance score: 0.92" in output
        assert "relevance score: 0.85" in output

    def test_text_mode_ignores_images(self) -> None:
        output = _make_tool()._format_results(_sample_results(with_images=True))
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# Tests: _format_results — image mode
# ---------------------------------------------------------------------------


class TestFormatResultsImageMode:
    def test_image_mode_has_header_and_image(self) -> None:
        output = _make_tool("image")._format_results(_sample_results(with_images=True))
        assert isinstance(output, list)
        assert len(output) == 4  # 2 × (header + image)
        types = [b["type"] for b in output]
        assert types == ["text", "image_url", "text", "image_url"]

    def test_image_mode_skips_pages_without_images(self) -> None:
        output = _make_tool("image")._format_results(_sample_results_mixed())
        assert isinstance(output, list)
        assert len(output) == 2  # only report_A
        assert "report_A" in output[0]["text"]

    def test_image_mode_header_includes_metadata(self) -> None:
        output = _make_tool("image", include_metadata=True)._format_results(
            _sample_results(with_images=True)[:1]
        )
        assert isinstance(output, list)
        header = output[0]["text"]
        assert "Page 3" in header
        assert "report_A" in header
        assert "relevance: 0.92" in header

    def test_no_images_falls_back_to_text(self) -> None:
        output = _make_tool("image")._format_results(_sample_results(with_images=False))
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# Tests: _format_results — hybrid mode
# ---------------------------------------------------------------------------


class TestFormatResultsHybridMode:
    def test_hybrid_has_header_image_and_text(self) -> None:
        output = _make_tool("hybrid")._format_results(_sample_results(with_images=True))
        assert isinstance(output, list)
        assert len(output) == 6  # 2 × (header + image + text)
        types = [b["type"] for b in output]
        assert types == ["text", "image_url", "text", "text", "image_url", "text"]

    def test_hybrid_includes_page_content(self) -> None:
        output = _make_tool("hybrid")._format_results(_sample_results(with_images=True)[:1])
        assert isinstance(output, list)
        assert len(output) == 3
        assert output[2]["type"] == "text"
        assert "climate change" in output[2]["text"]

    def test_hybrid_no_images_falls_back_to_text(self) -> None:
        output = _make_tool("hybrid")._format_results(_sample_results(with_images=False))
        assert isinstance(output, str)


# ---------------------------------------------------------------------------
# Tests: _pil_to_base64
# ---------------------------------------------------------------------------


class TestPilToBase64:
    def test_resizes_large_image(self) -> None:
        result = _pil_to_base64(_make_image(4000, 3000), max_dimension=2000)
        decoded = base64.b64decode(result)
        assert len(decoded) > 0


# ---------------------------------------------------------------------------
# Tests: _resolve_images
# ---------------------------------------------------------------------------


class TestResolveImages:
    def test_attaches_images_from_store(self) -> None:
        tool = _make_tool("image")
        tool.image_store = FakeImageStore()
        results = [
            VectorDatabaseResult(content="test", source="doc1.md", page=1, similarity_score=0.9)
        ]
        tool._resolve_images(results)
        assert results[0].image_base64 is not None

    def test_skips_when_text_mode(self) -> None:
        tool = _make_tool("text")
        tool.image_store = FakeImageStore()
        results = [
            VectorDatabaseResult(content="test", source="doc1.md", page=1, similarity_score=0.9)
        ]
        tool._resolve_images(results)
        assert results[0].image_base64 is None

    def test_skips_when_no_image_store(self) -> None:
        tool = _make_tool("image")
        tool.image_store = None
        results = [
            VectorDatabaseResult(content="test", source="doc1.md", page=1, similarity_score=0.9)
        ]
        tool._resolve_images(results)
        assert results[0].image_base64 is None

    def test_handles_missing_images_gracefully(self) -> None:
        tool = _make_tool("image")
        tool.image_store = EmptyImageStore()
        results = [
            VectorDatabaseResult(content="test", source="doc1.md", page=1, similarity_score=0.9)
        ]
        tool._resolve_images(results)
        assert results[0].image_base64 is None


# ---------------------------------------------------------------------------
# .xlsx loading (pandas + openpyxl, no xlrd)
# ---------------------------------------------------------------------------


def _write_xlsx(path: str) -> None:
    import pandas as pd

    pd.DataFrame(
        {"Recommended Action": ["Lubricate motor", "Replace belt"], "Assigned Group": ["A", "B"]}
    ).to_excel(path, index=False)


class TestLoadXlsx:
    def test_single_gives_one_csv_document(self, tmp_path: Path) -> None:
        path = str(tmp_path / "incidents.xlsx")
        _write_xlsx(path)
        (doc,) = _load_xlsx(path, single=True)
        assert doc.page_content.splitlines()[0] == "Recommended Action,Assigned Group"
        assert "Replace belt,B" in doc.page_content
        assert doc.metadata == {"source": path, "page": 0}

    def test_elements_gives_one_json_document_per_row(self, tmp_path: Path) -> None:
        path = str(tmp_path / "incidents.xlsx")
        _write_xlsx(path)
        docs = _load_xlsx(path, single=False)
        assert [d.metadata["page"] for d in docs] == [0, 1]
        assert '"Recommended Action": "Replace belt"' in docs[1].page_content


# ---------------------------------------------------------------------------
# .json loading
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_string_items_are_pages_of_text(self, tmp_path: Path) -> None:
        path = tmp_path / "chapter.json"
        path.write_text('["Heading 0101\\nLive horses", "Heading 0102"]', encoding="utf-8")
        docs = _load_json(str(path))
        assert [d.page_content for d in docs] == ["Heading 0101\nLive horses", "Heading 0102"]
        assert docs[1].metadata == {"source": str(path), "page": 1}

    def test_other_items_are_records_as_json(self, tmp_path: Path) -> None:
        path = tmp_path / "products.json"
        path.write_text('[{"Name": "Desk lamp", "Price": 25}]', encoding="utf-8")
        (doc,) = _load_json(str(path))
        assert doc.page_content == '{\n    "Name": "Desk lamp",\n    "Price": 25\n}'
