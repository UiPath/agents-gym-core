"""Workbook metadata profiles and CLI privacy/exit-status checks."""

import io
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
from test_bounded_excel_loader import (
    _auxiliary_workbook,
    _claim_size,
    _long_body_workbook,
    _rename_part,
    _repeat_shared_reference,
    _rewrite,
)
from test_bounded_excel_loader import (
    workbook_path as workbook_path,
)

from agents_gym_core.common_tools.vector_database.bounded_excel_loader import (
    ExcelReadLimits,
    _read_snapshot,
    profile_workbook,
)
from agents_gym_core.common_tools.vector_database.profile_excel import main as profile_main


@pytest.mark.parametrize("suffix", [".xlsx", ".xls"])
def test_profile_reports_validated_metadata_without_office_parsing_or_cell_content(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    import openpyxl
    import pandas

    workbook_path = workbook_path.rename(workbook_path.with_suffix(suffix))
    original = workbook_path.read_bytes()
    with ZipFile(io.BytesIO(original)) as archive:
        sizes = {member.filename: member.file_size for member in archive.infolist()}
    partition = Mock(side_effect=AssertionError("Profile must not invoke an Office parser"))
    monkeypatch.setattr(openpyxl, "load_workbook", partition)
    monkeypatch.setattr(pandas, "read_excel", partition)
    profile = profile_workbook(workbook_path)
    assert profile["accepted"] is True
    assert profile["rejection_policy"] == "abort_index_build"
    assert profile["limits"] == asdict(ExcelReadLimits())
    assert profile["archive"]["input_bytes"] == len(original)
    assert profile["archive"]["members"] == len(sizes)
    assert profile["archive"]["expanded_bytes"] == sum(sizes.values())
    assert profile["archive"]["canonical_bytes"] == len(
        _read_snapshot(str(workbook_path), ExcelReadLimits())[0]
    )
    assert {part["part"]: part["expanded_bytes"] for part in profile["archive"]["parts"]} == sizes
    assert profile["package"]["sheets"] == 3
    sheet = next(
        part for part in profile["xml_parts"] if part["part"] == "xl/worksheets/sheet1.xml"
    )
    assert (sheet["rows"], sheet["columns"], sheet["grid_cells"]) == (4, 2, 8)
    assert (sheet["declared_rows"], sheet["declared_columns"], sheet["declared_dimensions"]) == (
        4,
        2,
        1,
    )
    assert (sheet["observed_rows"], sheet["observed_columns"], sheet["sheet_materializations"]) == (
        4,
        2,
        1,
    )
    assert sheet["depth"] > 0 and sheet["nodes"] > 0
    assert profile["package"]["grid_cells"] == sum(
        part["grid_cells"] * max(1, part["sheet_materializations"]) for part in profile["xml_parts"]
    )
    assert profile["package"]["projected_cell_text_bytes"] == sum(
        part["projected_cell_text_bytes"] * max(1, part["sheet_materializations"])
        for part in profile["xml_parts"]
    )
    encoded = json.dumps(profile)
    for text in ("WO-1001", "Work order", "Work orders", "Assignments", "Example operator"):
        assert text not in encoded
    assert workbook_path.read_bytes() == original
    partition.assert_not_called()


@pytest.mark.parametrize("alias", [False, True])
def test_profile_distinguishes_declared_and_observed_bounds_including_opc_aliases(
    workbook_path: Path, alias: bool
) -> None:
    part = "xl/worksheets/sheet1.xml"
    _rewrite(
        workbook_path,
        part,
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b'<dimension ref="A1:Z100"/><sheetData><row><c><v>1</v></c></row>'
        b"</sheetData></worksheet>",
    )
    if alias:
        _rename_part(workbook_path, part, "xl/worksheets/data.bin")
        part = "xl/worksheets/data.bin"
    profile = profile_workbook(workbook_path)
    sheet = next(summary for summary in profile["xml_parts"] if summary["part"] == part)
    assert (sheet["declared_rows"], sheet["declared_columns"]) == (100, 26)
    assert (sheet["observed_rows"], sheet["observed_columns"]) == (1, 1)
    assert sheet["grid_cells"] == 2600


def test_profile_cli_success_is_json_metadata(
    workbook_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert profile_main([str(workbook_path)]) == 0
    output = capsys.readouterr()
    assert json.loads(output.out) == profile_workbook(workbook_path)
    assert not output.err
    assert "WO-1001" not in output.out


@pytest.mark.parametrize(
    "failure", ["grid", "rows", "member", "text", "corrupt", "xml", "dtd", "missing"]
)
def test_profile_cli_rejection_is_json_and_does_not_expose_parser_input(
    workbook_path: Path,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    part = "xl/worksheets/sheet1.xml"
    private = "CELL_PAYLOAD_NOT_FOR_REPORT"
    if failure == "text":
        _repeat_shared_reference(workbook_path, b"x" * (1024 * 1024), 65)
    elif failure in {"grid", "rows"}:
        dimension = "A1:Z20000" if failure == "grid" else "A1:Z60000"
        _rewrite(
            workbook_path,
            part,
            (
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                f'<dimension ref="{dimension}"/></worksheet>'
            ).encode(),
        )
    elif failure == "member":
        workbook_path.write_bytes(
            _claim_size(workbook_path.read_bytes(), part, ExcelReadLimits().max_member_bytes + 1, 0)
        )
    elif failure == "corrupt":
        workbook_path.write_text(private)
    elif failure == "xml":
        _rewrite(workbook_path, part, f"<{private}>".encode())
    elif failure == "dtd":
        _rewrite(workbook_path, part, f"<!DOCTYPE {private}><{private}/>".encode())
    else:
        workbook_path.unlink()
    assert profile_main([str(workbook_path)]) == 1
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert report["file"] == str(workbook_path)
    assert report["accepted"] is False
    assert report["rejection_policy"] == "abort_index_build"
    assert report["limits"] == asdict(ExcelReadLimits())
    assert "archive" not in report and "xml_parts" not in report
    assert "do not raise budgets or skip" in report["action"]
    if failure in {"grid", "rows", "member", "text"}:
        assert report["error"]["part"] == part
        assert report["error"]["observed"] > report["error"]["limit"]
    assert private not in output.out + output.err
    assert not output.err


@pytest.mark.parametrize("accepted", [False, True])
def test_profile_cli_standalone_entry_point(
    workbook_path: Path, tmp_path: Path, accepted: bool
) -> None:
    if not accepted:
        workbook_path.write_bytes(b"not a workbook")
    script = Path(__file__).resolve().parents[1] / (
        "src/agents_gym_core/common_tools/vector_database/profile_excel.py"
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            str(workbook_path),
        ],
        env={
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "PYTHONDONTWRITEBYTECODE": "1",
            "AGENTS_GYM_SKIP_DOTENV": "1",
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == (0 if accepted else 1), completed.stderr
    assert json.loads(completed.stdout)["accepted"] is accepted
    assert "WO-1001" not in completed.stdout + completed.stderr


def test_profile_cli_invalid_arguments_exit_two() -> None:
    with pytest.raises(SystemExit) as error:
        profile_main([])
    assert error.value.code == 2


def test_standalone_profile_never_imports_office_or_provider_packages(
    workbook_path: Path, tmp_path: Path
) -> None:
    directory = Path(__file__).resolve().parents[1] / (
        "src/agents_gym_core/common_tools/vector_database"
    )
    script = (
        "import importlib.abc, sys; "
        "blocked = {'pandas', 'openpyxl', 'unstructured', 'langchain_core', "
        "'langchain_community', 'langchain_chroma', 'uipath', 'dotenv', 'agents_gym_core'}; "
        "exec('class Guard(importlib.abc.MetaPathFinder):\\n"
        " def find_spec(self, fullname, path=None, target=None):\\n"
        '  if fullname.split(".")[0] in blocked: raise AssertionError(fullname)\'); '
        "sys.meta_path.insert(0, Guard()); "
        f"sys.path.insert(0, {str(directory)!r}); "
        "import profile_excel; "
        f"raise SystemExit(profile_excel.main([{str(workbook_path)!r}]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env={
            "PATH": os.defpath,
            "HOME": str(tmp_path),
            "PYTHONDONTWRITEBYTECODE": "1",
            "AGENTS_GYM_SKIP_DOTENV": "1",
        },
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["accepted"] is True
    assert "WO-1001" not in completed.stdout + completed.stderr


@pytest.mark.parametrize("kind", ["calc", "external"])
def test_profile_auxiliary_xml_reports_generic_caps_without_worksheet_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    import openpyxl
    import pandas

    path = tmp_path / "auxiliary.xlsx"
    part = _auxiliary_workbook(path, kind)
    alias = "xl/auxiliary.bin"
    _rename_part(path, part, alias)
    parser = Mock(side_effect=AssertionError("Auxiliary profiling must not parse Office files"))
    monkeypatch.setattr(openpyxl, "load_workbook", parser)
    monkeypatch.setattr(pandas, "read_excel", parser)
    profile = profile_workbook(path)
    auxiliary = next(item for item in profile["xml_parts"] if item["part"] == alias)
    assert auxiliary["nodes"] > 0 and auxiliary["depth"] > 0
    assert (auxiliary["rows"], auxiliary["columns"], auxiliary["grid_cells"]) == (0, 0, 0)
    assert auxiliary["projected_cell_text_bytes"] == 0
    assert profile["package"]["grid_cells"] == 2
    assert profile["package"]["sheets"] == 1
    encoded = json.dumps(profile)
    for private in ("Result", "40+2", "First", "Second"):
        assert private not in encoded
    parser.assert_not_called()


def test_profile_long_body_projection_is_bounded_and_contains_no_cell_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import openpyxl
    import pandas

    path = tmp_path / "long_body.xlsx"
    _long_body_workbook(path)
    parser = Mock(side_effect=AssertionError("Projection must stay independent of pandas"))
    monkeypatch.setattr(openpyxl, "load_workbook", parser)
    monkeypatch.setattr(pandas, "read_excel", parser)
    profile = profile_workbook(path)
    assert profile["package"]["grid_cells"] == 5000
    assert 121_361 <= profile["package"]["projected_cell_text_bytes"] < 1_000_000
    assert "Result" not in json.dumps(profile)
    assert "x" * 64 not in json.dumps(profile)
    parser.assert_not_called()
