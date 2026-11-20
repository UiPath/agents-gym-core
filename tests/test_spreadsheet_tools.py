"""The spreadsheet tools find repo-relative data files and write filtered data outside the package."""

import gc
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from agents_gym_core.common_tools import spreadsheet_tools
from agents_gym_core.common_tools.spreadsheet_tools import data_preview_tool, filter_csv_tool

# A fixture, not a real agent document.
REPO_RELATIVE = "src/agents_gym_core/agents/demo/documents/data.csv"


@pytest.fixture
def package_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    package = tmp_path / "package"
    documents = package / "agents" / "demo" / "documents"
    documents.mkdir(parents=True)
    (documents / "data.csv").write_text("city,country\nParis,France\nLyon,France\nRome,Italy\n")
    monkeypatch.setattr(spreadsheet_tools, "_PACKAGE_DIR", package)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return package


def test_a_repo_relative_path_resolves_from_another_directory(package_dir: Path) -> None:
    result = data_preview_tool.invoke({"file_path": REPO_RELATIVE, "n_rows": 2})

    assert result.error is None
    assert result.total_rows == 3
    assert result.columns == ["city", "country"]


def test_filtering_writes_a_new_file_outside_the_package(package_dir: Path) -> None:
    query = {"file_path": REPO_RELATIVE, "column": "country", "value": "France"}

    first = filter_csv_tool.invoke(query)
    second = filter_csv_tool.invoke(query)

    assert first.error is None and first.total_rows == 2
    assert first.filtered_file_path != second.filtered_file_path
    assert package_dir not in Path(first.filtered_file_path).parents
    assert Path(first.filtered_file_path).read_text().splitlines()[1:] == [
        "Paris,France",
        "Lyon,France",
    ]


def test_an_unreadable_path_returns_an_error_instead_of_raising(package_dir: Path) -> None:
    result = data_preview_tool.invoke({"file_path": "x" * 300 + ".csv"})

    assert result.error is not None


def test_concurrent_first_use_keeps_every_returned_file(
    package_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Slow down directory creation so every thread reaches the first-use check before any
    # of them stores the directory.
    created: list[tempfile.TemporaryDirectory[str]] = []
    real = tempfile.TemporaryDirectory

    def slow_temporary_directory(*args: Any, **kwargs: Any) -> tempfile.TemporaryDirectory[str]:
        time.sleep(0.05)
        directory = real(*args, **kwargs)
        created.append(directory)
        return directory

    monkeypatch.setattr(spreadsheet_tools, "_filter_output_dir", None)
    monkeypatch.setattr(tempfile, "TemporaryDirectory", slow_temporary_directory)
    query = {"file_path": REPO_RELATIVE, "column": "country", "value": "France"}
    calls = 16
    start = threading.Barrier(calls)

    def call(_: int) -> str:
        start.wait()
        result = filter_csv_tool.invoke(query)
        assert result.error is None
        return result.filtered_file_path

    with ThreadPoolExecutor(calls) as pool:
        paths = list(pool.map(call, range(calls)))
    gc.collect()

    assert len(created) == 1
    assert len(set(paths)) == calls
    assert [p for p in paths if not Path(p).exists()] == []


def test_files_outside_agent_documents_cannot_be_read(package_dir: Path, tmp_path: Path) -> None:
    answers = package_dir / "agents" / "demo" / "evaluations" / "answers.csv"
    answers.parent.mkdir(parents=True)
    answers.write_text("question,answer\nq,a\n")
    elsewhere = tmp_path / "secret.csv"
    elsewhere.write_text("key\nvalue\n")

    for path in (
        "src/agents_gym_core/agents/demo/evaluations/answers.csv",
        str(answers),
        str(elsewhere),
        str(package_dir / "agents" / "demo" / "documents" / ".." / ".." / "demo" / "evaluations"),
    ):
        result = data_preview_tool.invoke({"file_path": path})
        assert result.error is not None and "documents directory" in result.error, path


def test_a_filtered_file_can_be_read_back(package_dir: Path) -> None:
    query = {"file_path": REPO_RELATIVE, "column": "country", "value": "Italy"}
    filtered = filter_csv_tool.invoke(query).filtered_file_path

    result = data_preview_tool.invoke({"file_path": filtered})

    assert result.error is None and result.total_rows == 1


def test_an_output_path_outside_the_temporary_directory_is_ignored(
    package_dir: Path, tmp_path: Path
) -> None:
    target = tmp_path / "overwritten.csv"
    query = {"file_path": REPO_RELATIVE, "column": "country", "value": "France"}

    result = filter_csv_tool.invoke({**query, "output_path": str(target)})

    assert result.error is None and result.filtered_file_path != str(target)
    assert not target.exists()
