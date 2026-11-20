import os
import tempfile
import threading
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# src/agents_gym_core/common_tools/spreadsheet_tools.py
_PACKAGE_DIR = Path(__file__).resolve().parents[1]
_PACKAGE_PREFIX = ("src", "agents_gym_core")


def _is_agent_document(path: Path) -> bool:
    """Whether ``path`` lies in an ``agents/<agent>/documents`` directory."""
    parts = path.parts
    return any(parts[i] == "agents" and parts[i + 2] == "documents" for i in range(len(parts) - 3))


def _in_filter_output_dir(path: Path) -> bool:
    directory = _filter_output_dir
    return directory is not None and path.is_relative_to(Path(directory.name).resolve())


def _resolve_data_path(file_path: str) -> str:
    """Find a data file and refuse any file that is not agent data.

    An eval set may pass repo-relative paths such as
    src/agents_gym_core/agents/<agent>/documents/x.csv, which are looked up in the installed
    package first, so a run started anywhere finds them. The model chooses the path, so only
    files in an agent's ``documents`` directory, or files this module wrote, can be read:
    never an eval set, a prompt or anything else on disk.
    """
    path = Path(file_path)
    parts = path.parts
    if not path.is_absolute() and parts[: len(_PACKAGE_PREFIX)] == _PACKAGE_PREFIX:
        candidate = _PACKAGE_DIR.joinpath(*parts[len(_PACKAGE_PREFIX) :])
        if candidate.exists():
            path = candidate
    resolved = path.resolve()
    if not (_is_agent_document(resolved) or _in_filter_output_dir(resolved)):
        raise PermissionError(f"{file_path} is not a file in an agent's documents directory")
    return str(resolved)


_filter_output_dir: tempfile.TemporaryDirectory[str] | None = None
_filter_output_dir_lock = threading.Lock()


def _new_filter_output_path(stem: str) -> str:
    """Return a new file path in a directory that is removed when the process exits.

    The directory is created on first use and shared by every call in the process. Each
    path is unique, so parallel runs do not overwrite each other's file. The lock makes the
    first use safe under concurrent tool calls.
    """
    global _filter_output_dir
    with _filter_output_dir_lock:
        if _filter_output_dir is None:
            _filter_output_dir = tempfile.TemporaryDirectory(prefix="spreadsheet_tools_")
        directory = _filter_output_dir.name
    fd, path = tempfile.mkstemp(prefix=f"{stem}_", suffix="_filtered.csv", dir=directory)
    os.close(fd)
    return path


class DataPreviewOutput(BaseModel):
    """Output of ``data_preview_tool``."""

    preview_data: list[dict[str, Any]] | None = Field(
        description="First n_rows of the data as a list of dictionaries"
    )
    total_rows: int = Field(description="Total number of rows in the file")
    columns: list[str] = Field(description="List of column names")
    file_type: str | None = Field(description="Type of file read ('csv' or 'excel')")
    error: str | None = Field(default=None, description="Error message if something went wrong")


class ListColumnsOutput(BaseModel):
    """Output of ``list_columns_tool``."""

    columns: list[str] = Field(description="List of column names")
    file_type: str | None = Field(description="Type of file read ('csv' or 'excel')")
    error: str | None = Field(default=None, description="Error message if something went wrong")


class DescribeOutput(BaseModel):
    """Output of ``describe_tool``."""

    statistics: dict[str, dict[str, float]] = Field(
        description="Statistical summary of numerical columns"
    )
    file_type: str | None = Field(description="Type of file read ('csv' or 'excel')")
    error: str | None = Field(default=None, description="Error message if something went wrong")


class FilterCSVOutput(BaseModel):
    """Output of ``filter_csv_tool``."""

    filtered_file_path: str | None = Field(description="Path where the filtered CSV was saved")
    total_rows: int = Field(description="Total number of rows in the filtered file")
    error: str | None = Field(default=None, description="Error message if something went wrong")


@tool
def data_preview_tool(
    file_path: Annotated[str, "Path to the data file (.csv, .xls, or .xlsx)"],
    n_rows: Annotated[int | None, "Number of rows to preview. Defaults to 20."] = 20,
    sheet_name: Annotated[
        str | None, "Sheet name for Excel files. Defaults to None (first sheet)."
    ] = None,
) -> DataPreviewOutput:
    """Read and preview data from a CSV or Excel file."""
    try:
        file_path = _resolve_data_path(file_path)
        file_type = "csv" if file_path.lower().endswith(".csv") else "excel"

        if file_type == "csv":
            df = pd.read_csv(file_path, nrows=n_rows)
            total_rows = len(pd.read_csv(file_path, usecols=[0]))
        else:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=n_rows)
            else:
                df = pd.read_excel(file_path, nrows=n_rows)
            if sheet_name:
                total_df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                total_df = pd.read_excel(file_path)
            total_rows = len(total_df)

        preview_data: list[dict[str, Any]] = [
            {str(k): v for k, v in row.items()} for row in df.to_dict(orient="records")
        ]

        return DataPreviewOutput(
            preview_data=preview_data,
            total_rows=total_rows,
            columns=df.columns.tolist(),
            file_type=file_type,
            error=None,
        )
    except Exception as exc:
        return DataPreviewOutput(
            preview_data=None,
            total_rows=0,
            columns=[],
            file_type=None,
            error=f"Error reading file: {str(exc)}",
        )


@tool
def describe_tool(
    file_path: Annotated[str, "Path to the data file (.csv, .xls, or .xlsx)"],
    sheet_name: Annotated[
        str | None, "Sheet name for Excel files. Defaults to None (first sheet)."
    ] = None,
) -> DescribeOutput:
    """Get statistical descriptions of data from a CSV or Excel file."""
    try:
        file_path = _resolve_data_path(file_path)
        file_type = "csv" if file_path.lower().endswith(".csv") else "excel"

        if file_type == "csv":
            data = pd.read_csv(file_path)
        else:
            if sheet_name:
                data = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                data = pd.read_excel(file_path)

        stats_df = data.describe()
        stats_dict: dict[str, dict[str, float]] = {
            str(col): {str(k): float(v) for k, v in stats_df[col].to_dict().items()}
            for col in stats_df.columns
            if pd.api.types.is_numeric_dtype(data[col])
        }

        return DescribeOutput(
            statistics=stats_dict,
            file_type=file_type,
            error=None,
        )
    except Exception as exc:
        return DescribeOutput(
            statistics={},
            file_type=None,
            error=f"Error reading file: {str(exc)}",
        )


@tool
def filter_csv_tool(
    file_path: Annotated[str, "Path to the data file (.csv)"],
    column: Annotated[str, "Column name to filter on"],
    value: Annotated[str, "Value to filter for in the specified column"],
    output_path: Annotated[
        str | None,
        "Ignored unless it names a file this tool created earlier; by default a new temporary file is used",
    ] = None,
) -> FilterCSVOutput:
    """Filter rows in a CSV file based on a column value."""
    try:
        file_path = _resolve_data_path(file_path)
        if not file_path.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported")

        df = pd.read_csv(file_path)
        filtered_df = df[df[column].astype(str) == str(value)]

        # Writes go only to this module's temporary directory.
        if output_path is None or not _in_filter_output_dir(Path(output_path).resolve()):
            stem = os.path.splitext(os.path.basename(file_path))[0]
            output_path = _new_filter_output_path(stem)

        filtered_df.to_csv(output_path, index=False)

        return FilterCSVOutput(
            filtered_file_path=output_path,
            total_rows=len(filtered_df),
            error=None,
        )
    except Exception as exc:
        return FilterCSVOutput(
            filtered_file_path=None,
            total_rows=0,
            error=f"Error filtering file: {str(exc)}",
        )


@tool
def list_columns_tool(
    file_path: Annotated[str, "Path to the data file (.csv, .xls, or .xlsx)"],
    sheet_name: Annotated[
        str | None, "Sheet name for Excel files. Defaults to None (first sheet)."
    ] = None,
) -> ListColumnsOutput:
    """List columns in a CSV or Excel file."""
    try:
        file_path = _resolve_data_path(file_path)
        file_type = "csv" if file_path.lower().endswith(".csv") else "excel"

        if file_type == "csv":
            data = pd.read_csv(file_path, nrows=0)
        else:
            if sheet_name:
                data = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
            else:
                data = pd.read_excel(file_path, nrows=0)

        return ListColumnsOutput(
            columns=data.columns.tolist(),
            file_type=file_type,
            error=None,
        )
    except Exception as exc:
        return ListColumnsOutput(
            columns=[],
            file_type=None,
            error=f"Error reading file: {str(exc)}",
        )
