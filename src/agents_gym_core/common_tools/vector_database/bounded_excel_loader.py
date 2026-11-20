"""Bound XLSX work before pandas/openpyxl or any Office parser.

This module validates ZIP structure and XML content. Genuine BIFF and Word files use their
existing format-specific parsing paths.

The parser receives a rewritten stored ZIP, never the original container. ZIP
directories must be single-disk/32-bit; local ZIP64 sizes and data descriptors
are supported. OPC XML aliases are resolved by content types and relationships.
Ambiguous/encoded part URIs and noncanonical relationship-part locations fail
closed with a re-save instruction rather than acquiring a second parser view.
"""

from __future__ import annotations

import io
import math
import os
import posixpath
import re
import stat
import struct
import zlib
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from xml.etree.ElementTree import Element
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from defusedxml.ElementTree import iterparse

_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_CONTENT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_CELL_REFERENCE = re.compile(r"([A-Z]{1,3})([1-9][0-9]{0,6})\Z")
_END_HEADER = struct.Struct("<4s4H2IH")
_CENTRAL_HEADER = struct.Struct("<4s6H3I5H2I")
_LOCAL_HEADER = struct.Struct("<4s5H3I2H")
_BLOCK_BYTES = 64 * 1024
_ZIP_MAGIC = b"PK\x03\x04"
_SHARED_STRINGS_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sharedstrings+xml"
)
# String cells are charged their escaped text. pandas may still infer a column of
# digit-bearing strings as numbers; a float repr is at most 24 characters.
_NUMERIC_TEXT_BYTES = 24
# Non-string scalars (numbers, booleans, dates) render through pandas formatting:
# a datetime with microseconds is 26 characters and a Timedelta at most 29.
_SCALAR_FORMAT_BYTES = 32
# openpyxl's pandas reader turns an integral float into a Python int, so a cached
# 1E+308 renders as a 309-digit integer. Below this magnitude it fits the allowance.
_INTEGRAL_DIGITS_FROM = 1e16
_STRING_CELL_TYPES = frozenset({"s", "str", "inlineStr", "e"})
_DECIMAL_DIGIT = re.compile(r"\d")
_JSON_ESCAPED = re.compile(r"[\x00-\x1f\x7f-\uffff]")
_JSON_ASTRAL = re.compile(r"[\U00010000-\U0010ffff]")
_GRID_CELL_BYTES = 64
_SHEET_TEXT_BYTES = 1024
_WORKBOOK_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.template.main+xml",
    "application/vnd.ms-excel.sheet.macroenabled.main+xml",
    "application/vnd.ms-excel.template.macroenabled.main+xml",
}
_XML_RELATION_ROLES = {
    "officeDocument",
    "worksheet",
    "chartsheet",
    "sharedStrings",
    "styles",
    "theme",
    "comments",
    "table",
    "drawing",
    "chart",
    "chartUserShapes",
    "vmlDrawing",
    "pivotTable",
    "pivotCacheDefinition",
    "pivotCacheRecords",
    "externalLink",
    "calcChain",
    "connections",
    "queryTable",
    "core-properties",
    "extended-properties",
    "custom-properties",
    "metadata",
}


@dataclass(frozen=True)
class ExcelReadLimits:
    max_archive_bytes: int = 16 * 1024 * 1024
    max_members: int = 1024
    max_member_bytes: int = 16 * 1024 * 1024
    max_expanded_bytes: int = 64 * 1024 * 1024
    max_snapshot_bytes: int = (
        80 * 1024 * 1024
    )  # Separate emitted ZIP cap, not a larger read budget.
    max_name_bytes: int = 1024
    max_sheets: int = 32
    max_rows: int = 50_000
    max_columns: int = 256
    max_grid_cells: int = 500_000
    max_xml_depth: int = 128
    max_xml_nodes: int = 1_000_000
    max_cell_text_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        if any(type(value) is not int or value <= 0 for value in vars(self).values()):
            raise ValueError("Excel read limits must be positive integers")


class ExcelResourceLimitError(ValueError):
    """The workbook exceeds the ingestion budget; no partial documents are returned."""

    def __init__(
        self,
        message: str,
        *,
        limit: int,
        observed: int,
        part: str | None = None,
    ) -> None:
        super().__init__(message)
        self.limit = limit
        self.observed = observed
        self.part = part
        self.file_path: str | None = None

    def __str__(self) -> str:
        return (
            f"{super().__str__()} (file={self.file_path or '<snapshot>'!r}, "
            f"part={self.part or '<archive>'!r}, limit={self.limit}, observed={self.observed})"
        )


class _BoundedSnapshot(io.BytesIO):
    def __init__(self, limit: int) -> None:
        super().__init__()
        self.limit = limit

    def write(self, data: Any) -> int:
        if self.tell() + len(data) > self.limit:
            raise ExcelResourceLimitError(
                "XLSX canonical snapshot limit exceeded",
                limit=self.limit,
                observed=self.tell() + len(data),
                part="<canonical ZIP>",
            )
        return super().write(data)


def _directory_start(snapshot: bytes, limits: ExcelReadLimits) -> int:
    """Bound central-directory allocation before asking ZipFile to parse it.

    Small-workbook policy: single-disk, ordinary end records only. Local ZIP64
    size fields and streaming data descriptors are supported below.
    """
    end = snapshot.rfind(b"PK\x05\x06", max(0, len(snapshot) - 65557))
    if end < 0 or end + _END_HEADER.size > len(snapshot):
        raise ValueError("Corrupt XLSX ZIP end record")
    _, disk, directory_disk, disk_count, count, size, start, comment = _END_HEADER.unpack_from(
        snapshot, end
    )
    if end + _END_HEADER.size + comment != len(snapshot):
        raise ValueError("Corrupt XLSX ZIP end record")
    if disk or directory_disk or disk_count != count:
        raise ValueError("Multi-disk XLSX ZIP archives are unsupported")
    if count == 0xFFFF or start == 0xFFFFFFFF or size == 0xFFFFFFFF or start + size != end:
        raise ValueError("Unsupported ZIP64 directory or inconsistent XLSX ZIP bounds")
    cursor, actual_count = start, 0
    while cursor < end:
        if cursor + _CENTRAL_HEADER.size > end:
            raise ValueError("Corrupt XLSX ZIP directory")
        header = _CENTRAL_HEADER.unpack_from(snapshot, cursor)
        if header[0] != b"PK\x01\x02" or header[13]:
            raise ValueError("Corrupt or multi-disk XLSX ZIP directory")
        cursor += _CENTRAL_HEADER.size + header[10] + header[11] + header[12]
        actual_count += 1
        if actual_count > limits.max_members:
            raise ExcelResourceLimitError(
                "XLSX archive member limit exceeded",
                limit=limits.max_members,
                observed=actual_count,
            )
        if cursor > end:
            raise ValueError("Corrupt XLSX ZIP directory")
    if actual_count != count:
        raise ValueError("XLSX ZIP directory count mismatch")
    return start


def _local_sizes(extra: memoryview, size: int, compressed: int) -> tuple[int, int]:
    cursor, zip64 = 0, None
    while cursor < len(extra):
        if cursor + 4 > len(extra):
            raise ValueError("Truncated XLSX ZIP extra field")
        kind, length = struct.unpack_from("<HH", extra, cursor)
        cursor += 4
        if cursor + length > len(extra):
            raise ValueError("Truncated XLSX ZIP extra field")
        if kind == 1:
            if zip64 is not None:
                raise ValueError("Duplicate XLSX ZIP64 extra field")
            zip64 = extra[cursor : cursor + length]
        cursor += length
    offset = 0
    values = []
    for value in (size, compressed):
        if value == 0xFFFFFFFF:
            if zip64 is None or offset + 8 > len(zip64):
                raise ValueError("Missing XLSX ZIP64 size")
            value = struct.unpack_from("<Q", zip64, offset)[0]
            offset += 8
        values.append(value)
    return values[0], values[1]


def _inflate(data: memoryview, limit: int) -> bytes:
    """Count actual raw-DEFLATE output, independent of ZipExtFile's size clipping."""
    inflater = zlib.decompressobj(-zlib.MAX_WBITS)
    result = bytearray()
    cursor, pending = 0, b""
    while True:
        # Empty input drains decoded bytes zlib can retain after consuming its input.
        chunk = pending if pending else data[cursor : cursor + _BLOCK_BYTES]
        if not pending:
            cursor += len(chunk)
        output = inflater.decompress(chunk, min(_BLOCK_BYTES, limit - len(result) + 1))
        if len(result) + len(output) > limit:
            raise ExcelResourceLimitError(
                "XLSX actual expanded archive/member limit exceeded",
                limit=limit,
                observed=len(result) + len(output),
            )
        result.extend(output)
        pending = inflater.unconsumed_tail
        if inflater.eof:
            if inflater.unused_data or pending or cursor != len(data):
                raise ValueError("Trailing XLSX DEFLATE stream data")
            break
        if not output and not pending and cursor == len(data):
            raise ValueError("Truncated XLSX DEFLATE stream")
        if not output and len(pending) == len(chunk):
            raise ValueError("XLSX DEFLATE stream made no progress")
    return bytes(result)


def _read_member(
    snapshot: bytes,
    member: ZipInfo,
    boundary: int,
    limit: int,
) -> bytes:
    view = memoryview(snapshot)
    offset = member.header_offset
    if offset < 0 or offset + _LOCAL_HEADER.size > boundary:
        raise ValueError("Invalid XLSX local header bounds")
    signature, version, flags, method, _, _, crc, compressed, size, name_len, extra_len = (
        _LOCAL_HEADER.unpack_from(view, offset)
    )
    if signature != b"PK\x03\x04":
        raise ValueError("Invalid XLSX local header signature")
    if flags != member.flag_bits or method != member.compress_type:
        raise ValueError("XLSX local/central flags or compression mismatch")
    if version != (member.extract_version | (member.reserved << 8)):
        raise ValueError("XLSX local/central version mismatch")
    allowed_flags = 0x0808 | (6 if method == ZIP_DEFLATED else 0)
    if flags & ~allowed_flags or method not in (ZIP_STORED, ZIP_DEFLATED):
        raise ValueError("Encrypted or unsupported XLSX archive member")
    start = offset + _LOCAL_HEADER.size
    payload = start + name_len + extra_len
    end = payload + member.compress_size
    if payload > boundary or end > boundary:
        raise ValueError("Overlapping or truncated XLSX ZIP member")
    codec = "utf-8" if flags & 0x0800 else "cp437"
    if view[start : start + name_len].tobytes().decode(codec) != member.orig_filename:
        raise ValueError("XLSX local/central filename mismatch")
    zip64 = size == 0xFFFFFFFF or compressed == 0xFFFFFFFF
    size, compressed = _local_sizes(view[start + name_len : payload], size, compressed)
    if flags & 8:
        if (
            crc not in (0, member.CRC)
            or size not in (0, member.file_size)
            or compressed not in (0, member.compress_size)
        ):
            raise ValueError("XLSX streaming local/central size or CRC mismatch")
    elif (crc, compressed, size) != (member.CRC, member.compress_size, member.file_size):
        raise ValueError("XLSX local/central size or CRC mismatch")
    raw = view[payload:end]
    if method == ZIP_STORED:
        if len(raw) > limit:
            raise ExcelResourceLimitError(
                "XLSX actual expanded archive/member limit exceeded",
                limit=limit,
                observed=len(raw),
            )
        result = raw.tobytes()
    else:
        result = _inflate(raw, limit)
    if len(result) != member.file_size or zlib.crc32(result) != member.CRC:
        raise ValueError("XLSX actual size or CRC mismatch")
    if flags & 8:
        descriptor = struct.Struct("<IQQ" if zip64 else "<III")
        for signature_size in (0, 4):
            if signature_size and view[end : end + 4] != b"PK\x07\x08":
                continue
            at = end + signature_size
            if at + descriptor.size <= boundary and descriptor.unpack_from(view, at) == (
                member.CRC,
                member.compress_size,
                len(result),
            ):
                break
        else:
            raise ValueError("XLSX data descriptor mismatch")
    return result


def _cell_position(reference: str, limits: ExcelReadLimits) -> tuple[int, int]:
    match = _CELL_REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError("Invalid XLSX cell reference")
    column = 0
    for letter in match[1]:
        column = column * 26 + ord(letter) - ord("A") + 1
    row = int(match[2])
    if row > limits.max_rows:
        raise ExcelResourceLimitError(
            "XLSX row/column limit exceeded (rows)", limit=limits.max_rows, observed=row
        )
    if column > limits.max_columns:
        raise ExcelResourceLimitError(
            "XLSX row/column limit exceeded (columns)",
            limit=limits.max_columns,
            observed=column,
        )
    return row, column


def _escaped_text_bytes(text: str | None) -> int:
    if not text:
        return 0
    html_bytes = (
        len(text.encode("utf-8"))
        + 4 * text.count("&")
        + 3 * (text.count("<") + text.count(">"))
        + 5 * (text.count('"') + text.count("'"))
        + text.count("\t")
        + text.count("\n")
        + text.count("\r")
    )
    # JSON rows use ensure_ascii=True. Count escapes without retaining cell
    # values or constructing potentially much larger serialized strings.
    json_bytes = (
        len(text)
        + text.count('"')
        + text.count("\\")
        + 5 * sum(1 for _ in _JSON_ESCAPED.finditer(text))
        + 11 * sum(1 for _ in _JSON_ASTRAL.finditer(text))
    )
    return max(html_bytes, json_bytes)


@lru_cache(maxsize=1024)
def _integral_digits(value: str) -> int:
    """Rendered length of a cached number that pandas receives as a Python int."""
    if len(value) > 320:
        return 0  # Its lexical text, already charged, is longer than any float's digits.
    try:
        number = float(value)
    except ValueError:
        return 0
    if not math.isfinite(number) or not number.is_integer():
        return 0
    if abs(number) < _INTEGRAL_DIGITS_FROM:
        return 0
    return len(str(int(number)))


class _CellTextCounter:
    """Count lengths, never expanded cell values or retained XML subtrees."""

    def __init__(self, limits: ExcelReadLimits, strings: list[int], *, table: bool = False) -> None:
        self.limits = limits
        self.strings = strings
        self.table = table
        self.projected = 0
        self.item_depth: int | None = None
        self.cell_depth: int | None = None
        self.row = self.column = 0
        self.header_bytes: dict[int, int] = {}
        self.cell_type = ""
        self.text_bytes = self.values = self.digits = 0
        self.has_digit = False
        self.index: int | None = None
        self.value_depth: int | None = None
        self.value_has_children = False

    def _count(self, text: str | None, *, digits: bool) -> None:
        if not text:
            return
        # A digit only matters while the text is shorter than any numeric repr.
        if digits and not self.has_digit and self.text_bytes < _NUMERIC_TEXT_BYTES:
            self.has_digit = _DECIMAL_DIGIT.search(text) is not None
        self.text_bytes += _escaped_text_bytes(text)

    def _string_bytes(self) -> int:
        if self.has_digit:
            return max(self.text_bytes, _NUMERIC_TEXT_BYTES)
        return self.text_bytes

    def start(self, element: Element, depth: int, *, cell: bool, row: int, column: int) -> None:
        if self.table:
            if element.tag == f"{{{_MAIN_NS}}}si":
                if self.item_depth is not None or depth != 2:
                    raise ValueError("Malformed XLSX shared-string table")
                self.item_depth, self.text_bytes, self.has_digit = depth, 0, False
            return
        if cell:
            self.cell_depth, self.cell_type = depth, element.get("t", "n")
            self.row, self.column = row, column
            self.text_bytes = self.values = self.digits = 0
            self.has_digit = False
            self.index = self.value_depth = None
        elif self.cell_depth is not None:
            if depth == self.cell_depth + 1 and element.tag == f"{{{_MAIN_NS}}}v":
                self.value_depth, self.value_has_children = depth, False
            elif self.value_depth is not None and depth > self.value_depth:
                self.value_has_children = True

    def end(self, element: Element, depth: int) -> None:
        if self.table:
            if self.item_depth is not None:
                self._count(element.text, digits=True)
                if depth > self.item_depth:
                    self._count(element.tail, digits=True)
                else:
                    self.strings.append(self._string_bytes())
                    self.item_depth = None
            return
        if self.cell_depth is None:
            return
        # A shared-string index is not cell text, so its digits never make it numeric.
        inline_text = self.cell_type in {"str", "inlineStr"}
        self._count(element.text, digits=inline_text)
        if depth > self.cell_depth:
            self._count(element.tail, digits=inline_text)
        if depth == self.value_depth:
            self.values += 1
            if self.cell_type == "s":
                value = element.text or ""
                if (
                    self.values != 1
                    or self.value_has_children
                    or not value
                    or len(value) > 10
                    or not value.isascii()
                    or not value.isdecimal()
                ):
                    raise ValueError("Malformed XLSX shared-string reference")
                self.index = int(value)
                if self.index >= len(self.strings):
                    raise ValueError("XLSX shared-string reference is out of range")
            elif self.cell_type == "n" and element.text:
                self.digits = max(self.digits, _integral_digits(element.text))
            self.value_depth = None
        if depth == self.cell_depth:
            if self.cell_type == "s":
                if self.index is None or self.values != 1:
                    raise ValueError("Missing or ambiguous XLSX shared-string reference")
                self.text_bytes += self.strings[self.index]
            if self.text_bytes or self.values:
                if self.cell_type in _STRING_CELL_TYPES:
                    # pandas does not reformat strings. Numeric inference is floored above.
                    size = self._string_bytes()
                else:
                    # Number/date/boolean formatting can exceed the cached lexical size.
                    size = self.text_bytes + max(_SCALAR_FORMAT_BYTES, self.digits)
                self.projected += size
                if self.row == 1:
                    self.header_bytes[self.column] = size
            if self.projected > self.limits.max_cell_text_bytes:
                raise ExcelResourceLimitError(
                    "XLSX projected cell text limit exceeded",
                    limit=self.limits.max_cell_text_bytes,
                    observed=self.projected,
                )
            self.cell_depth = None

    def materialized_bytes(self, rows: int, columns: int) -> int:
        # pandas names blank headers "Unnamed: n" and mangles collisions with
        # a numeric suffix no larger than the number of columns. Count header
        # lengths, not body maxima; a long note is never padded across its column.
        suffix_bytes = 1 + len(str(columns))
        headers = sum(
            max(self.header_bytes.get(column, 0), len(f"Unnamed: {column - 1}")) + suffix_bytes
            for column in range(1, columns + 1)
        )
        return self.projected + max(0, rows - 1) * headers


@dataclass(frozen=True)
class _XmlStats:
    grid_cells: int
    sheets: int
    root: str
    rows: int
    columns: int
    observed_rows: int
    observed_columns: int
    declared_rows: int
    declared_columns: int
    declared_dimensions: int
    depth: int
    nodes: int
    projected_cell_text_bytes: int


@dataclass(frozen=True)
class _PackageStats:
    xml_parts: dict[str, _XmlStats]
    sheet_uses: dict[str, int]
    grid_cells: int
    sheets: int
    content_types: int
    workbook_relationships: int
    projected_cell_text_bytes: int


def _validate_xml(
    data: bytes,
    limits: ExcelReadLimits,
    visit: Callable[[Element], None] | None = None,
    *,
    part: str,
    text_counter: _CellTextCounter | None = None,
) -> _XmlStats:
    try:
        return _parse_xml(data, limits, visit, text_counter)
    except ExcelResourceLimitError as exc:
        exc.part = exc.part or part
        raise


def _parse_xml(
    data: bytes,
    limits: ExcelReadLimits,
    visit: Callable[[Element], None] | None,
    text_counter: _CellTextCounter | None = None,
) -> _XmlStats:
    text_counter = text_counter or _CellTextCounter(limits, [])
    max_row = max_column = sheet_count = 0
    observed_rows = observed_columns = declared_rows = declared_columns = dimensions = 0
    row_counter = column_counter = 0
    row_depth: int | None = None
    nodes = depth = 0
    stack: list[Element] = []
    root = ""
    worksheet = False
    # Parse every XML part, including shared strings/styles/relationships, before
    # openpyxl can allocate them. A declared dimension alone is not trustworthy.
    for event, element in iterparse(
        io.BytesIO(data),
        events=("start", "end"),
        forbid_dtd=True,
        forbid_entities=True,
        forbid_external=True,
    ):
        if event == "start":
            is_cell = False
            stack.append(element)
            nodes += 1
            depth = max(depth, len(stack))
            if len(stack) > limits.max_xml_depth:
                raise ExcelResourceLimitError(
                    "XLSX XML complexity limit exceeded (depth)",
                    limit=limits.max_xml_depth,
                    observed=len(stack),
                )
            if nodes > limits.max_xml_nodes:
                raise ExcelResourceLimitError(
                    "XLSX XML complexity limit exceeded (nodes)",
                    limit=limits.max_xml_nodes,
                    observed=nodes,
                )
            if len(stack) == 1:
                root = element.tag
                worksheet = root == f"{{{_MAIN_NS}}}worksheet"
            if worksheet and element.tag == f"{{{_MAIN_NS}}}row":
                if row_depth is not None:
                    raise ValueError("Nested XLSX rows are unsupported")
                reference = element.get("r")
                if reference is not None and (not reference.isascii() or not reference.isdecimal()):
                    raise ValueError("Invalid XLSX row coordinate")
                row = row_counter + 1 if reference is None else int(reference)
                if row < 1 or row > limits.max_rows:
                    raise ExcelResourceLimitError(
                        "XLSX row limit exceeded", limit=limits.max_rows, observed=row
                    )
                if row <= row_counter:
                    raise ValueError("XLSX row coordinates must increase")
                row_counter, column_counter, row_depth = row, 0, len(stack)
                max_row = max(max_row, row)
                observed_rows = max(observed_rows, row)
            elif worksheet and (
                element.tag == f"{{{_MAIN_NS}}}c"
                or (row_depth is not None and len(stack) == row_depth + 1)
            ):
                is_cell = True
                # openpyxl parses every direct row child as a cell, not only
                # children with the conventional SpreadsheetML c tag.
                if row_depth is None or len(stack) != row_depth + 1:
                    raise ValueError("XLSX cell must belong to a row")
                reference = element.get("r")
                if reference is None:
                    row, column = row_counter, column_counter + 1
                    if column > limits.max_columns:
                        raise ExcelResourceLimitError(
                            "XLSX row/column limit exceeded (columns)",
                            limit=limits.max_columns,
                            observed=column,
                        )
                else:
                    row, column = _cell_position(reference, limits)
                if row != row_counter or column <= column_counter:
                    raise ValueError("Inconsistent or non-increasing XLSX cell coordinates")
                column_counter = column
                max_row, max_column = max(max_row, row), max(max_column, column)
                observed_rows = max(observed_rows, row)
                observed_columns = max(observed_columns, column)
            text_counter.start(
                element, len(stack), cell=is_cell, row=row_counter, column=column_counter
            )
            continue
        text_counter.end(element, len(stack))
        if root == f"{{{_MAIN_NS}}}workbook" and element.tag == f"{{{_MAIN_NS}}}sheet":
            sheet_count += 1
        elif worksheet and element.tag == f"{{{_MAIN_NS}}}dimension":
            dimensions += 1
            for reference in element.get("ref", "").split(":"):
                row, column = _cell_position(reference, limits)
                max_row, max_column = max(max_row, row), max(max_column, column)
                declared_rows = max(declared_rows, row)
                declared_columns = max(declared_columns, column)
        elif worksheet and element.tag == f"{{{_MAIN_NS}}}row":
            row_depth = None
        if visit is not None:
            visit(element)
        element.clear()
        if len(stack) > 1:
            stack[-2].remove(element)
        stack.pop()
    return _XmlStats(
        grid_cells=max_row * max_column,
        sheets=sheet_count,
        root=root,
        rows=max_row,
        columns=max_column,
        observed_rows=observed_rows,
        observed_columns=observed_columns,
        declared_rows=declared_rows,
        declared_columns=declared_columns,
        declared_dimensions=dimensions,
        depth=depth,
        nodes=nodes,
        projected_cell_text_bytes=(
            text_counter.materialized_bytes(max_row, max_column)
            + max_row * max_column * _GRID_CELL_BYTES
            + (_SHEET_TEXT_BYTES if root == f"{{{_MAIN_NS}}}worksheet" else 0)
        ),
    )


def _part_name(name: str, limits: ExcelReadLimits) -> str:
    # Reject ambiguous URI encodings rather than validating one spelling and
    # allowing a downstream package reader to resolve another.
    if (
        not name
        or name.startswith("/")
        or any(char in name for char in "\\%?#:")
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
        or any(part in ("", ".", "..") for part in name.removesuffix("/").split("/"))
    ):
        raise ValueError("Unsupported noncanonical XLSX part name/URI; re-save the workbook")
    if len(name.encode("utf-8")) > limits.max_name_bytes:
        raise ExcelResourceLimitError(
            "XLSX part name limit exceeded",
            limit=limits.max_name_bytes,
            observed=len(name.encode("utf-8")),
            part=name,
        )
    return name


def _is_xml_type(content_type: str) -> bool:
    return content_type in ("application/xml", "text/xml") or content_type.endswith("+xml")


def _relationship_source(name: str) -> str:
    if name == "_rels/.rels":
        return ""
    directory, _, filename = name.rpartition("/")
    if (
        posixpath.basename(directory) != "_rels"
        or not filename.endswith(".rels")
        or len(filename) <= 5
    ):
        raise ValueError("Unsupported nonstandard XLSX relationships part; re-save the workbook")
    base = directory[:-5].rstrip("/")
    return posixpath.join(base, filename[:-5])


def _validate_package(parts: dict[str, bytes], limits: ExcelReadLimits) -> _PackageStats:
    if "[Content_Types].xml" not in parts or "_rels/.rels" not in parts:
        raise ValueError("Not an OPC XLSX workbook")
    defaults: dict[str, str] = {}
    overrides: dict[str, str] = {}

    def content_type(element: Element) -> None:
        local_name = element.tag.rsplit("}", 1)[-1]
        if (
            local_name in {"Default", "Override"}
            and element.tag != f"{{{_CONTENT_NS}}}{local_name}"
        ):
            raise ValueError("Unsupported XLSX content-type namespace")
        if element.tag == f"{{{_CONTENT_NS}}}Default":
            extension = element.get("Extension", "").lower()
            mime = element.get("ContentType", "").split(";", 1)[0].strip().lower()
            if not extension or not mime or extension in defaults:
                raise ValueError("Ambiguous XLSX default content type")
            defaults[extension] = mime
        elif element.tag == f"{{{_CONTENT_NS}}}Override":
            target = element.get("PartName", "")
            mime = element.get("ContentType", "").split(";", 1)[0].strip().lower()
            if not target.startswith("/"):
                raise ValueError("Invalid XLSX content-type part URI")
            name = _part_name(target[1:], limits)
            if name not in parts or name in overrides or not mime:
                raise ValueError("Ambiguous or missing XLSX content-type part")
            overrides[name] = mime
        if len(defaults) + len(overrides) > limits.max_members:
            raise ExcelResourceLimitError(
                "XLSX content-type declaration limit exceeded",
                limit=limits.max_members,
                observed=len(defaults) + len(overrides),
            )

    manifest = _validate_xml(
        parts["[Content_Types].xml"], limits, content_type, part="[Content_Types].xml"
    )
    if manifest.root != f"{{{_CONTENT_NS}}}Types":
        raise ValueError("Invalid XLSX content-type manifest")
    types: dict[str, str] = {}
    for name in parts:
        if name == "[Content_Types].xml" or name.endswith("/"):
            continue
        mime = overrides.get(name, defaults.get(name.rsplit(".", 1)[-1].lower(), ""))
        if not mime:
            raise ValueError("XLSX part has no declared content type")
        types[name] = mime
    workbooks = {name for name, mime in types.items() if mime in _WORKBOOK_TYPES}
    if len(workbooks) != 1:
        raise ValueError("XLSX must have one unambiguous OPC workbook part")
    workbook_part = next(iter(workbooks))
    string_parts = {name for name, mime in types.items() if mime == _SHARED_STRINGS_TYPE}
    if len(string_parts) > 1 or any(name not in overrides for name in string_parts):
        raise ValueError("Ambiguous XLSX shared-string content type; re-save the workbook")
    xml_names = {
        name
        for name in types
        if _is_xml_type(types[name]) or name.lower().endswith((".xml", ".rels"))
    }
    relationships = {
        name
        for name in types
        if name.endswith(".rels")
        or types[name] == "application/vnd.openxmlformats-package.relationships+xml"
    }
    xml_names.update(relationships)
    expected_roots: dict[str, str] = {}
    office_documents: set[str] = set()
    workbook_links: dict[str, tuple[str, str]] = {}
    relationship_ids: dict[str, set[str]] = {}
    sheet_uses: dict[str, int] = {}

    def relationship(source: str, element: Element) -> None:
        if (
            element.tag.rsplit("}", 1)[-1] == "Relationship"
            and element.tag != f"{{{_REL_NS}}}Relationship"
        ):
            raise ValueError("Unsupported XLSX relationships namespace")
        if element.tag != f"{{{_REL_NS}}}Relationship":
            return
        identifier = element.get("Id", "")
        identifiers = relationship_ids.setdefault(source, set())
        if not identifier or identifier in identifiers:
            raise ValueError("Ambiguous XLSX relationship ID")
        identifiers.add(identifier)
        mode = element.get("TargetMode", "Internal")
        if mode == "External":
            return  # Preserve hyperlinks, but never dereference them during validation.
        if mode != "Internal":
            raise ValueError("Invalid XLSX relationship target mode")
        target, role = element.get("Target", ""), element.get("Type", "").rsplit("/", 1)[-1]
        uri = urlsplit(target)
        if (
            not target
            or not role
            or uri.scheme
            or uri.netloc
            or uri.query
            or uri.fragment
            or "\\" in target
            or "%" in target
        ):
            raise ValueError("Unsupported XLSX internal relationship URI; re-save the workbook")
        name = (
            target[1:]
            if target.startswith("/")
            else posixpath.join(posixpath.dirname(source), target)
        )
        name = _part_name(posixpath.normpath(name), limits)
        if name not in types:
            raise ValueError("XLSX relationship target is missing")
        if source == workbook_part:
            workbook_links[identifier] = (name, role)
            if len(workbook_links) > limits.max_members:
                raise ExcelResourceLimitError(
                    "XLSX workbook relationship limit exceeded",
                    limit=limits.max_members,
                    observed=len(workbook_links),
                )
        if role in _XML_RELATION_ROLES:
            xml_names.add(name)
        if role in {"officeDocument", "worksheet", "chartsheet"}:
            root = {
                "officeDocument": "workbook",
                "worksheet": "worksheet",
                "chartsheet": "chartsheet",
            }[role]
            expected = f"{{{_MAIN_NS}}}{root}"
            if name in expected_roots and expected_roots[name] != expected:
                raise ValueError("Conflicting XLSX XML relationship roles")
            expected_roots[name] = expected
        if not source and role == "officeDocument":
            office_documents.add(name)

    def workbook_binding(element: Element) -> None:
        local_name = element.tag.rsplit("}", 1)[-1]
        if local_name not in {"sheet", "externalReference"}:
            return
        if element.tag != f"{{{_MAIN_NS}}}{local_name}":
            raise ValueError("Unsupported XLSX workbook child namespace")
        identifier = element.get(f"{{{_OFFICE_REL_NS}}}id", "")
        if identifier not in workbook_links:
            raise ValueError("XLSX workbook reference is missing")
        target, role = workbook_links[identifier]
        allowed = {"worksheet", "chartsheet"} if local_name == "sheet" else {"externalLink"}
        if role not in allowed:
            raise ValueError("XLSX workbook reference has an inconsistent XML role")
        if local_name == "sheet":
            sheet_uses[target] = sheet_uses.get(target, 0) + 1
            if sum(sheet_uses.values()) > limits.max_sheets:
                raise ExcelResourceLimitError(
                    "XLSX sheet limit exceeded",
                    limit=limits.max_sheets,
                    observed=sum(sheet_uses.values()),
                )

    summaries = {"[Content_Types].xml": manifest}
    strings: list[int] = []
    for name in sorted(string_parts):
        summaries[name] = _validate_xml(
            parts[name],
            limits,
            part=name,
            text_counter=_CellTextCounter(limits, strings, table=True),
        )
        if summaries[name].root != f"{{{_MAIN_NS}}}sst":
            raise ValueError("Malformed XLSX shared-string table root")
    for name in sorted(relationships):
        source = _relationship_source(name)
        if source and source not in parts:
            raise ValueError("XLSX relationships source is missing")
        summaries[name] = _validate_xml(
            parts[name],
            limits,
            lambda element, source=source: relationship(source, element),
            part=name,
            text_counter=_CellTextCounter(limits, strings),
        )
        if summaries[name].root != f"{{{_REL_NS}}}Relationships":
            raise ValueError("Invalid XLSX relationships part")
    for name in sorted(xml_names - summaries.keys()):
        summaries[name] = _validate_xml(
            parts[name],
            limits,
            workbook_binding if name == workbook_part else None,
            part=name,
            text_counter=_CellTextCounter(limits, strings),
        )
    for name, root in expected_roots.items():
        if summaries[name].root != root:
            raise ValueError("Unsupported or inconsistent XLSX SpreadsheetML namespace/root")
    if len(office_documents) != 1 or workbooks != office_documents:
        raise ValueError("XLSX must have one unambiguous OPC workbook part")
    grid_cells = projected_cell_text_bytes = 0
    for name, summary in summaries.items():
        grid_cells += summary.grid_cells * max(1, sheet_uses.get(name, 0))
        if grid_cells > limits.max_grid_cells:
            raise ExcelResourceLimitError(
                "XLSX dense grid limit exceeded",
                limit=limits.max_grid_cells,
                observed=grid_cells,
                part=name,
            )
        projected_cell_text_bytes += summary.projected_cell_text_bytes * max(
            1, sheet_uses.get(name, 0)
        )
        if projected_cell_text_bytes > limits.max_cell_text_bytes:
            raise ExcelResourceLimitError(
                "XLSX projected cell text limit exceeded",
                limit=limits.max_cell_text_bytes,
                observed=projected_cell_text_bytes,
                part=name,
            )
    sheets = sum(summary.sheets for summary in summaries.values())
    if sheets > limits.max_sheets:
        raise ExcelResourceLimitError(
            "XLSX sheet limit exceeded",
            limit=limits.max_sheets,
            observed=sheets,
            part=workbook_part,
        )
    return _PackageStats(
        xml_parts=summaries,
        sheet_uses=sheet_uses,
        grid_cells=grid_cells,
        sheets=sheets,
        content_types=len(defaults) + len(overrides),
        workbook_relationships=len(workbook_links),
        projected_cell_text_bytes=projected_cell_text_bytes,
    )


def _validate_archive(
    snapshot: bytes, limits: ExcelReadLimits, *, profile: dict[str, Any] | None = None
) -> bytes:
    try:
        directory = _directory_start(snapshot, limits)
        with ZipFile(io.BytesIO(snapshot)) as archive:
            members = archive.infolist()
            names = [member.filename for member in members]
            if len(set(names)) != len(names):
                raise ValueError("Duplicate XLSX archive members are not supported")
            ordered = sorted(members, key=lambda member: member.header_offset)
            if len({member.header_offset for member in ordered}) != len(ordered):
                raise ValueError("Overlapping XLSX local headers")
            parts: dict[str, bytes] = {}
            expanded = 0
            for index, member in enumerate(ordered):
                name = _part_name(member.filename, limits)
                if member.filename != member.orig_filename:
                    raise ValueError("Ambiguous XLSX filename")
                if member.file_size > limits.max_member_bytes:
                    raise ExcelResourceLimitError(
                        "XLSX expanded member limit exceeded",
                        limit=limits.max_member_bytes,
                        observed=member.file_size,
                        part=name,
                    )
                remaining = min(limits.max_member_bytes, limits.max_expanded_bytes - expanded)
                if member.file_size > remaining:
                    raise ExcelResourceLimitError(
                        "XLSX expanded archive limit exceeded",
                        limit=limits.max_expanded_bytes,
                        observed=expanded + member.file_size,
                        part=name,
                    )
                boundary = (
                    ordered[index + 1].header_offset if index + 1 < len(ordered) else directory
                )
                try:
                    data = _read_member(snapshot, member, boundary, remaining)
                except ExcelResourceLimitError as exc:
                    exc.part = name
                    if remaining < limits.max_member_bytes:
                        exc.limit = limits.max_expanded_bytes
                        exc.observed += expanded
                    raise
                expanded += len(data)
                if member.is_dir() and data:
                    raise ValueError("XLSX directory member must be empty")
                parts[name] = data
        package = _validate_package(parts, limits)
        output = _BoundedSnapshot(limits.max_snapshot_bytes)
        with ZipFile(output, "w", compression=ZIP_STORED) as canonical:
            for name in names:
                info = ZipInfo(name)  # Fixed timestamp, no input extras/comments/attributes.
                info.compress_type = ZIP_STORED
                canonical.writestr(info, parts[name])
        if profile is not None:
            profile.update(
                archive={
                    "input_bytes": len(snapshot),
                    "members": len(members),
                    "expanded_bytes": expanded,
                    "canonical_bytes": output.getbuffer().nbytes,
                    "parts": [
                        {"part": member.filename, "expanded_bytes": len(parts[member.filename])}
                        for member in members
                    ],
                },
                package={
                    "sheets": package.sheets,
                    "grid_cells": package.grid_cells,
                    "content_types": package.content_types,
                    "workbook_relationships": package.workbook_relationships,
                    "projected_cell_text_bytes": package.projected_cell_text_bytes,
                },
                xml_parts=[
                    {
                        "part": name,
                        "rows": summary.rows,
                        "columns": summary.columns,
                        "observed_rows": summary.observed_rows,
                        "observed_columns": summary.observed_columns,
                        "declared_rows": summary.declared_rows,
                        "declared_columns": summary.declared_columns,
                        "declared_dimensions": summary.declared_dimensions,
                        "grid_cells": summary.grid_cells,
                        "depth": summary.depth,
                        "nodes": summary.nodes,
                        "sheet_materializations": package.sheet_uses.get(name, 0),
                        "projected_cell_text_bytes": summary.projected_cell_text_bytes,
                    }
                    for name, summary in sorted(package.xml_parts.items())
                ],
            )
        return output.getvalue()
    except (BadZipFile, struct.error, zlib.error, UnicodeError) as exc:
        raise ValueError("Corrupt or non-OOXML XLSX workbook") from exc


def _read_snapshot(
    file_path: str,
    limits: ExcelReadLimits,
    *,
    profile: dict[str, Any] | None = None,
    legacy_xls: bool = False,
) -> tuple[bytes, str]:
    """Read one workbook snapshot for validation and downstream parsing.

    OOXML snapshots are validated and rewritten before parsing. With ``legacy_xls``,
    genuine BIFF bytes are returned without archive validation. Sniffing and parsing
    use the same snapshot.
    """
    try:
        fd = os.open(file_path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(fd, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("XLSX input must be a regular file")
            magic = stream.read(len(_ZIP_MAGIC)) if legacy_xls else b""
            if legacy_xls and magic != _ZIP_MAGIC:
                # pandas picks its engine from these bytes too: not ZIP is not XLSX.
                snapshot = magic + stream.read()
            else:
                if info.st_size > limits.max_archive_bytes:
                    raise ExcelResourceLimitError(
                        "XLSX compressed archive limit exceeded",
                        limit=limits.max_archive_bytes,
                        observed=info.st_size,
                    )
                snapshot = magic + stream.read(limits.max_archive_bytes + 1 - len(magic))
                if len(snapshot) > limits.max_archive_bytes:
                    raise ExcelResourceLimitError(
                        "XLSX compressed archive limit exceeded",
                        limit=limits.max_archive_bytes,
                        observed=len(snapshot),
                    )
                snapshot = _validate_archive(snapshot, limits, profile=profile)
    except ExcelResourceLimitError as exc:
        exc.file_path = file_path
        raise
    modified = datetime.fromtimestamp(info.st_mtime).strftime("%Y-%m-%dT%H:%M:%S%z")
    return snapshot, modified


def profile_workbook(
    file_path: str | Path, *, limits: ExcelReadLimits = ExcelReadLimits()
) -> dict[str, Any]:
    """Return workbook structure and resource-validation metadata under ingestion budgets.

    Uses ingestion's validation rules and rejection errors. The profile contains metadata
    rather than parsed Office documents.
    """
    profile: dict[str, Any] = {
        "file": str(file_path),
        "accepted": True,
        "rejection_policy": "abort_index_build",
        "limits": asdict(limits),
    }
    _read_snapshot(str(file_path), limits, profile=profile)
    return profile
