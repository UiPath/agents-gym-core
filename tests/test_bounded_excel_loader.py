"""Synthetic ZIP/OPC/XML/grid and workbook document-contract regressions."""

import io
import json
import os
import posixpath
import struct
import zlib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest
from defusedxml.common import DefusedXmlException
from openpyxl import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from agents_gym_core.common_tools.vector_database.bounded_excel_loader import (
    ExcelReadLimits,
    ExcelResourceLimitError,
    _inflate,
    _read_snapshot,
    _validate_archive,
    profile_workbook,
)


def validate_workbook(path: str | Path, *, limits: ExcelReadLimits = ExcelReadLimits()) -> bytes:
    return _read_snapshot(str(path), limits)[0]


@pytest.fixture
def workbook_path(tmp_path: Path) -> Path:
    path = tmp_path / "work_orders.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    assert isinstance(sheet, Worksheet)
    sheet.title = "Work orders"
    sheet.append(["Work order", "Status"])
    sheet.append(["WO-1001", "Open"])
    sheet.append([None, None])
    sheet.append(["WO-1002", "Closed"])
    second = workbook.create_sheet("Assignments")
    second.append(["Group", "Assigned to"])
    second.append(["Service desk", "Example operator"])
    workbook.create_sheet("Empty")
    workbook.save(path)
    workbook.close()
    return path


@pytest.mark.parametrize("payload", [b"", b"not a zip", b"PK\x03\x04broken"])
def test_corrupt_files_fail_closed(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "corrupt.xlsx"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="Corrupt"):
        validate_workbook(path)


def _rewrite(path: Path, name: str, value: bytes) -> None:
    with ZipFile(path) as source:
        members = {item: source.read(item) for item in source.namelist()}
    members[name] = value
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as target:
        for item, content in members.items():
            target.writestr(item, content)


@pytest.mark.parametrize("suffix", [".xlsx", ".xls"])
@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_archive_bytes", 32, "compressed archive"),
        ("max_members", 2, "member limit"),
        ("max_member_bytes", 128, "expanded member"),
        ("max_expanded_bytes", 2048, "expanded archive"),
        ("max_snapshot_bytes", 128, "snapshot limit"),
        ("max_name_bytes", 8, "name limit"),
        ("max_sheets", 1, "sheet limit"),
        ("max_rows", 2, "row/column"),
        ("max_columns", 1, "row/column"),
        ("max_grid_cells", 8, "dense grid"),
        ("max_xml_depth", 1, "XML complexity"),
        ("max_xml_nodes", 2, "XML complexity"),
        ("max_cell_text_bytes", 128, "projected cell text"),
    ],
)
def test_configured_limits_apply_before_parsing(
    workbook_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
    message: str,
    suffix: str,
) -> None:
    workbook_path = workbook_path.rename(workbook_path.with_suffix(suffix))
    configured = {"max_rows": 60_000, field: value}
    limits = ExcelReadLimits(**configured)
    with pytest.raises(ExcelResourceLimitError, match=message) as rejected:
        validate_workbook(workbook_path, limits=limits)
    error = rejected.value
    assert error.file_path == str(workbook_path)
    assert error.limit == value
    assert error.observed > value
    assert f"file={str(workbook_path)!r}" in str(error)
    assert f"part={error.part or '<archive>'!r}" in str(error)
    assert f"limit={value}, observed={error.observed}" in str(error)
    with pytest.raises(ExcelResourceLimitError) as profile_rejected:
        profile_workbook(workbook_path, limits=limits)
    assert str(profile_rejected.value) == str(error)

    from agents_gym_core.common_tools.vector_database import core_vector_database_tool as module

    earlier = workbook_path.parent / "earlier.txt"
    earlier.write_text("Synthetic earlier document")
    index = workbook_path.parent / "new-index"
    parser = Mock(side_effect=AssertionError("Rejected workbook must not enter pandas"))
    create = Mock()
    monkeypatch.setattr(module, "read_excel", parser)
    monkeypatch.setattr(module.Chroma, "from_documents", create)
    tool = module.CoreVectorDatabaseTool.model_validate(
        {
            "file_paths": [str(earlier), str(workbook_path)],
            "vector_store_type": "chroma",
            "persist_directory": str(index),
            "embeddings": object(),
            "search_strategy": "hybrid",
            "excel_limits": configured,
        }
    )
    with pytest.raises(ExcelResourceLimitError) as ingestion_rejected:
        tool.initialize_vector_store()
    assert str(ingestion_rejected.value) == str(error)
    parser.assert_not_called()
    create.assert_not_called()
    assert not index.exists()
    assert str(index) not in tool._store_cache
    assert str(index) not in tool._bm25_cache


def test_member_rejection_identifies_the_file_part_limit_and_declared_size(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    part = "xl/worksheets/sheet1.xml"
    content = b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"/>'
    content += b" " * 40_000
    _rewrite(workbook_path, part, content)
    with pytest.raises(ExcelResourceLimitError, match="expanded member") as rejected:
        validate_workbook(workbook_path, limits=replace(ExcelReadLimits(), max_member_bytes=40_000))
    assert rejected.value.file_path == str(workbook_path)
    assert rejected.value.part == part
    assert rejected.value.limit == 40_000
    assert rejected.value.observed == len(content)
    assert f"part={part!r}" in str(rejected.value)


@pytest.mark.parametrize("name", ["xl/sharedStrings.xml", "xl/styles.xml", "_rels/.rels"])
@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_all_xml_parts_reject_dtd_before_office_parser(
    workbook_path: Path, name: str, encoding: str
) -> None:
    xml = '<?xml version="1.0" encoding="ENC"?><!DOCTYPE x [<!ENTITY a "boom">]><x>&a;</x>'
    _rewrite(workbook_path, name, xml.replace("ENC", encoding).encode(encoding))
    with pytest.raises(DefusedXmlException):
        validate_workbook(workbook_path)


@pytest.mark.parametrize("reference", ["A1:XFD1048576", "bad", "A0"])
def test_dimension_is_validated(workbook_path: Path, reference: str) -> None:
    _rewrite(
        workbook_path,
        "xl/worksheets/sheet1.xml",
        f'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<dimension ref="{reference}"/></worksheet>'.encode(),
    )
    with pytest.raises(ValueError):
        validate_workbook(workbook_path)


def test_sparse_coordinates_cannot_bypass_small_declared_dimension(workbook_path: Path) -> None:
    _rewrite(
        workbook_path,
        "xl/worksheets/sheet1.xml",
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b'<dimension ref="A1"/><sheetData><row r="50001"><c r="A50001"/>'
        b"</row></sheetData></worksheet>",
    )
    with pytest.raises(ExcelResourceLimitError):
        validate_workbook(workbook_path)


def test_duplicate_members_fail_closed(workbook_path: Path) -> None:
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(workbook_path, "a") as archive:
            archive.writestr("xl/workbook.xml", "<workbook/>")
    with pytest.raises(ValueError, match="Duplicate"):
        validate_workbook(workbook_path)


def test_shared_strings_expansion_is_budgeted(workbook_path: Path) -> None:
    _rewrite(workbook_path, "xl/sharedStrings.xml", b"<sst>" + b"x" * 200_000 + b"</sst>")
    assert workbook_path.stat().st_size < 20_000
    with pytest.raises(ExcelResourceLimitError, match="expanded member"):
        validate_workbook(
            workbook_path, limits=replace(ExcelReadLimits(), max_member_bytes=100_000)
        )


def test_deep_xml_is_rejected_before_the_whole_tree_is_allocated(workbook_path: Path) -> None:
    _rewrite(workbook_path, "xl/sharedStrings.xml", b"<x>" * 1000 + b"</x>" * 1000)
    with pytest.raises(ExcelResourceLimitError, match="XML complexity"):
        validate_workbook(workbook_path)


def _zip_offsets(data: bytes | bytearray, name: str) -> tuple[int, int]:
    with ZipFile(io.BytesIO(data)) as archive:
        local = archive.getinfo(name).header_offset
        cursor = archive.start_dir
    while data[cursor : cursor + 4] == b"PK\x01\x02":
        name_len, extra_len, comment_len = struct.unpack_from("<HHH", data, cursor + 28)
        if data[cursor + 46 : cursor + 46 + name_len].decode() == name:
            return local, cursor
        cursor += 46 + name_len + extra_len + comment_len
    raise AssertionError("test member missing")


def _small_zip(content: bytes, *, compression: int = ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=compression) as archive:
        archive.writestr("xl/workbook.xml", content)
    return buffer.getvalue()


def _claim_size(data: bytes, name: str, size: int, crc: int) -> bytes:
    local, central = _zip_offsets(data, name)
    changed = bytearray(data)
    struct.pack_into("<I", changed, local + 22, size)
    struct.pack_into("<I", changed, central + 24, size)
    struct.pack_into("<I", changed, local + 14, crc)
    struct.pack_into("<I", changed, central + 16, crc)
    return bytes(changed)


@pytest.mark.parametrize(
    ("offset", "format", "value"),
    [(6, "H", 8), (8, "H", ZIP_STORED), (14, "I", 0), (18, "I", 1), (22, "I", 1)],
)
def test_local_central_metadata_mismatch_fails_before_parser(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, offset: int, format: str, value: int
) -> None:

    changed = bytearray(workbook_path.read_bytes())
    local, _ = _zip_offsets(changed, "xl/workbook.xml")
    struct.pack_into("<" + format, changed, local + offset, value)
    workbook_path.write_bytes(changed)
    with pytest.raises(ValueError, match="mismatch"):
        validate_workbook(workbook_path)


@pytest.mark.parametrize("compression", [ZIP_STORED, ZIP_DEFLATED])
def test_understated_size_and_prefix_crc_do_not_validate_only_a_prefix(compression: int) -> None:
    content = b"<workbook/>" + b" " * 64
    original = _small_zip(content, compression=compression)
    claimed = len(b"<workbook/>")
    changed = _claim_size(original, "xl/workbook.xml", claimed, zlib.crc32(content[:claimed]))
    with pytest.raises(ValueError, match="actual size or CRC"):
        _validate_archive(changed, ExcelReadLimits())


def test_actual_emitted_output_limit_is_not_the_claimed_zip_size() -> None:
    content = b"x" * 256
    changed = _claim_size(_small_zip(content), "xl/workbook.xml", 1, zlib.crc32(content[:1]))
    with pytest.raises(ExcelResourceLimitError, match="actual expanded") as rejected:
        _validate_archive(changed, replace(ExcelReadLimits(), max_member_bytes=64))
    assert rejected.value.part == "xl/workbook.xml"
    assert rejected.value.limit == 64
    assert rejected.value.observed == 65


def test_actual_aggregate_output_budget_includes_previously_verified_parts() -> None:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("first.bin", b"a" * 300)
        archive.writestr("second.bin", b"b" * 300)
    changed = _claim_size(buffer.getvalue(), "second.bin", 1, zlib.crc32(b"b"))
    with pytest.raises(ExcelResourceLimitError, match="actual expanded") as rejected:
        _validate_archive(changed, replace(ExcelReadLimits(), max_expanded_bytes=400))
    assert rejected.value.part == "second.bin"
    assert rejected.value.limit == 400
    assert rejected.value.observed == 401


def test_truncated_deflate_requires_actual_end_of_stream() -> None:
    data = bytearray(_small_zip(b"<workbook/>" * 10))
    local, central = _zip_offsets(data, "xl/workbook.xml")
    size = struct.unpack_from("<I", data, local + 18)[0] - 1
    struct.pack_into("<I", data, local + 18, size)
    struct.pack_into("<I", data, central + 20, size)
    with pytest.raises(ValueError, match="Truncated XLSX DEFLATE"):
        _validate_archive(bytes(data), ExcelReadLimits())


def test_trailing_deflate_bytes_are_not_silently_ignored() -> None:
    original = _small_zip(b"<workbook/>")
    local, central = _zip_offsets(original, "xl/workbook.xml")
    data = bytearray(original[:central] + b"\x00" + original[central:])
    compressed = struct.unpack_from("<I", data, local + 18)[0] + 1
    struct.pack_into("<I", data, local + 18, compressed)
    struct.pack_into("<I", data, central + 1 + 20, compressed)
    end = data.rfind(b"PK\x05\x06")
    struct.pack_into("<I", data, end + 16, central + 1)
    with pytest.raises(ValueError, match="Trailing XLSX DEFLATE"):
        _validate_archive(bytes(data), ExcelReadLimits())


def test_crc_is_checked_against_all_actual_output_bytes() -> None:
    data = _small_zip(b"<workbook/>")
    changed = _claim_size(data, "xl/workbook.xml", len(b"<workbook/>"), 0)
    with pytest.raises(ValueError, match="actual size or CRC"):
        _validate_archive(changed, ExcelReadLimits())


def test_directory_count_is_crosschecked_before_parsing_members(workbook_path: Path) -> None:
    changed = bytearray(workbook_path.read_bytes())
    end = changed.rfind(b"PK\x05\x06")
    count = struct.unpack_from("<H", changed, end + 10)[0] + 1
    struct.pack_into("<HH", changed, end + 8, count, count)
    with pytest.raises(ValueError, match="count mismatch"):
        _validate_archive(bytes(changed), ExcelReadLimits())


@pytest.mark.parametrize("force_zip64", [False, True])
def test_normal_streaming_descriptors_and_local_zip64_remain_supported(
    workbook_path: Path, force_zip64: bool
) -> None:
    class StreamingBuffer(io.BytesIO):
        def seek(self, *_args: object, **_kwargs: object) -> int:
            raise OSError("nonseekable test writer")

    buffer = StreamingBuffer()
    with ZipFile(workbook_path) as source, ZipFile(buffer, "w", compression=ZIP_DEFLATED) as target:
        for name in source.namelist():
            with target.open(name, "w", force_zip64=force_zip64) as member:
                member.write(source.read(name))
    workbook_path.write_bytes(buffer.getvalue())
    canonical = validate_workbook(workbook_path)
    assert isinstance(canonical, bytes)
    with ZipFile(io.BytesIO(canonical)) as archive:
        assert archive.testzip() is None


def test_canonical_snapshot_drops_original_metadata_and_has_an_explicit_output_cap(
    workbook_path: Path,
) -> None:
    with ZipFile(workbook_path, "a") as archive:
        archive.comment = b"synthetic archive comment"
    canonical, _ = _read_snapshot(str(workbook_path), ExcelReadLimits())
    with ZipFile(io.BytesIO(canonical)) as archive:
        assert not archive.comment
        assert all(member.compress_type == ZIP_STORED for member in archive.infolist())
        assert all(member.date_time == (1980, 1, 1, 0, 0, 0) for member in archive.infolist())
    assert len(canonical) <= ExcelReadLimits().max_snapshot_bytes
    with pytest.raises(ExcelResourceLimitError, match="snapshot limit"):
        _read_snapshot(
            str(workbook_path), replace(ExcelReadLimits(), max_snapshot_bytes=len(canonical) - 1)
        )


@pytest.mark.parametrize("field", ["max_rows", "max_columns", "max_grid_cells"])
def test_implicit_coordinates_cannot_exceed_the_inferred_grid_budget(
    workbook_path: Path, field: str
) -> None:
    xml = (
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<sheetData><row><c/><c/></row><row><c/><c/></row></sheetData></worksheet>"
    )
    _rewrite(workbook_path, "xl/worksheets/sheet1.xml", xml)
    with pytest.raises(ExcelResourceLimitError):
        validate_workbook(workbook_path, limits=replace(ExcelReadLimits(), **{field: 1}))


@pytest.mark.parametrize(("row_count", "max_rows"), [(3, 2), (50_001, 50_000)])
def test_implicit_rows_exceed_row_limit_without_exceeding_grid_budget(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, row_count: int, max_rows: int
) -> None:

    part = "xl/worksheets/sheet1.xml"
    xml = (
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<sheetData>" + b"<row><c><v>1</v></c></row>" * row_count + b"</sheetData></worksheet>"
    )
    _rewrite(workbook_path, part, xml)
    limits = replace(ExcelReadLimits(), max_rows=max_rows)
    assert row_count < limits.max_grid_cells
    with pytest.raises(ExcelResourceLimitError, match="row limit") as rejected:
        validate_workbook(workbook_path, limits=limits)
    assert rejected.value.part == part
    assert rejected.value.limit == max_rows
    assert rejected.value.observed == max_rows + 1


def _rename_part(path: Path, old: str, new: str, *, content_type: str | None = None) -> None:
    with ZipFile(path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    for name, content in list(parts.items()):
        if not name.endswith((".xml", ".rels")):
            continue
        tree = ElementTree.fromstring(content)
        for element in tree.iter():
            if element.get("PartName") == "/" + old:
                element.set("PartName", "/" + new)
                if content_type is not None:
                    element.set("ContentType", content_type)
            target = element.get("Target")
            if target is not None:
                base = (
                    "" if name == "_rels/.rels" else posixpath.dirname(name).removesuffix("/_rels")
                )
                resolved = (
                    target[1:]
                    if target.startswith("/")
                    else posixpath.normpath(posixpath.join(base, target))
                )
                if resolved == old:
                    element.set(
                        "Target",
                        "/" + new
                        if target.startswith("/")
                        else posixpath.relpath(new, base or "."),
                    )
        parts[name] = ElementTree.tostring(tree)
    parts[new] = parts.pop(old)
    old_rels = posixpath.join(posixpath.dirname(old), "_rels", posixpath.basename(old) + ".rels")
    if old_rels in parts:
        new_rels = posixpath.join(
            posixpath.dirname(new), "_rels", posixpath.basename(new) + ".rels"
        )
        parts[new_rels] = parts.pop(old_rels)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)


@pytest.mark.parametrize("declared", [None, "application/octet-stream"])
@pytest.mark.parametrize("failure", ["dtd", "depth", "grid"])
def test_opc_aliases_cannot_bypass_xml_guards_before_the_office_parser(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, declared: str | None, failure: str
) -> None:

    _rename_part(
        workbook_path, "xl/worksheets/sheet1.xml", "xl/worksheets/data.bin", content_type=declared
    )
    if failure == "dtd":
        xml = b'<!DOCTYPE x [<!ENTITY q "synthetic">]><x>&q;</x>'
        expected = DefusedXmlException
    elif failure == "depth":
        xml = b"<x>" * 130 + b"</x>" * 130
        expected = ExcelResourceLimitError
    else:
        xml = b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><dimension ref="XFD1048576"/></worksheet>'
        expected = ExcelResourceLimitError
    _rewrite(workbook_path, "xl/worksheets/data.bin", xml)
    with pytest.raises(expected):
        validate_workbook(workbook_path)


def _add_shared_string_alias(path: Path, content: bytes) -> None:
    with ZipFile(path) as archive:
        manifest = ElementTree.fromstring(archive.read("[Content_Types].xml"))
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    ElementTree.SubElement(
        manifest,
        "{http://schemas.openxmlformats.org/package/2006/content-types}Override",
        PartName="/xl/strings.bin",
        ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml",
    )
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    cell = next(sheet.iter(namespace + "c"))
    for child in list(cell):
        cell.remove(child)
    cell.set("t", "s")
    ElementTree.SubElement(cell, namespace + "v").text = "0"
    _rewrite(path, "[Content_Types].xml", ElementTree.tostring(manifest))
    _rewrite(path, "xl/worksheets/sheet1.xml", ElementTree.tostring(sheet))
    _rewrite(path, "xl/strings.bin", content)


def _repeat_shared_reference(path: Path, text: bytes, count: int) -> None:
    _add_shared_string_alias(
        path,
        b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<si><t>" + text + b"</t></si></sst>",
    )
    _rewrite(
        path,
        "xl/worksheets/sheet1.xml",
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<sheetData>" + b'<row><c t="s"><v>0</v></c></row>' * count + b"</sheetData></worksheet>",
    )


def test_repeated_shared_string_amplification_never_enters_office_parser(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    _repeat_shared_reference(workbook_path, b"x" * (1024 * 1024), 65)
    assert workbook_path.stat().st_size < 10_000
    for validate in (profile_workbook, lambda path: validate_workbook(path)):
        with pytest.raises(ExcelResourceLimitError, match="projected cell text") as rejected:
            validate(workbook_path)
        assert rejected.value.file_path == str(workbook_path)
        assert rejected.value.part == "xl/worksheets/sheet1.xml"
        assert rejected.value.limit == 64 * 1024 * 1024
        assert rejected.value.observed > rejected.value.limit


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize(
    "kind",
    [
        "shared",
        "shared_rich",
        "inline",
        "inline_rich",
        "cached",
        "numeric",
        "escaped",
        "unicode",
        "foreign_cell",
        "padding",
    ],
)
def test_projected_text_budget_covers_all_cell_views_before_office(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, alias: bool
) -> None:

    text = b"x" * 4096
    count = 8
    if kind == "escaped":
        text = b"&amp;" * 1000
    elif kind == "unicode":
        text, count = "\u5b57".encode() * 1000, 12
    _repeat_shared_reference(workbook_path, text, count)
    if kind == "shared_rich":
        _rewrite(
            workbook_path,
            "xl/strings.bin",
            b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<si><r><t>" + text[:2048] + b"</t></r><r><t>" + text[2048:] + b"</t></r></si></sst>",
        )
    if kind in {"inline", "inline_rich", "cached", "numeric", "foreign_cell", "padding"}:
        if kind == "inline":
            cell = b'<c t="inlineStr"><is><t>' + text + b"</t></is></c>"
        elif kind == "inline_rich":
            cell = (
                b'<c t="inlineStr"><is><r><t>'
                + text[:2048]
                + b"</t></r><r><t>"
                + text[2048:]
                + b"</t></r></is></c>"
            )
        elif kind == "cached":
            cell = b'<c t="str"><f>CONCAT()</f><v>' + text + b"</v></c>"
        elif kind == "numeric":
            # 640 lexical bytes. Only the 309-digit integer rendering exceeds the budget.
            cell, count = b'<c t="n"><f>1</f><v>1e308</v></c>', 128
        elif kind == "foreign_cell":
            cell = b'<x:cell xmlns:x="urn:synthetic" t="s"><v>0</v></x:cell>'
        else:
            cell = b'<c t="s"><v>0</v></c>'
        rows = b"<row>" + cell + b"</row>"
        if kind == "padding":
            count = 12
            rows += b'<row><c t="inlineStr"><is><t>y</t></is></c></row>' * (count - 1)
        else:
            rows *= count
        _rewrite(
            workbook_path,
            "xl/worksheets/sheet1.xml",
            b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<sheetData>" + rows + b"</sheetData></worksheet>",
        )
    part = "xl/worksheets/sheet1.xml"
    if alias:
        _rename_part(workbook_path, part, "xl/worksheets/data.bin")
        part = "xl/worksheets/data.bin"
    limits = replace(ExcelReadLimits(), max_cell_text_bytes=40_000)
    for validate in (
        lambda path: profile_workbook(path, limits=limits),
        lambda path: validate_workbook(path, limits=limits),
    ):
        with pytest.raises(ExcelResourceLimitError, match="projected cell text") as rejected:
            validate(workbook_path)
        assert rejected.value.part == part
        assert rejected.value.limit == 40_000
        assert rejected.value.observed > 40_000


def _sheet(rows: list[bytes]) -> bytes:
    return (
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<sheetData>" + b"".join(rows) + b"</sheetData></worksheet>"
    )


@pytest.mark.parametrize("shared", [False, True])
def test_digit_strings_are_charged_a_float_repr_for_pandas_numeric_inference(
    workbook_path: Path, shared: bool
) -> None:
    """pandas turns a column of "1" strings into floats ("1.0"), so a digit costs 24 bytes."""
    original = workbook_path.read_bytes()

    def projected(text: bytes) -> int:
        workbook_path.write_bytes(original)
        if shared:
            _repeat_shared_reference(workbook_path, text, 1_000)
        else:
            cell = b'<row><c t="inlineStr"><is><t>' + text + b"</t></is></c></row>"
            _rewrite(workbook_path, "xl/worksheets/sheet1.xml", _sheet([cell] * 1_000))
        parts = profile_workbook(workbook_path)["xml_parts"]
        part = "xl/worksheets/sheet1.xml"
        return next(item for item in parts if item["part"] == part)["projected_cell_text_bytes"]

    assert projected(b"1") - projected(b"y") >= 1_000 * (24 - 1)


def test_empty_grid_cells_are_charged_html_markup(workbook_path: Path) -> None:
    """Two populated corners make 4,000 dense cells. Only markup exceeds the budget."""
    _rewrite(
        workbook_path,
        "xl/worksheets/sheet1.xml",
        _sheet(
            [
                b'<row r="1"><c r="A1" t="inlineStr"><is><t>a</t></is></c></row>',
                b'<row r="400"><c r="J400" t="inlineStr"><is><t>b</t></is></c></row>',
            ]
        ),
    )
    limits = replace(ExcelReadLimits(), max_cell_text_bytes=200_000)
    with pytest.raises(ExcelResourceLimitError, match="projected cell text") as rejected:
        profile_workbook(workbook_path, limits=limits)
    assert rejected.value.part == "xl/worksheets/sheet1.xml"
    assert rejected.value.observed >= 400 * 10 * 64


def test_text_budget_stops_at_the_first_cell_past_the_limit(workbook_path: Path) -> None:
    """The parse stops at cell 101 of 1,000. A package-level check would report all of them."""
    cell = b'<row><c t="inlineStr"><is><t>' + b"y" * 100 + b"</t></is></c></row>"
    _rewrite(workbook_path, "xl/worksheets/sheet1.xml", _sheet([cell] * 1_000))
    limits = replace(ExcelReadLimits(), max_cell_text_bytes=10_000)
    with pytest.raises(ExcelResourceLimitError, match="projected cell text") as rejected:
        profile_workbook(workbook_path, limits=limits)
    assert rejected.value.part == "xl/worksheets/sheet1.xml"
    assert rejected.value.observed == 10_100


@pytest.mark.parametrize(
    "value",
    ["-1", "+0", "1", "", " 0", "0 ", "1e0", "\u0660", "0" * 11, "missing", "duplicate", "nested"],
)
def test_shared_string_references_are_unambiguous_in_range_and_well_formed(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:

    _repeat_shared_reference(workbook_path, b"safe", 1)
    content = f"<v>{value}</v>".encode()
    if value == "missing":
        content = b""
    elif value == "duplicate":
        content = b"<v>0</v><v>0</v>"
    elif value == "nested":
        content = b"<v>0<x/></v>"
    _rewrite(
        workbook_path,
        "xl/worksheets/sheet1.xml",
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b'<sheetData><row><c t="s">' + content + b"</c></row></sheetData></worksheet>",
    )
    with pytest.raises(ValueError, match="shared-string reference"):
        validate_workbook(workbook_path)


@pytest.mark.parametrize(
    "kind", ["duplicate_table", "default_type", "wrong_root", "nested_si", "missing_table"]
)
def test_ambiguous_or_malformed_shared_string_tables_fail_closed(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:

    _repeat_shared_reference(workbook_path, b"safe", 1)
    with ZipFile(workbook_path) as archive:
        manifest = ElementTree.fromstring(archive.read("[Content_Types].xml"))
    namespace = "{http://schemas.openxmlformats.org/package/2006/content-types}"
    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"
    if kind == "duplicate_table":
        # A second table that is valid on its own, so only the ambiguity check can reject it.
        _rewrite(
            workbook_path,
            "xl/otherStrings.xml",
            b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<si><t>safe</t></si></sst>",
        )
        ElementTree.SubElement(
            manifest, namespace + "Override", PartName="/xl/otherStrings.xml", ContentType=mime
        )
    elif kind in {"default_type", "missing_table"}:
        for child in list(manifest):
            if child.get("PartName") == "/xl/strings.bin":
                child.set("ContentType", "application/xml")
                if kind == "default_type":
                    manifest.remove(child)
        if kind == "default_type":
            ElementTree.SubElement(
                manifest, namespace + "Default", Extension="bin", ContentType=mime
            )
    else:
        root = "x" if kind == "wrong_root" else "sst"
        body = b"<si><si><t>safe</t></si></si>" if kind == "nested_si" else b"<si><t>safe</t></si>"
        _rewrite(
            workbook_path,
            "xl/strings.bin",
            f'<{root} xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'.encode()
            + body
            + f"</{root}>".encode(),
        )
    _rewrite(workbook_path, "[Content_Types].xml", ElementTree.tostring(manifest))
    ambiguous = kind in {"duplicate_table", "default_type"}
    match = "Ambiguous XLSX shared-string content type" if ambiguous else "shared-string"
    with pytest.raises(ValueError, match=match):
        validate_workbook(workbook_path)


@pytest.mark.parametrize("small_table", ["xl/a.xml", "xl/z.xml"])
def test_two_valid_shared_string_tables_are_rejected_whatever_their_order(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, small_table: str
) -> None:
    """openpyxl reads the first override. Charging the other would undercount its text.

    The small table sorts before and after the 1 MiB table, so the result never
    depends on which table the gate happens to visit first.
    """

    _repeat_shared_reference(workbook_path, b"x" * (1024 * 1024), 200)
    with ZipFile(workbook_path) as archive:
        manifest = ElementTree.fromstring(archive.read("[Content_Types].xml"))
    _rewrite(
        workbook_path,
        small_table,
        b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b"<si><t>x</t></si></sst>",
    )
    ElementTree.SubElement(
        manifest,
        "{http://schemas.openxmlformats.org/package/2006/content-types}Override",
        PartName="/" + small_table,
        ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml",
    )
    _rewrite(workbook_path, "[Content_Types].xml", ElementTree.tostring(manifest))
    for validate in (profile_workbook, lambda path: validate_workbook(path)):
        with pytest.raises(ValueError, match="Ambiguous XLSX shared-string content type"):
            validate(workbook_path)


def test_actual_sst_content_type_wins_over_a_conventional_name_and_relationship_hint(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    _repeat_shared_reference(workbook_path, b"x" * 4096, 8)
    _rewrite(
        workbook_path,
        "xl/sharedStrings.xml",
        b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><si><t>x</t></si></sst>',
    )
    with ZipFile(workbook_path) as archive:
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    ElementTree.SubElement(
        relationships,
        "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship",
        Id="fake-sst",
        Target="sharedStrings.xml",
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings",
    )
    _rewrite(workbook_path, "xl/_rels/workbook.xml.rels", ElementTree.tostring(relationships))
    with pytest.raises(ExcelResourceLimitError, match="projected cell text"):
        validate_workbook(
            workbook_path, limits=replace(ExcelReadLimits(), max_cell_text_bytes=40_000)
        )


def test_projected_text_counts_reused_aliased_worksheets_for_each_materialization(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:

    _repeat_shared_reference(workbook_path, b"x" * 4096, 4)
    _rename_part(workbook_path, "xl/worksheets/sheet1.xml", "xl/worksheets/data.bin")
    with ZipFile(workbook_path) as archive:
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    sheets = [child for child in relationships if child.get("Type", "").endswith("/worksheet")]
    sheets[1].set("Target", sheets[0].get("Target", ""))
    _rewrite(workbook_path, "xl/_rels/workbook.xml.rels", ElementTree.tostring(relationships))
    profile = profile_workbook(workbook_path)
    sheet = next(part for part in profile["xml_parts"] if part["part"] == "xl/worksheets/data.bin")
    assert sheet["sheet_materializations"] == 2
    assert (
        sheet["projected_cell_text_bytes"]
        < 40_000
        < profile["package"]["projected_cell_text_bytes"]
    )
    with pytest.raises(ExcelResourceLimitError, match="projected cell text"):
        validate_workbook(
            workbook_path, limits=replace(ExcelReadLimits(), max_cell_text_bytes=40_000)
        )


def test_shared_strings_alias_is_validated_even_without_xml_filename(workbook_path: Path) -> None:
    _add_shared_string_alias(workbook_path, b'<!DOCTYPE x [<!ENTITY q "synthetic">]><x>&q;</x>')
    with pytest.raises(DefusedXmlException):
        validate_workbook(workbook_path)


def test_workbook_and_its_relationship_part_can_use_an_opc_alias(workbook_path: Path) -> None:
    _rename_part(workbook_path, "xl/workbook.xml", "xl/book.bin")
    snapshot, _ = _read_snapshot(str(workbook_path), ExcelReadLimits())
    with ZipFile(io.BytesIO(snapshot)) as archive:
        assert "xl/book.bin" in archive.namelist()
        assert "xl/_rels/book.bin.rels" in archive.namelist()


@pytest.mark.parametrize(
    "target",
    ["missing.xml", "../../outside.xml", "sheet%31.xml", "https://example.invalid/file.xml"],
)
def test_ambiguous_or_missing_internal_relationships_are_rejected(
    workbook_path: Path, target: str
) -> None:
    with ZipFile(workbook_path) as archive:
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    next(iter(relationships)).set("Target", target)
    _rewrite(workbook_path, "xl/_rels/workbook.xml.rels", ElementTree.tostring(relationships))
    with pytest.raises(ValueError, match="relationship|part name"):
        validate_workbook(workbook_path)


def test_canonical_snapshot_drops_unknown_extra_fields(workbook_path: Path) -> None:
    with ZipFile(workbook_path) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    with ZipFile(workbook_path, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            info = ZipInfo(name)
            info.compress_type = ZIP_DEFLATED
            info.extra = struct.pack("<HH", 0xCAFE, 4) + b"test"
            archive.writestr(info, content)
    snapshot, _ = _read_snapshot(str(workbook_path), ExcelReadLimits())
    with ZipFile(io.BytesIO(snapshot)) as canonical:
        assert all(not member.extra for member in canonical.infolist())


@pytest.mark.parametrize("role", ["image", "unrecognized-chartsheet"])
def test_workbook_sheet_ids_cannot_reclassify_binary_relationships_as_unchecked_xml(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, role: str
) -> None:

    _rename_part(
        workbook_path,
        "xl/worksheets/sheet1.xml",
        "xl/worksheets/data.bin",
        content_type="application/octet-stream",
    )
    with ZipFile(workbook_path) as archive:
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    next(iter(relationships)).set(
        "Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/" + role
    )
    _rewrite(workbook_path, "xl/_rels/workbook.xml.rels", ElementTree.tostring(relationships))
    with pytest.raises(ValueError, match="inconsistent XML role"):
        validate_workbook(workbook_path)


@pytest.mark.parametrize("part", ["manifest", "relationships", "workbook"])
def test_namespace_aliases_cannot_create_different_validator_and_parser_views(
    workbook_path: Path, part: str
) -> None:
    names = {
        "manifest": "[Content_Types].xml",
        "relationships": "xl/_rels/workbook.xml.rels",
        "workbook": "xl/workbook.xml",
    }
    with ZipFile(workbook_path) as archive:
        tree = ElementTree.fromstring(archive.read(names[part]))
    if part == "workbook":
        element = next(
            tree.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}sheet")
        )
    else:
        element = next(iter(tree))
    element.tag = "{urn:synthetic-foreign}" + element.tag.rsplit("}", 1)[-1]
    _rewrite(workbook_path, names[part], ElementTree.tostring(tree))
    with pytest.raises(ValueError, match="namespace"):
        validate_workbook(workbook_path)


def test_every_direct_row_child_uses_the_office_parsers_coordinate_budget(
    workbook_path: Path,
) -> None:
    xml = (
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:x="urn:synthetic">'
        b'<dimension ref="A1"/><sheetData><row r="1"><x:c r="XFD1"><v>1</v></x:c></row></sheetData></worksheet>'
    )
    _rewrite(workbook_path, "xl/worksheets/sheet1.xml", xml)
    with pytest.raises(ExcelResourceLimitError, match="row/column"):
        validate_workbook(workbook_path)


def test_reused_sheet_parts_are_counted_for_each_parser_materialization(
    workbook_path: Path,
) -> None:
    with ZipFile(workbook_path) as archive:
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    sheets = [
        element for element in relationships if element.get("Type", "").endswith("/worksheet")
    ]
    sheets[1].set("Target", sheets[0].get("Target", ""))
    _rewrite(workbook_path, "xl/_rels/workbook.xml.rels", ElementTree.tostring(relationships))
    with pytest.raises(ExcelResourceLimitError, match="dense grid"):
        validate_workbook(workbook_path, limits=replace(ExcelReadLimits(), max_grid_cells=15))


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX FIFO behavior")
def test_non_regular_file_rejected_without_blocking(tmp_path: Path) -> None:
    path = tmp_path / "fifo.xlsx"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        _read_snapshot(str(path), ExcelReadLimits())
    with pytest.raises(ValueError, match="regular file"):
        profile_workbook(path)


@pytest.mark.parametrize("field", [item.name for item in fields(ExcelReadLimits)])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1", None, float("inf"), float("nan")])
def test_non_positive_or_non_integer_limits_are_invalid(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match="positive"):
        ExcelReadLimits(**{field: value})


@pytest.mark.parametrize("field", [item.name for item in fields(ExcelReadLimits)])
def test_exact_budget_boundary_accepts_and_one_less_rejects(
    workbook_path: Path, field: str
) -> None:
    report = profile_workbook(workbook_path)
    archive, package, xml = report["archive"], report["package"], report["xml_parts"]
    boundaries = {
        "max_archive_bytes": archive["input_bytes"],
        "max_members": archive["members"],
        "max_member_bytes": max(part["expanded_bytes"] for part in archive["parts"]),
        "max_expanded_bytes": archive["expanded_bytes"],
        "max_snapshot_bytes": archive["canonical_bytes"],
        "max_name_bytes": max(len(part["part"].encode()) for part in archive["parts"]),
        "max_sheets": package["sheets"],
        "max_rows": max(part["rows"] for part in xml),
        "max_columns": max(part["columns"] for part in xml),
        "max_grid_cells": package["grid_cells"],
        "max_xml_depth": max(part["depth"] for part in xml),
        "max_xml_nodes": max(part["nodes"] for part in xml),
        "max_cell_text_bytes": package["projected_cell_text_bytes"],
    }
    boundary = boundaries[field]
    assert validate_workbook(workbook_path, limits=replace(ExcelReadLimits(), **{field: boundary}))
    with pytest.raises(ExcelResourceLimitError):
        validate_workbook(workbook_path, limits=replace(ExcelReadLimits(), **{field: boundary - 1}))


@pytest.mark.parametrize("mode", ["Internal", "External"])
@pytest.mark.parametrize("position", ["before", "after"])
def test_duplicate_relationship_ids_cannot_change_the_parser_binding(
    workbook_path: Path, mode: str, position: str
) -> None:
    name = "xl/_rels/workbook.xml.rels"
    with ZipFile(workbook_path) as archive:
        relationships = ElementTree.fromstring(archive.read(name))
    first = next(iter(relationships))
    duplicate = ElementTree.Element(first.tag, first.attrib)
    duplicate.set("TargetMode", mode)
    relationships.insert(0 if position == "before" else len(relationships), duplicate)
    _rewrite(workbook_path, name, ElementTree.tostring(relationships))
    with pytest.raises(ValueError, match="relationship ID"):
        validate_workbook(workbook_path)


def test_external_hyperlink_is_never_dereferenced(workbook_path: Path) -> None:
    from openpyxl import load_workbook

    workbook = load_workbook(workbook_path)
    workbook.worksheets[0]["A2"].hyperlink = "https://example.invalid/synthetic"
    workbook.save(workbook_path)
    workbook.close()
    assert profile_workbook(workbook_path)["accepted"] is True


def _expected_documents(path: Path, single: bool) -> list:
    from langchain_core.documents import Document
    from pandas import read_excel

    sheets = read_excel(path, sheet_name=None, engine="openpyxl")
    if single:
        return [
            Document(
                page_content="\n\n".join(df.to_csv(index=False) for df in sheets.values()),
                metadata={"source": str(path), "page": 0},
            )
        ]
    return [
        Document(
            page_content=json.dumps(row.to_dict(), indent=4, default=str),
            metadata={"source": str(path), "page": i},
        )
        for i, row in enumerate(row for df in sheets.values() for _, row in df.iterrows())
    ]


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("suffix", [".xlsx", ".xls"])
def test_core_accepted_modes_match_pandas_and_never_reopen_the_path(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, single: bool, suffix: str
) -> None:
    from agents_gym_core.common_tools.vector_database import core_vector_database_tool as module

    workbook_path = workbook_path.rename(workbook_path.with_suffix(suffix))
    expected = _expected_documents(workbook_path, single)
    original = workbook_path.read_bytes()
    read_excel = module.read_excel

    def consume_snapshot(stream: io.BytesIO, **kwargs: Any) -> dict:
        assert isinstance(stream, io.BytesIO)
        canonical = stream.getvalue()
        assert isinstance(canonical, bytes) and canonical != original
        with ZipFile(io.BytesIO(original)) as source, ZipFile(stream) as sanitized:
            assert sanitized.namelist() == source.namelist()
            for member in sanitized.infolist():
                assert member.compress_type == ZIP_STORED
                assert not member.extra and not member.comment
                assert sanitized.read(member) == source.read(member.filename)
        workbook_path.write_bytes(b"replaced after validation")
        stream.seek(0)
        return read_excel(stream, **kwargs)

    monkeypatch.setattr(module, "read_excel", consume_snapshot)
    actual = module._load_xlsx(
        str(workbook_path), single=single, limits=ExcelReadLimits(max_rows=60_000)
    )
    assert actual == expected
    assert [doc.metadata["page"] for doc in actual] == list(range(len(actual)))


@pytest.mark.parametrize("method", ["single", "page", "single_recursive", "page_recursive"])
@pytest.mark.parametrize("suffix", [".xlsx", ".xls"])
def test_core_splitting_preserves_content_metadata_and_row_indices(
    workbook_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str, suffix: str
) -> None:
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from agents_gym_core.common_tools.vector_database import core_vector_database_tool as module

    workbook_path = workbook_path.rename(workbook_path.with_suffix(suffix))
    expected = _expected_documents(workbook_path, "single" in method)
    if "recursive" in method:
        splitter = RecursiveCharacterTextSplitter(chunk_size=32, chunk_overlap=4)
        expected = [
            Document(page_content=chunk, metadata=doc.metadata)
            for doc in expected
            for chunk in splitter.split_text(doc.page_content)
        ]
    create = Mock(return_value=object())
    monkeypatch.setattr(module.Chroma, "from_documents", create)
    tool = module.CoreVectorDatabaseTool.model_validate(
        {
            "file_paths": [str(workbook_path)],
            "vector_store_type": "chroma",
            "persist_directory": str(tmp_path / "index"),
            "splitting_method": method,
            "splitting_chunk_size": 32,
            "splitting_chunk_overlap": 4,
            "embeddings": object(),
            "excel_limits": {"max_rows": 60_000},
        }
    )
    tool.initialize_vector_store()
    assert create.call_args.kwargs["documents"] == expected


@pytest.mark.parametrize("suffix", [".xlsx", ".xls"])
@pytest.mark.parametrize("failure", ["grid", "corrupt", "rows", "text"])
def test_core_rejection_aborts_without_partial_index_or_cache(
    workbook_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str, suffix: str
) -> None:
    from agents_gym_core.common_tools.vector_database import core_vector_database_tool as module

    workbook_path = workbook_path.rename(workbook_path.with_suffix(suffix))
    earlier = tmp_path / "earlier.txt"
    earlier.write_text("Synthetic earlier document")
    later = tmp_path / "later.xlsx"
    later.write_bytes(workbook_path.read_bytes())
    if failure == "text":
        _repeat_shared_reference(workbook_path, b"x" * 4096, 9000)
    elif failure in {"grid", "rows"}:
        coordinate = "Z20000" if failure == "grid" else "A50001"
        _rewrite(
            workbook_path,
            "xl/worksheets/sheet1.xml",
            b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            + f'<dimension ref="A1:{coordinate}"/></worksheet>'.encode(),
        )
    else:
        workbook_path.write_bytes(b"PK\x03\x04not a workbook")
    parser = Mock(side_effect=AssertionError("Rejected workbook must not enter pandas"))
    create = Mock()
    monkeypatch.setattr(module, "read_excel", parser)
    monkeypatch.setattr(module.Chroma, "from_documents", create)
    index = tmp_path / "new-index"
    tool = module.CoreVectorDatabaseTool(
        file_paths=[str(earlier), str(workbook_path), str(later)],
        vector_store_type="chroma",
        persist_directory=str(index),
        search_strategy="hybrid",
        embeddings=object(),
    )
    with pytest.raises(ValueError):
        tool.initialize_vector_store()
    parser.assert_not_called()
    create.assert_not_called()
    assert not index.exists()
    assert str(index) not in module.CoreVectorDatabaseTool._store_cache
    assert str(index) not in module.CoreVectorDatabaseTool._bm25_cache


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("kind", ["shared_rich", "inline_rich", "cached", "implicit"])
def test_core_rich_cached_and_implicit_cells_keep_pandas_semantics(
    workbook_path: Path, single: bool, kind: str
) -> None:
    from agents_gym_core.common_tools.vector_database.core_vector_database_tool import _load_xlsx

    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    if kind == "shared_rich":
        _add_shared_string_alias(
            workbook_path,
            b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<si><r><t>Work </t></r><r><t>order</t></r></si></sst>",
        )
    else:
        with ZipFile(workbook_path) as archive:
            tree = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        if kind == "implicit":
            for row in tree.iter(namespace + "row"):
                row.attrib.pop("r", None)
                for cell in row:
                    cell.attrib.pop("r", None)
        else:
            cell = next(tree.iter(namespace + "c"))
            for child in list(cell):
                cell.remove(child)
            cell.set("t", "str" if kind == "cached" else "inlineStr")
            if kind == "cached":
                ElementTree.SubElement(cell, namespace + "f").text = '"Work order"'
                ElementTree.SubElement(cell, namespace + "v").text = "Work order"
            else:
                inline = ElementTree.SubElement(cell, namespace + "is")
                for text in ("Work ", "order"):
                    run = ElementTree.SubElement(inline, namespace + "r")
                    ElementTree.SubElement(run, namespace + "t").text = text
        _rewrite(workbook_path, "xl/worksheets/sheet1.xml", ElementTree.tostring(tree))
    assert _load_xlsx(str(workbook_path), single) == _expected_documents(workbook_path, single)


@pytest.mark.parametrize("single", [False, True])
def test_core_empty_workbook_keeps_pandas_semantics(tmp_path: Path, single: bool) -> None:
    from agents_gym_core.common_tools.vector_database.core_vector_database_tool import _load_xlsx

    path = tmp_path / "empty.xlsx"
    workbook = Workbook()
    workbook.save(path)
    workbook.close()
    assert _load_xlsx(str(path), single) == _expected_documents(path, single)


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("alias", ["worksheet", "workbook", "shared_strings"])
def test_core_opc_aliases_preserve_accepted_pandas_output(
    workbook_path: Path, single: bool, alias: str
) -> None:
    from agents_gym_core.common_tools.vector_database.core_vector_database_tool import _load_xlsx

    if alias == "worksheet":
        _rename_part(workbook_path, "xl/worksheets/sheet1.xml", "xl/worksheets/data.bin")
    elif alias == "workbook":
        _rename_part(workbook_path, "xl/workbook.xml", "xl/book.bin")
    else:
        _add_shared_string_alias(
            workbook_path,
            b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<si><t>Work order</t></si></sst>",
        )
    assert _load_xlsx(str(workbook_path), single) == _expected_documents(workbook_path, single)


@pytest.mark.parametrize("single", [False, True])
def test_core_long_json_headers_and_unicode_are_bounded_before_pandas(
    workbook_path: Path, monkeypatch: pytest.MonkeyPatch, single: bool
) -> None:
    from agents_gym_core.common_tools.vector_database import core_vector_database_tool as module

    workbook = Workbook()
    sheet = workbook.active
    assert isinstance(sheet, Worksheet)
    sheet.append(["\u5b57" * 1000, '\\"' * 1000])
    for _ in range(20):
        sheet.append(["\U0001f600" * 100, "small value"])
    workbook.save(workbook_path)
    workbook.close()
    report = profile_workbook(workbook_path)
    expected = _expected_documents(workbook_path, single)
    actual_bytes = sum(len(doc.page_content.encode()) for doc in expected)
    assert actual_bytes <= report["package"]["projected_cell_text_bytes"]
    parser = Mock(side_effect=AssertionError("Header expansion must be rejected before pandas"))
    monkeypatch.setattr(module, "read_excel", parser)
    with pytest.raises(ExcelResourceLimitError, match="projected cell text"):
        module._load_xlsx(
            str(workbook_path),
            single,
            limits=replace(ExcelReadLimits(), max_cell_text_bytes=20_000),
        )
    parser.assert_not_called()


@pytest.mark.parametrize("single", [False, True])
def test_core_legacy_biff_keeps_partitioner_and_single_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, single: bool
) -> None:
    import sys
    import types
    from datetime import datetime

    from unstructured.documents.elements import Text

    from agents_gym_core.common_tools.vector_database import core_vector_database_tool as module

    payload = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1synthetic BIFF; parser is mocked"
    path = tmp_path / "legacy.xls"
    path.write_bytes(payload)
    modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%dT%H:%M:%S%z")
    replacement = tmp_path / "replacement.xls"
    replacement.write_bytes(b"PK\x03\x04replacement ZIP after sniff")
    parser_module = types.ModuleType("unstructured.partition.xlsx")
    received = []

    def partition(*, file: io.BytesIO, **kwargs: Any) -> list:
        received.append((file.getvalue(), kwargs))
        return [Text("Synthetic BIFF content")]

    monkeypatch.setattr(parser_module, "partition_xlsx", partition, raising=False)
    monkeypatch.setitem(sys.modules, "unstructured.partition.xlsx", parser_module)
    real_open = os.open
    opens = []

    def open_once(name: Any, *args: Any, **kwargs: Any) -> int:
        fd = real_open(name, *args, **kwargs)
        if os.fspath(name) == str(path):
            opens.append(name)
            os.replace(replacement, path)
        return fd

    monkeypatch.setattr(os, "open", open_once)
    documents = module._load_xlsx(
        str(path), single, limits=ExcelReadLimits(max_archive_bytes=1, max_rows=1)
    )
    assert opens == [str(path)]
    assert received == [
        (payload, {"metadata_filename": str(path), "metadata_last_modified": modified})
    ]
    assert documents[0].metadata["source"] == str(path)
    assert documents[0].page_content == "Synthetic BIFF content"


@pytest.mark.parametrize("suffix", [".doc", ".docx"])
def test_core_word_formats_keep_the_existing_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    from langchain_core.documents import Document

    from agents_gym_core.common_tools.vector_database import core_vector_database_tool as module

    path = tmp_path / f"document{suffix}"
    path.write_bytes(b"synthetic Word fixture; parser is mocked")
    loader = Mock()
    loader.return_value.load.return_value = [Document(page_content="Synthetic Word content")]
    monkeypatch.setattr(module, "UnstructuredWordDocumentLoader", loader)
    excel = Mock(side_effect=AssertionError("Word must keep its existing loader"))
    monkeypatch.setattr(module, "_load_xlsx", excel)
    monkeypatch.setattr(module.Chroma, "from_documents", Mock(return_value=object()))
    tool = module.CoreVectorDatabaseTool.model_validate(
        {
            "file_paths": [str(path)],
            "vector_store_type": "chroma",
            "persist_directory": str(tmp_path / "index"),
            "embeddings": object(),
            "excel_limits": {item.name: 1 for item in fields(ExcelReadLimits)},
        }
    )
    tool.initialize_vector_store()
    loader.assert_called_once_with(str(path), mode="elements")
    excel.assert_not_called()


def _auxiliary_workbook(path: Path, kind: str) -> str:
    """Use real formula/cache structures without reading an external workbook."""
    from openpyxl.packaging.relationship import Relationship
    from openpyxl.workbook.external_link.external import (
        ExternalBook,
        ExternalCell,
        ExternalLink,
        ExternalRow,
        ExternalSheetData,
        ExternalSheetDataSet,
        ExternalSheetNames,
    )

    workbook = Workbook()
    sheet = workbook.active
    assert isinstance(sheet, Worksheet)
    sheet.append(["Result"])
    sheet.append([42])
    if kind == "external":
        link = ExternalLink(
            externalBook=ExternalBook(
                id="rId1",
                sheetNames=ExternalSheetNames(sheetName=["First", "Second"]),
                sheetDataSet=ExternalSheetDataSet(
                    sheetData=[
                        ExternalSheetData(
                            sheetId=index,
                            row=[ExternalRow(r=1, cell=[ExternalCell(r="A1", t="n", v="7")])],
                        )
                        for index in range(2)
                    ]
                ),
            )
        )
        link.file_link = Relationship(
            Id="rId1",
            Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/externalLinkPath",
            Target="file:///nonexistent/core-review.xlsx",
            TargetMode="External",
        )
        getattr(workbook, "_external_links").append(link)
    workbook.save(path)
    workbook.close()
    if kind == "external":
        return "xl/externalLinks/externalLink1.xml"

    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with ZipFile(path) as archive:
        manifest = ElementTree.fromstring(archive.read("[Content_Types].xml"))
        relationships = ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        worksheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    cell = worksheet.find(f'.//{namespace}c[@r="A2"]')
    assert cell is not None
    cell.clear()
    cell.set("r", "A2")
    ElementTree.SubElement(cell, namespace + "f").text = "40+2"
    ElementTree.SubElement(cell, namespace + "v").text = "42"
    ElementTree.SubElement(
        manifest,
        "{http://schemas.openxmlformats.org/package/2006/content-types}Override",
        PartName="/xl/calcChain.xml",
        ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.calcChain+xml",
    )
    ElementTree.SubElement(
        relationships,
        "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship",
        Id="calc",
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/calcChain",
        Target="calcChain.xml",
    )
    _rewrite(path, "[Content_Types].xml", ElementTree.tostring(manifest))
    _rewrite(path, "xl/_rels/workbook.xml.rels", ElementTree.tostring(relationships))
    _rewrite(path, "xl/worksheets/sheet1.xml", ElementTree.tostring(worksheet))
    part = "xl/calcChain.xml"
    _rewrite(
        path,
        part,
        b'<calcChain xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b'<c r="A2" i="1"/></calcChain>',
    )
    return part


@pytest.mark.parametrize("kind", ["calc", "external"])
@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("single", [False, True])
def test_auxiliary_spreadsheetml_uses_its_own_grammar(
    tmp_path: Path, kind: str, alias: bool, single: bool
) -> None:
    from openpyxl import load_workbook

    from agents_gym_core.common_tools.vector_database.core_vector_database_tool import _load_xlsx

    path = tmp_path / "auxiliary.xlsx"
    _auxiliary_workbook(path, kind)
    loaded = load_workbook(path, keep_links=True, data_only=False)
    if kind == "calc":
        assert loaded.active is not None and loaded.active["A2"].value == "=40+2"
    else:
        assert len(getattr(loaded, "_external_links")[0].externalBook.sheetDataSet.sheetData) == 2
    loaded.close()
    if alias:
        _rename_part(path, "xl/worksheets/sheet1.xml", "xl/worksheets/data.bin")
    assert _load_xlsx(str(path), single) == _expected_documents(path, single)
    assert "42" in _load_xlsx(str(path), single)[0].page_content


@pytest.mark.parametrize("kind", ["calc", "external"])
@pytest.mark.parametrize("failure", ["dtd", "nodes", "depth"])
def test_auxiliary_xml_keeps_generic_resource_and_entity_guards(
    tmp_path: Path, kind: str, failure: str
) -> None:
    path = tmp_path / "auxiliary.xlsx"
    part = _auxiliary_workbook(path, kind)
    alias = "xl/auxiliary.bin"
    _rename_part(path, part, alias)
    limits = ExcelReadLimits()
    if failure == "dtd":
        content = b'<!DOCTYPE x [<!ENTITY q "forbidden">]><x>&q;</x>'
        error = DefusedXmlException
    elif failure == "depth":
        content = b"<x>" * 129 + b"</x>" * 129
        error = ExcelResourceLimitError
    else:
        content = b"<x>" + b"<y/>" * 200 + b"</x>"
        limits = replace(limits, max_xml_nodes=100)
        error = ExcelResourceLimitError
    _rewrite(path, alias, content)
    with pytest.raises(error) as rejected:
        validate_workbook(path, limits=limits)
    if isinstance(rejected.value, ExcelResourceLimitError):
        assert rejected.value.part == alias


def test_worksheet_opc_role_cannot_be_changed_by_an_auxiliary_root(workbook_path: Path) -> None:
    _rename_part(workbook_path, "xl/worksheets/sheet1.xml", "xl/worksheets/data.bin")
    _rewrite(
        workbook_path,
        "xl/worksheets/data.bin",
        b'<calcChain xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b'<c r="A2" i="1"/></calcChain>',
    )
    with pytest.raises(ValueError, match="namespace/root"):
        validate_workbook(workbook_path)


@pytest.mark.parametrize("level", [1, 6, 9])
def test_bounded_deflate_drains_buffered_output_at_input_eof(level: int) -> None:
    payload = b"a" * 65_537
    compressor = zlib.compressobj(level=level, wbits=-zlib.MAX_WBITS)
    compressed = compressor.compress(payload) + compressor.flush()
    assert zlib.decompress(compressed, wbits=-zlib.MAX_WBITS) == payload
    assert _inflate(memoryview(compressed), len(payload)) == payload
    with pytest.raises(ExcelResourceLimitError) as rejected:
        _inflate(memoryview(compressed), len(payload) - 1)
    assert rejected.value.limit == len(payload) - 1
    assert rejected.value.observed == len(payload)
    for invalid in (compressed[:-1], compressed + b"\x00", compressed + compressed):
        with pytest.raises(ValueError, match="Truncated|Trailing"):
            _inflate(memoryview(invalid), len(payload) * 2)


def test_bounded_deflate_rejects_no_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    stalled = SimpleNamespace(decompress=lambda chunk, limit: b"", unconsumed_tail=b"x", eof=False)
    monkeypatch.setattr(zlib, "decompressobj", lambda *_: stalled)
    with pytest.raises(ValueError, match="no progress"):
        _inflate(memoryview(b"x"), 10)


def test_deflate_eof_drain_is_used_by_the_core_loader(tmp_path: Path) -> None:
    from agents_gym_core.common_tools.vector_database.core_vector_database_tool import _load_xlsx

    path = tmp_path / "deflate.xlsx"
    _auxiliary_workbook(path, "calc")
    with ZipFile(path) as archive:
        manifest = ElementTree.fromstring(archive.read("[Content_Types].xml"))
    ElementTree.SubElement(
        manifest,
        "{http://schemas.openxmlformats.org/package/2006/content-types}Default",
        Extension="bin",
        ContentType="application/octet-stream",
    )
    _rewrite(path, "[Content_Types].xml", ElementTree.tostring(manifest))
    _rewrite(path, "custom/review.bin", b"a" * 65_537)
    assert _load_xlsx(str(path), False) == _expected_documents(path, False)


def _long_body_workbook(path: Path, row: int = 2500) -> None:
    workbook = Workbook()
    sheet = workbook.active
    assert isinstance(sheet, Worksheet)
    sheet.append(["Result"])
    for index in range(2, 5001):
        sheet.append(["x" * 16_384 if index == row else "x"])
    workbook.save(path)
    workbook.close()


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("row", [2, 2500])
def test_one_long_body_cell_is_not_repeated_as_a_header_or_padding(
    tmp_path: Path, single: bool, row: int
) -> None:
    from agents_gym_core.common_tools.vector_database.core_vector_database_tool import _load_xlsx

    path = tmp_path / "one_long_body.xlsx"
    _long_body_workbook(path, row)
    report = profile_workbook(path)
    expected = _expected_documents(path, single)
    actual = sum(len(document.page_content.encode()) for document in expected)
    projected = report["package"]["projected_cell_text_bytes"]
    assert path.stat().st_size < 50_000
    assert actual < 200_000
    assert actual <= projected < 1_000_000
    assert _load_xlsx(str(path), single) == expected


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("kind", ["duplicates", "many_duplicates", "blanks", "unicode", "numeric"])
def test_projection_covers_pandas_header_fallbacks_mangling_and_escaping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, single: bool, kind: str
) -> None:
    from agents_gym_core.common_tools.vector_database import core_vector_database_tool as module

    headers: list[Any]
    if kind == "duplicates":
        headers = ["name", "name", "name.1", "name", "name.2", None, "Unnamed: 5", None]
    elif kind == "many_duplicates":
        headers = ["name"] * 254 + ["name.1", None]
    elif kind == "blanks":
        headers = [None] * 8
    elif kind == "numeric":
        headers = [1, 1, 1.0, True, False, 1e308, 1e-300, 0]
    else:
        headers = ["\u5b57" * 1000, "\U0001f600" * 500, '\\"' * 1000] * 2
    workbook = Workbook()
    sheet = workbook.active
    assert isinstance(sheet, Worksheet)
    sheet.append(headers)
    for _ in range(50):
        sheet.append(["body"] * len(headers))
    path = tmp_path / "headers.xlsx"
    workbook.save(path)
    workbook.close()
    expected = _expected_documents(path, single)
    actual = sum(len(document.page_content.encode()) for document in expected)
    projected = profile_workbook(path)["package"]["projected_cell_text_bytes"]
    assert actual <= projected
    assert module._load_xlsx(str(path), single) == expected
    parser = Mock(side_effect=AssertionError("Text rejection must precede pandas"))
    monkeypatch.setattr(module, "read_excel", parser)
    with pytest.raises(ExcelResourceLimitError, match="projected cell text"):
        module._load_xlsx(
            str(path), single, limits=replace(ExcelReadLimits(), max_cell_text_bytes=actual - 1)
        )
    parser.assert_not_called()


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("kind", ["shared_rich", "inline_rich", "shared_blank"])
def test_projection_counts_rich_shared_and_blank_header_values(
    tmp_path: Path, single: bool, kind: str
) -> None:
    from agents_gym_core.common_tools.vector_database.core_vector_database_tool import _load_xlsx

    path = tmp_path / "encoded_headers.xlsx"
    _auxiliary_workbook(path, "calc")
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    rich = (
        "<r><t>" + "\u5b57" * 1000 + "</t></r><r><t>" + "\U0001f600" * 500 + "</t></r>"
    ).encode()
    if kind == "shared_blank":
        _add_shared_string_alias(
            path,
            b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            b"<si><t/></si></sst>",
        )
    else:
        _repeat_shared_reference(path, b"placeholder", 20)
        if kind == "shared_rich":
            _rewrite(
                path,
                "xl/strings.bin",
                b'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                b"<si>" + rich + b"</si></sst>",
            )
        else:
            with ZipFile(path) as archive:
                worksheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            for cell in worksheet.iter(namespace + "c"):
                cell.clear()
                cell.set("t", "inlineStr")
                inline = ElementTree.SubElement(cell, namespace + "is")
                for run in ElementTree.fromstring(b"<runs>" + rich + b"</runs>"):
                    for descendant in run.iter():
                        descendant.tag = namespace + descendant.tag
                    inline.append(run)
            _rewrite(path, "xl/worksheets/sheet1.xml", ElementTree.tostring(worksheet))
    expected = _expected_documents(path, single)
    actual = sum(len(document.page_content.encode()) for document in expected)
    assert actual > 0
    assert actual <= profile_workbook(path)["package"]["projected_cell_text_bytes"]
    assert _load_xlsx(str(path), single) == expected
