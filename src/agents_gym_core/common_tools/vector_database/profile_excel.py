"""Validate a vector-tool XLSX workbook and emit bounded JSON metadata only."""

import argparse
import json
from dataclasses import asdict
from typing import TYPE_CHECKING
from xml.etree.ElementTree import ParseError

if TYPE_CHECKING or __package__:
    from .bounded_excel_loader import ExcelReadLimits, ExcelResourceLimitError, profile_workbook
else:
    # Direct script execution avoids importing Core's provider/vector packages.
    from bounded_excel_loader import ExcelReadLimits, ExcelResourceLimitError, profile_workbook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="XLSX or ZIP-content .xls to validate under default budgets")
    args = parser.parse_args(argv)
    limits = ExcelReadLimits()
    try:
        report = profile_workbook(args.file, limits=limits)
    except (OSError, ValueError, ParseError) as exc:
        error: dict[str, object] = {
            "type": type(exc).__name__,
            "message": "Workbook is unsafe or unsupported; re-save as standard XLSX and retry.",
        }
        if isinstance(exc, ExcelResourceLimitError):
            part = exc.part or "<archive>"
            error.update(
                message=exc.args[0],
                part=part[: limits.max_name_bytes],
                part_truncated=len(part) > limits.max_name_bytes,
                limit=exc.limit,
                observed=exc.observed,
            )
        elif isinstance(exc, OSError):
            error.update(
                message="Cannot read workbook; check the file path, permissions and file type.",
                errno=exc.errno,
            )
        # XML parser exception text can contain input-controlled names or values.
        report = {
            "file": args.file,
            "accepted": False,
            "rejection_policy": "abort_index_build",
            "limits": asdict(limits),
            "error": error,
            "action": "No index was built. Review the export/profile with its owner; do not "
            "raise budgets or skip rejected files without explicit approval.",
        }
        print(json.dumps(report, sort_keys=True))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
