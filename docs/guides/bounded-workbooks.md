# Bounded workbook ingestion

The local vector tool validates `.xlsx` workbooks and ZIP-content workbooks
renamed to `.xls` before pandas/openpyxl can read them. It opens the input once,
checks that it is a regular file, validates that snapshot, and passes only
immutable canonical ZIP bytes to pandas. It never reopens the caller path for
parsing. The canonical ZIP uses stored members and drops input ZIP extras,
comments, and timestamps while retaining validated part contents.

## Document contract

Workbook output follows the pandas document contract: splitting methods
containing `single` produce one document containing every sheet as CSV,
separated by two newlines. Other methods produce one JSON document per pandas
row, in sheet order. `source` remains the original path and `page` is a
consecutive, zero-based row index across sheets (zero for single mode).
Pandas header, missing-value, cached-value and type-inference behavior applies as
usual. The recursive/character splitting and retrieval pipeline operates on those
documents and preserves their metadata.

Genuine non-ZIP BIFF `.xls` files use the Unstructured partitioner
and its single/elements metadata contract, using bytes from the same read that
sniffed the file. Word `.doc`/`.docx` files use their format-specific loaders.

## Default Limits

`ExcelReadLimits` is immutable. Every configured limit must be a positive Python
integer; booleans, floats (including integral floats), strings, and null are
rejected without coercion. Unknown keys are rejected. Equal-to-limit inputs are
accepted; exceeding any limit aborts ingestion.

| Budget | Configuration Key | Default |
| --- | --- | ---: |
| Input archive | `max_archive_bytes` | 16 MiB |
| ZIP members | `max_members` | 1,024 |
| Expanded member | `max_member_bytes` | 16 MiB |
| Total expanded content | `max_expanded_bytes` | 64 MiB |
| Emitted canonical ZIP | `max_snapshot_bytes` | 80 MiB |
| Part name, UTF-8 | `max_name_bytes` | 1,024 bytes |
| Sheets/materializations | `max_sheets` | 32 |
| Row coordinate | `max_rows` | 50,000 |
| Column coordinate | `max_columns` | 256 |
| Aggregate dense grid | `max_grid_cells` | 500,000 cells |
| XML depth per part | `max_xml_depth` | 128 |
| XML nodes per part | `max_xml_nodes` | 1,000,000 |
| Projected materialized text | `max_cell_text_bytes` | 64 MiB |

## Trusted Configuration

Set `excel_limits` on the vector tool in trusted `agent.yaml` configuration.
Omitted keys retain their defaults; omitting the field retains the entire
default profile. For example, an explicitly reviewed 60,000-row-coordinate
export can use:

```yaml
agent:
  tools:
    - _target_: agents_gym_core.common_tools.vector_database.core_vector_database_tool.CoreVectorDatabaseTool
      name: workbook_search
      description: Search workbook exports.
      file_paths: [my_agent/data/export.xlsx]
      persist_directory: /path/to/new-index
      excel_limits:
        max_rows: 60000
```

`file_paths` are resolved relative to `agents_gym_core/agents/`. Row coordinates
include the header row: this example permits row 60,000, not 60,000 data rows
plus a header. All other budgets remain independent, including the 500,000-cell
dense-grid cap, ZIP byte limits, XML complexity, and projected-text cap. Raising
the row budget does not disable those checks or relax malformed-package rejection.

Python callers can pass the same mapping, or an `ExcelReadLimits` instance, as
the `excel_limits` keyword to `CoreVectorDatabaseTool`. Both forms use
the dataclass's strict positive-integer validation before Pydantic can coerce
values. `excel_limits` is construction-time configuration, never an LLM-visible
argument; the search input schema remains `query` only.

The configured profile applies when ingesting `.xlsx` and ZIP-content `.xls`
files into a new index. It does not affect genuine BIFF or Word loading,
pre-loaded `documents`, or reuse of an existing in-memory/persisted store.
Store/cache identity is unchanged: changing budgets does not invalidate or
revalidate an existing index. Use a fresh `persist_directory` when a new build
under the reviewed profile is required.

The ZIP gate cross-checks directory/local headers, actual DEFLATE output,
sizes, and CRCs rather than trusting declared uncompressed sizes. Ordinary
streaming data descriptors and local ZIP64 sizes are supported; ZIP64 central
directories, multi-disk archives, duplicate parts, encryption, and ambiguous
part URIs fail closed. OPC content types and relationship bindings select XML
parts even when their names do not end in `.xml`. Worksheet coordinate grammar
is applied to worksheet roots, whose OPC roles and roots are cross-checked;
calculation chains and external-link caches are not interpreted as worksheet cells.
All XML parts retain the generic node/depth and DTD/entity guards. External
hyperlinks are not fetched. Bounded DEFLATE reads drain buffered output before
checking EOF, without relaxing output limits or accepting trailing streams.

Grid accounting combines declared dimensions with explicit and inferred
coordinates and charges reused worksheet parts for every materialization.
Materialized-text accounting includes shared-string references, rich/inline and
cached text, scalar formatting, Unicode/JSON escaping, actual cell-text totals,
repeated first-row JSON headers (including blank-name and duplicate-suffix
allowances), and dense-cell overhead. A long body cell is charged once, not
as every row's header or padded value. It is a conservative projection, not a
truncation allowance or a precise prediction of process memory.

## Metadata-Only Profile

From the repository root, run the standalone script to avoid importing Core's
provider/vector packages:

```sh
python src/agents_gym_core/common_tools/vector_database/profile_excel.py workbook.xlsx
```

The script emits JSON and exits with `0` on acceptance, `1` on validation/read
failure, and argparse's `2` on invalid command-line arguments. It uses the same
default `ExcelReadLimits` as an unconfigured tool. It loads the validation module directly
and reports structural metadata. Document parsing, embedding and indexing belong to the
ingestion workflow. `profile_workbook()` provides the same report programmatically through
Core's normal package imports.

To profile a custom ingestion budget, pass exactly the same limits to the API:

```python
from agents_gym_core.common_tools.vector_database.bounded_excel_loader import (
    ExcelReadLimits,
    profile_workbook,
)

limits = ExcelReadLimits(max_rows=60_000)
report = profile_workbook("workbook.xlsx", limits=limits)
# For an already configured tool, use limits=tool.excel_limits instead.
```

Custom profiles use the same validation and fail-closed rejection policy as ingestion. Use the API for
custom limits and the standalone CLI for the default profile.

Reports contain the caller path, part names, byte counts, package counts,
declared/observed dimensions, XML complexity, and projected text sizes. They
contain no cell values, sheet titles, formulas, or extracted documents. Parser
exceptions are sanitized; rejected files return no partial profile. Paths and
part names can themselves be sensitive, so review reports before sharing them.

## Rejection Policy

New index builds validate the selected workbooks before embedding documents or updating
vector/BM25 stores. A rejection aborts that build as a whole: rejected workbooks are never
skipped, truncated or partially indexed. Existing persisted-store reuse
follows the tool's normal cache behavior.

Review the export and profile with the workbook owner. Agree on standard XLSX exports,
partitioning, file selection and budget changes before ingestion. Validate Office feature
handling and parser compatibility with representative workbooks, and measure retrieval
quality with task-specific evaluations.
