"""Validate eval-set and evaluator JSON files.

Usage:
    python -m agents_gym_core.prompt_optimization.check_eval_sets [--agent AGENT_NAME] [--check-evaluators]
"""

import argparse
import ast
import json
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

# UUID pattern: 8-4-4-4-12 hex digits
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)


class Severity(Enum):
    FAIL = "FAIL"
    WARN = "WARN"
    PASS = "PASS"


@dataclass
class Issue:
    severity: Severity
    file: str
    message: str


def _is_uuid(s: str) -> bool:
    return bool(_UUID_RE.match(s))


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _annassign_has_default(item: ast.AnnAssign) -> bool:
    """Return whether an annotated assignment supplies a Pydantic default."""
    value = item.value
    if value is None:
        return False

    if not isinstance(value, ast.Call):
        return True

    function_name = None
    if isinstance(value.func, ast.Name):
        function_name = value.func.id
    elif isinstance(value.func, ast.Attribute):
        function_name = value.func.attr

    if function_name != "Field":
        return True

    if value.args:
        return not (isinstance(value.args[0], ast.Constant) and value.args[0].value is Ellipsis)

    for keyword in value.keywords:
        if keyword.arg == "default":
            return not (isinstance(keyword.value, ast.Constant) and keyword.value.value is Ellipsis)
        if keyword.arg == "default_factory":
            return not (isinstance(keyword.value, ast.Constant) and keyword.value.value is None)

    return False


def _parse_custom_criteria_fields(evaluator_path: Path) -> set[str] | None:
    """Extract required field names from a custom evaluator's criteria class.

    Reads the evaluator JSON to find evaluatorSchema
    (e.g., ``"file://claim_history_evaluators.py:ClaimHistoryLookupSuccessEvaluator"``),
    resolves the Python file in the ``custom/`` subdirectory, and uses AST to
    find the ``BaseEvaluationCriteria`` subclass and its field names.

    Returns *None* if the evaluator has no ``evaluatorSchema`` (built-in type).
    """
    try:
        parsed_edata: object = json.loads(evaluator_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(parsed_edata, dict):
        return None
    edata = cast(dict[str, object], parsed_edata)

    schema_ref = edata.get("evaluatorSchema")
    if not isinstance(schema_ref, str) or not schema_ref.startswith("file://"):
        return None

    # Parse "file://filename.py:ClassName"
    ref_part = schema_ref[len("file://") :]
    if ":" not in ref_part:
        return None

    py_filename, _evaluator_class = ref_part.split(":", 1)

    # Resolve file path: custom/ subdirectory relative to evaluator file
    custom_dir = evaluator_path.parent / "custom"
    py_file = custom_dir / py_filename
    if not py_file.is_file():
        return None

    # Parse Python file with AST
    try:
        source = py_file.read_text()
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    # Find BaseEvaluationCriteria subclass and extract field names
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        # Check if this class inherits from BaseEvaluationCriteria
        if not any(
            (isinstance(base, ast.Name) and base.id == "BaseEvaluationCriteria")
            or (isinstance(base, ast.Attribute) and base.attr == "BaseEvaluationCriteria")
            for base in node.bases
        ):
            continue

        # Found the criteria class — extract annotated fields that Pydantic
        # considers required. Fields with direct defaults or Field defaults are
        # optional in serialized evaluation criteria.
        fields: set[str] = set()
        for item in node.body:
            if (
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and not _annassign_has_default(item)
            ):
                fields.add(item.target.id)

        return fields

    return None


# ---------------------------------------------------------------------------
# Eval-set checks
# ---------------------------------------------------------------------------


def check_eval_set(
    path: Path,
    evaluator_ids: set[str] | None = None,
    evaluators_using_behavior: set[str] | None = None,
    evaluator_types: dict[str, str] | None = None,
    custom_required_fields: dict[str, set[str]] | None = None,
) -> list[Issue]:
    """Validate a single eval-set JSON file.

    Args:
        path: Path to the eval-set JSON file.
        evaluator_ids: Set of evaluator IDs found on disk (for cross-ref checks).
        evaluators_using_behavior: Set of evaluator IDs whose prompts use {{ExpectedAgentBehavior}}.
        evaluator_types: Dict mapping evaluator ID to its evaluatorTypeId.
        custom_required_fields: Dict mapping evaluator ID to the set of required
            field names parsed from its BaseEvaluationCriteria subclass.
    """
    fname = path.name
    issues: list[Issue] = []
    is_runtime = fname == "evaluation-set-runtime.json"

    try:
        parsed_data: object = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [Issue(Severity.FAIL, fname, f"cannot read file: {e}")]
    if not isinstance(parsed_data, dict):
        return [Issue(Severity.FAIL, fname, "top-level JSON value must be an object")]
    data = cast(dict[str, object], parsed_data)

    # Rule 1: version must be "1.0"
    version = data.get("version")
    if version != "1.0":
        issues.append(Issue(Severity.FAIL, fname, f'version is "{version}", expected "1.0"'))

    # evaluatorRefs
    raw_evaluator_refs = data.get("evaluatorRefs", [])
    evaluator_refs = (
        [ref for ref in cast(list[object], raw_evaluator_refs) if isinstance(ref, str)]
        if isinstance(raw_evaluator_refs, list)
        else []
    )

    if is_runtime:
        # Rule 15: runtime eval-sets must have empty evaluatorRefs
        if evaluator_refs:
            issues.append(
                Issue(
                    Severity.FAIL,
                    fname,
                    f"runtime eval-set should have empty evaluatorRefs, found: {evaluator_refs}",
                )
            )
    else:
        if not evaluator_refs:
            issues.append(Issue(Severity.FAIL, fname, "evaluatorRefs is missing or empty"))

        # Rule 3: evaluatorRefs should use semantic names, not UUIDs
        uuid_refs = [r for r in evaluator_refs if _is_uuid(r)]
        if uuid_refs:
            issues.append(
                Issue(
                    Severity.FAIL,
                    fname,
                    f"evaluatorRefs uses UUIDs instead of semantic names: {uuid_refs}",
                )
            )

        # Rule 6 (cross-ref): evaluatorRefs should reference evaluators that exist on disk
        if evaluator_ids is not None:
            missing = [r for r in evaluator_refs if r not in evaluator_ids]
            if missing:
                issues.append(
                    Issue(
                        Severity.FAIL,
                        fname,
                        f"evaluatorRefs reference evaluators not found on disk: {missing}",
                    )
                )

    evaluator_refs_set = set(evaluator_refs)
    # Track which evaluatorRefs are covered across all evaluations (for Rule 5)
    covered_evaluator_refs: set[str] = set()

    # Per-evaluation checks
    raw_evaluations = data.get("evaluations", [])
    evaluations: list[object] = (
        cast(list[object], raw_evaluations) if isinstance(raw_evaluations, list) else []
    )
    if not evaluations:
        issues.append(Issue(Severity.WARN, fname, "evaluations array is empty"))

    for raw_eval_item in evaluations:
        if not isinstance(raw_eval_item, dict):
            issues.append(Issue(Severity.FAIL, fname, "evaluation entry is not an object"))
            continue
        eval_item = cast(dict[str, object], raw_eval_item)
        eval_name = eval_item.get("name", eval_item.get("id", "<unnamed>"))

        criterias = eval_item.get("evaluationCriterias")

        if is_runtime:
            # Rule 15: runtime eval-sets must NOT have evaluationCriterias
            if criterias is not None:
                issues.append(
                    Issue(
                        Severity.FAIL,
                        fname,
                        f'evaluation "{eval_name}" — runtime eval-set should not have evaluationCriterias',
                    )
                )
            continue

        # Rule 2: evaluationCriterias required
        if criterias is None:
            expected_output = eval_item.get("expectedOutput")
            if expected_output not in (None, {}):
                issues.append(
                    Issue(
                        Severity.FAIL,
                        fname,
                        f'evaluation "{eval_name}" — expectedOutput should be {{}} (empty), '
                        "data belongs in evaluationCriterias",
                    )
                )
            issues.append(
                Issue(
                    Severity.FAIL, fname, f'evaluation "{eval_name}" — missing evaluationCriterias'
                )
            )
            continue

        if not isinstance(criterias, dict):
            issues.append(
                Issue(
                    Severity.FAIL,
                    fname,
                    f'evaluation "{eval_name}" — evaluationCriterias is not a dict',
                )
            )
            continue
        typed_criterias = cast(dict[str, object], criterias)

        # Track covered evaluatorRefs (keys that are present and not null)
        criteria_keys = set(typed_criterias)
        covered_evaluator_refs.update(criteria_keys)

        # Rule 5 (per-evaluation): no extra keys that aren't in evaluatorRefs
        extra_in_criteria = criteria_keys - evaluator_refs_set
        if extra_in_criteria:
            issues.append(
                Issue(
                    Severity.FAIL,
                    fname,
                    f'evaluation "{eval_name}" — evaluationCriterias has extra keys not in evaluatorRefs: {sorted(extra_in_criteria)}',
                )
            )

        # Rule 12: each evaluationCriterias entry must have the correct data field
        # based on evaluator type. Criteria can be null when evaluator has defaultEvaluationCriteria.
        #
        # Required criteria field per evaluatorTypeId; any other type needs expectedOutput.
        type_to_required_field = {
            "uipath-tool-call-args": "toolCalls",
            "uipath-tool-call-output": "toolOutputs",
            "uipath-tool-call-count": "toolCallsCount",
            "uipath-tool-call-order": "toolCallsOrder",
            "uipath-llm-judge-trajectory-simulation": "expectedAgentBehavior",
            "uipath-llm-judge-trajectory-similarity": "expectedAgentBehavior",
            "uipath-llm-judge-trajectory": "expectedAgentBehavior",
            "uipath-binary-classification": "expectedClass",
            "uipath-multiclass-classification": "expectedClass",
            "uipath-contains": "searchText",
        }
        for ref_id, criteria in typed_criterias.items():
            # null criteria = use evaluator's defaultEvaluationCriteria — always valid
            if criteria is None:
                continue
            if not isinstance(criteria, dict):
                issues.append(
                    Issue(
                        Severity.FAIL,
                        fname,
                        f'evaluation "{eval_name}" — evaluationCriterias["{ref_id}"] is not a dict (and not null)',
                    )
                )
                continue
            typed_criteria = cast(dict[str, object], criteria)

            # Custom evaluator: required fields come from BaseEvaluationCriteria subclass
            # Accept both snake_case (Python) and camelCase (JSON) variants
            if custom_required_fields and ref_id in custom_required_fields:
                missing: list[str] = []
                for f in custom_required_fields[ref_id]:
                    camel = _snake_to_camel(f)
                    if f not in typed_criteria and camel not in typed_criteria:
                        missing.append(f"{f} (or {camel})" if f != camel else f)
                if missing:
                    issues.append(
                        Issue(
                            Severity.FAIL,
                            fname,
                            f'evaluation "{eval_name}" — evaluationCriterias["{ref_id}"] missing custom field(s): {missing}',
                        )
                    )
            else:
                # Built-in evaluator type: determine required field from evaluatorTypeId.
                # An unknown type (no evaluator file for ref_id) cannot be checked here;
                # a dangling ref is reported by the evaluatorRefs checks instead.
                eval_type = (evaluator_types or {}).get(ref_id, "")
                required_field = type_to_required_field.get(eval_type, "expectedOutput")

                if eval_type and required_field not in typed_criteria:
                    issues.append(
                        Issue(
                            Severity.FAIL,
                            fname,
                            f'evaluation "{eval_name}" — evaluationCriterias["{ref_id}"] missing {required_field}',
                        )
                    )

            # expectedAgentBehavior only required when evaluator prompt uses {{ExpectedAgentBehavior}}
            if "expectedAgentBehavior" not in typed_criteria:
                if evaluators_using_behavior and ref_id in evaluators_using_behavior:
                    issues.append(
                        Issue(
                            Severity.WARN,
                            fname,
                            f'evaluation "{eval_name}" — evaluationCriterias["{ref_id}"] missing expectedAgentBehavior (evaluator prompt uses {{{{ExpectedAgentBehavior}}}})',
                        )
                    )

    # Rule 5 (eval-set level): every evaluatorRef must be covered by at least one evaluation
    if not is_runtime and evaluator_refs_set:
        uncovered = evaluator_refs_set - covered_evaluator_refs
        if uncovered:
            issues.append(
                Issue(
                    Severity.FAIL,
                    fname,
                    f"evaluatorRefs not covered by any evaluation: {sorted(uncovered)}",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Evaluator checks
# ---------------------------------------------------------------------------


def check_evaluator(path: Path) -> list[Issue]:
    """Validate a single evaluator JSON file."""
    fname = path.name
    issues: list[Issue] = []

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return [Issue(Severity.FAIL, fname, f"cannot read file: {e}")]

    # Rule 8: version must be "1.0"
    version = data.get("version")
    if version != "1.0":
        issues.append(Issue(Severity.FAIL, fname, f'version is "{version}", expected "1.0"'))

    # Rule 7: filename must match id
    evaluator_id = data.get("id", "")
    expected_fname = f"evaluator-{evaluator_id}.json"
    if path.name != expected_fname:
        issues.append(
            Issue(
                Severity.FAIL,
                fname,
                f'id is "{evaluator_id}" but filename is "{path.name}" (expected "{expected_fname}")',
            )
        )

    # Rule 4: no createdAt / updatedAt
    for field in ("createdAt", "updatedAt"):
        if field in data:
            issues.append(
                Issue(Severity.FAIL, fname, f'contains "{field}" — not allowed in evaluator files')
            )

    # Rule 9: must have evaluatorTypeId + evaluatorConfig
    if "evaluatorTypeId" not in data:
        issues.append(Issue(Severity.FAIL, fname, "missing evaluatorTypeId"))
    if "evaluatorConfig" not in data:
        issues.append(Issue(Severity.FAIL, fname, "missing evaluatorConfig"))

    # Rule 10: LLM judge prompts must use correct placeholders
    # Output judges: {{ExpectedOutput}} + {{ActualOutput}}
    # Trajectory judges: {{ExpectedAgentBehavior}} + {{AgentRunHistory}}
    config = data.get("evaluatorConfig", {})
    prompt = config.get("prompt", "")
    evaluator_type = data.get("evaluatorTypeId", "")
    if "llm-judge" in evaluator_type and prompt:
        is_trajectory = "trajectory" in evaluator_type
        if is_trajectory:
            if "{{ExpectedAgentBehavior}}" not in prompt:
                issues.append(
                    Issue(
                        Severity.FAIL,
                        fname,
                        "Trajectory LLM judge prompt missing {{ExpectedAgentBehavior}} placeholder",
                    )
                )
            if "{{AgentRunHistory}}" not in prompt:
                issues.append(
                    Issue(
                        Severity.FAIL,
                        fname,
                        "Trajectory LLM judge prompt missing {{AgentRunHistory}} placeholder",
                    )
                )
        else:
            if "{{ExpectedOutput}}" not in prompt:
                issues.append(
                    Issue(
                        Severity.FAIL,
                        fname,
                        "LLM judge prompt missing {{ExpectedOutput}} placeholder",
                    )
                )
            if "{{ActualOutput}}" not in prompt:
                issues.append(
                    Issue(
                        Severity.FAIL,
                        fname,
                        "LLM judge prompt missing {{ActualOutput}} placeholder",
                    )
                )

    return issues


# ---------------------------------------------------------------------------
# Agent-level orchestration
# ---------------------------------------------------------------------------


def check_agent(
    agent_dir: Path, check_evaluators: bool, exclude: frozenset[str] = frozenset()
) -> list[Issue]:
    """Validate the eval-sets (except the ``exclude`` file names) and evaluators of one agent."""
    issues: list[Issue] = []

    eval_sets_dir = agent_dir / "evaluations" / "eval-sets"
    evaluators_dir = agent_dir / "evaluations" / "evaluators"

    # Discover evaluator IDs, types, prompts, and custom criteria fields from disk
    evaluator_ids: set[str] = set()
    evaluators_using_behavior: set[str] = set()
    evaluator_types: dict[str, str] = {}  # evaluator_id -> evaluatorTypeId
    custom_required_fields: dict[str, set[str]] = {}  # evaluator_id -> criteria fields
    evaluator_files: list[Path] = []
    if evaluators_dir.is_dir():
        evaluator_files = sorted(evaluators_dir.glob("evaluator-*.json"))
        for ef in evaluator_files:
            try:
                edata = json.loads(ef.read_text())
                eid = edata.get("id", "")
                if eid:
                    evaluator_ids.add(eid)
                    # Store evaluator type
                    etype = edata.get("evaluatorTypeId", "")
                    if etype:
                        evaluator_types[eid] = etype
                    # Check if evaluator prompt uses {{ExpectedAgentBehavior}}
                    prompt = edata.get("evaluatorConfig", {}).get("prompt", "")
                    if "{{ExpectedAgentBehavior}}" in prompt:
                        evaluators_using_behavior.add(eid)
                    # Parse custom evaluator criteria fields from evaluatorSchema
                    fields = _parse_custom_criteria_fields(ef)
                    if fields is not None:
                        custom_required_fields[eid] = fields
            except (json.JSONDecodeError, OSError):
                pass

    # Check eval-sets
    if eval_sets_dir.is_dir():
        eval_set_files = sorted(
            f for f in eval_sets_dir.glob("evaluation-set-*.json") if f.name not in exclude
        )
        if not eval_set_files:
            issues.append(Issue(Severity.WARN, agent_dir.name, "no eval-set files found"))
        for esf in eval_set_files:
            issues.extend(
                check_eval_set(
                    esf,
                    evaluator_ids if check_evaluators else None,
                    evaluators_using_behavior if check_evaluators else None,
                    # Types and custom fields only describe what each criteria entry must
                    # contain (Rule 12), so they apply with or without --check-evaluators.
                    evaluator_types,
                    custom_required_fields,
                )
            )
    else:
        issues.append(Issue(Severity.WARN, agent_dir.name, "no evaluations/eval-sets/ directory"))

    if not check_evaluators:
        return issues

    # Check evaluators
    if not evaluators_dir.is_dir():
        issues.append(Issue(Severity.WARN, agent_dir.name, "no evaluations/evaluators/ directory"))
        return issues

    if not evaluator_files:
        issues.append(Issue(Severity.WARN, agent_dir.name, "no evaluator files found"))

    for ef in evaluator_files:
        issues.extend(check_evaluator(ef))

    # Cross-ref: every evaluator should be referenced by at least one eval-set
    referenced_ids: set[str] = set()
    if eval_sets_dir.is_dir():
        for esf in eval_sets_dir.glob("evaluation-set-*.json"):
            if esf.name in exclude:
                continue
            try:
                es_data = json.loads(esf.read_text())
                referenced_ids.update(es_data.get("evaluatorRefs", []))
            except (json.JSONDecodeError, OSError):
                pass

    for ef in evaluator_files:
        try:
            eid = json.loads(ef.read_text()).get("id", "")
            if eid and eid not in referenced_ids:
                issues.append(
                    Issue(Severity.WARN, ef.name, "evaluator not referenced by any eval-set")
                )
        except (json.JSONDecodeError, OSError):
            pass

    return issues


# ---------------------------------------------------------------------------
# Programmatic API
# ---------------------------------------------------------------------------


def validate_agent(agent_dir: Path, check_evaluators: bool = True, verbose: bool = True) -> bool:
    """Validate eval-sets and evaluators for a single agent directory.

    Prints issues and returns True if no errors found.
    """
    issues = check_agent(agent_dir, check_evaluators)
    errors = [i for i in issues if i.severity == Severity.FAIL]
    warnings = [i for i in issues if i.severity == Severity.WARN]

    if verbose and (errors or warnings):
        print(f"\n=== Validation: {agent_dir.name} ===")
        for issue in issues:
            if issue.severity != Severity.PASS:
                print(f"  {issue.severity.value}: [{issue.file}] {issue.message}")

    if verbose:
        print(f"\n  Errors: {len(errors)}, Warnings: {len(warnings)}")

    return len(errors) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _get_agents_root() -> Path:
    """Return the agents directory root."""
    return Path(__file__).resolve().parents[1] / "agents"


def _discover_agent_dirs(agents_root: Path, agent_name: str | None) -> list[Path]:
    """Find agent directories that have evaluations."""
    if agent_name:
        if Path(agent_name).name != agent_name or agent_name in (".", ".."):
            print(f"Error: not an agent folder name: {agent_name}")
            sys.exit(1)
        agent_dir = agents_root / agent_name
        if not agent_dir.is_dir():
            print(f"Error: agent directory not found: {agent_dir}")
            sys.exit(1)
        return [agent_dir]

    dirs: list[Path] = []
    if not agents_root.is_dir():
        return dirs
    for d in sorted(agents_root.iterdir()):
        if d.is_dir() and (d / "evaluations").is_dir():
            dirs.append(d)
    return dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate eval-set and evaluator JSON files.")
    parser.add_argument(
        "--agents-root", type=Path, help="explicit agents directory (default: bundled)"
    )
    parser.add_argument("--agent", help="Check a specific agent (default: all)")
    parser.add_argument(
        "--exclude", action="append", default=[], help="eval-set file name to leave out"
    )
    parser.add_argument(
        "--check-evaluators", action="store_true", help="Also validate evaluator files"
    )
    parser.add_argument(
        "--core",
        action="store_true",
        help="Accepted for compatibility; the core agents are always checked",
    )
    args = parser.parse_args()

    agents_root = args.agents_root or _get_agents_root()
    agent_dirs = _discover_agent_dirs(agents_root, args.agent)

    if not agent_dirs:
        print("No agents with evaluations found.")
        sys.exit(0)

    total_agents = 0
    total_eval_sets = 0
    total_evaluators = 0
    total_errors = 0
    total_warnings = 0

    for agent_dir in agent_dirs:
        issues = check_agent(agent_dir, args.check_evaluators, frozenset(args.exclude))

        # Count files
        es_dir = agent_dir / "evaluations" / "eval-sets"
        ev_dir = agent_dir / "evaluations" / "evaluators"
        n_es = (
            len([f for f in es_dir.glob("evaluation-set-*.json") if f.name not in args.exclude])
            if es_dir.is_dir()
            else 0
        )
        n_ev = len(list(ev_dir.glob("evaluator-*.json"))) if ev_dir.is_dir() else 0

        errors = [i for i in issues if i.severity == Severity.FAIL]
        warnings = [i for i in issues if i.severity == Severity.WARN]

        if errors or warnings:
            print(f"\n=== Agent: {agent_dir.name} ===")
            for issue in issues:
                if issue.severity != Severity.PASS:
                    print(f"  {issue.severity.value}: [{issue.file}] {issue.message}")

        total_agents += 1
        total_eval_sets += n_es
        total_evaluators += n_ev
        total_errors += len(errors)
        total_warnings += len(warnings)

    print(f"\n{'=' * 40}")
    print(f"Agents checked:     {total_agents}")
    print(f"Eval-sets checked:  {total_eval_sets}")
    if args.check_evaluators:
        print(f"Evaluators checked: {total_evaluators}")
    print(f"Errors:             {total_errors}")
    print(f"Warnings:           {total_warnings}")

    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
