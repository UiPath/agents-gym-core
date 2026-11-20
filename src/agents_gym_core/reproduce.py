"""``gym reproduce``: evaluate the paper's prompts on the test splits, one paper table at a time.

Each :class:`Table` lists the agents, the prompt versions (one per row of the paper's table) and
the system model; :func:`plan` turns one into testbed runs with the paper's settings.
"""

import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agents_gym_core.paths import project_root
from agents_gym_core.utils.gym_settings_core import GymSettingsCore

AGENTS_DIR = Path(__file__).resolve().parent / "agents"
OUTPUT_DIR = project_root() / "experiments" / "reproduce"

BENCHMARKS = ("hotpotqa", "hover", "aime_2025_agent", "livebench_math", "ifbench", "papillon")
ENTERPRISE = (
    "claim_history",
    "customer_support_triage_agent",
    "meeting_management",
    "product_questions_agent",
    "sql",
)
GPT_41_MINI = "gpt-4.1-mini-2025-04-14"
GPT_5 = "gpt-5-2025-08-07"
QWEN_9B = "Qwen/Qwen3.5-9B"
# Qwen3.5-122B-A10B has no public default endpoint: the table without a system model takes the
# model name of yours from this variable or --model.
QWEN_122B_MODEL_ENV = "QWEN_122B_MODEL"

EVAL_TEMPERATURE = "1.0"  # every paper evaluation
# The paper's score (Eqs. 1-2): each datapoint's mean over its evaluators, then the mean over
# datapoints. With one evaluator per example (the public benchmarks) it is the per-example mean.
AGGREGATION = "datapoint_run"
DEFAULT_RUNS = 5
# basic:auto-graceful forces end_execution after this many model calls: the 95th percentile
# of model calls in successful runs with a 30-call budget.
FORCED_CALL_LIMITS = {"sql": 12, "meeting_management": 12}
FORCED_CALL_LIMIT = 6
# The other runs get the setting's default, not a value left in the shell or .env.
CALL_LIMIT = GymSettingsCore.model_fields["UIPATH_AGENTS_GYM_LOOP_RECURSION_LIMIT"].default


@dataclass(frozen=True)
class Table:
    name: str
    title: str
    agents: tuple[str, ...]
    rows: tuple[tuple[str, str], ...]  # (the paper's row label, agent version)
    model: str  # "" = no public default: QWEN_122B_MODEL or the caller's --model
    loop: str = ""  # "" = each agent's own loop (basic; dspy-program for the benchmarks)
    runs: dict[str, int] = field(default_factory=dict)  # agents not run DEFAULT_RUNS times
    note: str = ""


_MAIN_ENTERPRISE_ROWS = (
    ("Baseline", "baseline-test"),
    ("GEPA", "gepa__gpt-5-test"),
    ("ReAPO", "reapo__gpt-5-test"),
)

TABLES: tuple[Table, ...] = (
    Table(
        "main-mini",
        "Public benchmarks, GPT-4.1 Mini optimizer",
        BENCHMARKS,
        (
            ("Baseline", "baseline-test"),
            ("MIPROv2-ND", "miprov2-nd-test"),
            ("MIPROv2", "miprov2-test"),
            ("GEPA", "gepa__gpt-4.1-mini-test"),
            ("GEPA-Merge", "gepa-merge__gpt-4.1-mini-test"),
            ("ReAPO", "reapo__gpt-4.1-mini-test"),
        ),
        GPT_41_MINI,
        note="AIME is a single-module program, on which GEPA and GEPA-Merge are equivalent: one prompt.",
    ),
    Table(
        "main-gpt5",
        "Public benchmarks, GPT-5 optimizer",
        BENCHMARKS,
        (("GEPA", "gepa__gpt-5-test"), ("ReAPO", "reapo__gpt-5-test")),
        GPT_41_MINI,
    ),
    Table(
        "transfer",
        "Cross-model transfer to Qwen3.5-9B (GPT-4.1 Mini prompts)",
        BENCHMARKS,
        (
            ("Baseline", "baseline-test"),
            ("GEPA", "gepa__gpt-4.1-mini-test"),
            ("ReAPO", "reapo__gpt-4.1-mini-test"),
        ),
        QWEN_9B,
        runs={"aime_2025_agent": 10},
    ),
    Table(
        "self-opt-9b",
        "Self-optimization, Qwen3.5-9B as optimizer and system model",
        ("hotpotqa", "hover", "aime_2025_agent", "livebench_math"),
        (
            ("Baseline", "baseline-test"),
            ("GEPA", "gepa__qwen3.5-9b-test"),
            ("ReAPO", "reapo__qwen3.5-9b-test"),
        ),
        QWEN_9B,
        runs={"aime_2025_agent": 10},
    ),
    Table(
        "self-opt-122b",
        "Self-optimization, Qwen3.5-122B-A10B as optimizer and system model",
        ("hotpotqa", "hover", "aime_2025_agent"),
        (
            ("Baseline", "baseline-test"),
            ("GEPA", "gepa__qwen3.5-122b-test"),
            ("ReAPO", "reapo__qwen3.5-122b-test"),
        ),
        "",
    ),
    Table(
        "tool-ablation",
        "Tool ablation, public benchmarks",
        ("hotpotqa", "hover", "ifbench", "livebench_math"),
        (
            ("GPT-5 Full", "reapo__gpt-5-test"),
            ("GPT-5 Score-Only", "reapo__gpt-5__score-only-test"),
            ("GPT-5 Macro", "reapo__gpt-5__macro-test"),
            ("GPT-4.1 Mini Full", "reapo__gpt-4.1-mini-test"),
            (
                "GPT-4.1 Mini Score-Only",
                "reapo__gpt-4.1-mini__score-only-test",
            ),
            ("GPT-4.1 Mini Macro", "reapo__gpt-4.1-mini__macro-test"),
        ),
        GPT_41_MINI,
        note=(
            "HotpotQA and LiveBench use `baseline-test` for GPT-4.1 Mini Score-Only: "
            "those optimization runs selected the baseline prompt."
        ),
    ),
    Table(
        "skill",
        "Skill prompt ablation",
        ("hotpotqa", "hover", "aime_2025_agent", "livebench_math", "ifbench"),
        (
            ("Baseline", "baseline-test"),
            ("σ_Trimmed", "reapo__gpt-5__skill-trimmed-test"),
            ("σ_Full", "reapo__gpt-5-test"),
        ),
        GPT_41_MINI,
    ),
    Table(
        "frontier",
        "Frontier optimizer ablation (GPT-5 vs Claude Opus 4.6)",
        BENCHMARKS,
        (
            ("Baseline", "baseline-test"),
            ("GPT-5", "reapo__gpt-5-test"),
            ("Opus", "reapo__opus-4.6-test"),
        ),
        GPT_41_MINI,
    ),
    Table("agents", "Enterprise agents, GPT-5", ENTERPRISE, _MAIN_ENTERPRISE_ROWS, GPT_5),
    Table(
        "agents-forced",
        "Enterprise agents with a forced final answer (6 model calls, 12 for SQL and Meeting Mgmt)",
        ENTERPRISE,
        _MAIN_ENTERPRISE_ROWS,
        GPT_5,
        loop="basic:auto-graceful",
    ),
    Table(
        "tool-ablation-agents",
        "Tool ablation, enterprise agents (GPT-5 optimizer)",
        ENTERPRISE,
        (
            ("Baseline", "baseline-test"),
            ("Full", "reapo__gpt-5__full-test"),
            ("Score-Only", "reapo__gpt-5__score-only-test"),
        ),
        GPT_41_MINI,
    ),
    Table(
        "qwen-agents",
        "Enterprise agents with a Qwen3.5-9B system model",
        ("claim_history", "product_questions_agent"),
        (
            ("Baseline", "baseline-test"),
            ("ReAPO (GPT-5 prompts)", "reapo__gpt-5__transfer-test"),
            ("ReAPO (Qwen3.5-9B prompts)", "reapo__qwen3.5-9b-test"),
        ),
        QWEN_9B,
    ),
)
TABLES_BY_NAME = {t.name: t for t in TABLES}


@dataclass(frozen=True)
class Run:
    """One testbed invocation."""

    label: str
    argv: tuple[str, ...]
    env: dict[str, str]
    summary: str

    @property
    def cells(self) -> tuple[str, ...]:
        """The ``agent:version`` cells the run evaluates (its ``--agents``)."""
        start = self.argv.index("--agents") + 1
        end = next(i for i in range(start, len(self.argv)) if self.argv[i].startswith("--"))
        return self.argv[start:end]


def versions_of(agent: str) -> set[str]:
    path = AGENTS_DIR / agent / "versions.yaml"
    versions = (yaml.safe_load(path.read_text()) if path.is_file() else None) or {}
    return set(versions)


def plan(
    table: Table,
    *,
    model: str | None = None,
    agents: Sequence[str] | None = None,
    versions: Sequence[str] | None = None,
    runs: int | None = None,
    skip: Sequence[str] = (),
    output_dir: Path = OUTPUT_DIR,
    experiment_name: str | None = None,
    max_concurrency: int = 8,
) -> tuple[list[Run], list[str]]:
    """The testbed runs for ``table``, and the (agent, version) cells left out because the agent
    has no such version.

    ``agents`` and ``versions`` narrow the table; ``skip`` drops single ``agent:version`` cells
    (e.g. ones already evaluated).
    """
    model = model or table.model or os.environ.get(QWEN_122B_MODEL_ENV, "")
    if not model:
        raise ValueError(
            f"{table.name} needs a Qwen3.5-122B-A10B endpoint: set {QWEN_122B_MODEL_ENV} or pass "
            "--model, and set AGENTS_GYM_MODEL_BASE_URL (docs/REPRODUCING.md)"
        )
    agents = list(agents or table.agents)
    if unknown := set(agents) - set(table.agents):
        raise ValueError(f"{table.name} has no agents {sorted(unknown)}; it has {table.agents}")
    table_versions = tuple(v for _, v in table.rows)
    rows = list(table_versions) if versions is None else list(versions)
    agent_versions = {agent: versions_of(agent) for agent in agents}
    # A version outside the table's rows is accepted when an agent has it (side experiments).
    if unknown := set(rows) - set(table_versions).union(*agent_versions.values()):
        raise ValueError(
            f"{table.name} has no versions {sorted(unknown)} for {agents}; "
            f"its rows are {table_versions}"
        )
    if stray := set(skip) - {f"{a}:{v}" for a in agents for v in rows}:
        raise ValueError(f"--skip names cells outside this run: {sorted(stray)}")
    name = experiment_name or table.name
    groups: dict[tuple[int, int | None], list[str]] = {}
    missing: list[str] = []
    for agent in agents:
        available = agent_versions[agent]
        cells = [f"{agent}:{v}" for v in rows if f"{agent}:{v}" not in skip]
        specs = [c for c in cells if c.split(":", 1)[1] in available]
        missing += [c for c in cells if c not in specs]
        if not specs:
            continue
        limit = (
            FORCED_CALL_LIMITS.get(agent, FORCED_CALL_LIMIT)
            if table.loop == "basic:auto-graceful"
            else None
        )
        n = runs or table.runs.get(agent, DEFAULT_RUNS)
        groups.setdefault((n, limit), []).extend(specs)

    planned = []
    for (n, limit), specs in groups.items():
        label = (
            name + (f"_{limit}calls" if limit else "") + (f"_{n}runs" if n != DEFAULT_RUNS else "")
        )
        env = {
            "UIPATH_AGENTS_GYM_TEMPERATURE": EVAL_TEMPERATURE,
            "UIPATH_AGENTS_GYM_LOOP_RECURSION_LIMIT": str(limit or CALL_LIMIT),
        }
        argv = [sys.executable, "-m", "agents_gym_core.testbed_core", "--agents", *specs]
        if table.loop:
            argv += ["--loops", table.loop]
        argv += [
            "--models", model,
            "--num_runs", str(n),
            "--aggregation_mode", AGGREGATION,
            "--max_concurrency", str(max_concurrency),
            "--output_dir", str(output_dir),
            "--experiment_name", label,
            "--mode", "fill_skip_run_errors",
        ]  # fmt: skip
        summary = f"model={model}, runs={n}" + (f", loop={table.loop}" if table.loop else "")
        summary += f", call limit={limit}" if limit else ""
        planned.append(Run(label, tuple(argv), env, summary))
    return planned, missing


def _results_dirs(experiment_dir: Path, cells: Sequence[str]) -> list[Path]:
    """The ``results`` directories of the ``agent:version`` cells under ``experiment_dir``.

    Only the loop keys ``final_score_per_agent.json`` lists for a cell are read, so runs left by
    an earlier model name, which fill mode does not run again, are left out.
    """
    try:
        summary = json.loads((experiment_dir / "final_score_per_agent.json").read_text())
    except (OSError, ValueError):
        summary = {}
    roots: list[Path] = []
    for cell in cells:
        cell_dir = experiment_dir / Path(*cell.split(":", 1))
        agent_key = cell.replace(":", "__", 1)
        loop_keys = {
            loop_key
            for loop_key, agents in (summary.items() if isinstance(summary, dict) else ())
            if isinstance(agents, dict) and agent_key in agents
        }
        if not loop_keys or not cell_dir.is_dir():
            roots.append(cell_dir)
            continue
        roots += [d / key for d in cell_dir.iterdir() if d.is_dir() for key in loop_keys]
    return [d for root in roots if root.is_dir() for d in root.rglob("results")]


def endpoint_failures(experiment_dir: Path, cells: Sequence[str]) -> Counter[str]:
    """Datapoint runs of the ``agent:version`` cells under ``experiment_dir`` whose model endpoint
    had an outage, counted by exception type (see :func:`_results_dirs`).
    """
    from agents_gym_core.runtime.run_utils import is_endpoint_error

    failures: Counter[str] = Counter()
    for results_dir in _results_dirs(experiment_dir, cells):
        for result_file in sorted(results_dir.glob("*.json")):
            try:
                agent_output = json.loads(result_file.read_text()).get("agent_output")
            except (OSError, ValueError, AttributeError):
                continue
            if is_endpoint_error(agent_output):
                failures[agent_output["title"]] += 1
                break
    return failures


def evaluator_exceptions(experiment_dir: Path, cells: Sequence[str]) -> int:
    """Evaluations of the ``agent:version`` cells under ``experiment_dir`` whose evaluator raised
    (saved as score 0) and that fill_skip_run_errors evaluates again: those of agent runs that
    did not fail (see :func:`_results_dirs`).
    """
    from agents_gym_core.runtime.run_utils import EVALUATOR_EXCEPTION_PREFIX

    count = 0
    for results_dir in _results_dirs(experiment_dir, cells):
        for result_file in results_dir.glob("*.json"):
            try:
                result = json.loads(result_file.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(result, dict):
                continue
            agent_output = result.get("agent_output")
            if isinstance(agent_output, dict) and agent_output.get("type") == "error":
                continue
            details = result.get("details")
            if isinstance(details, str) and details.startswith(EVALUATOR_EXCEPTION_PREFIX):
                count += 1
    return count


SUMMARY_FILES = (
    "final_score_per_agent.json",
    "combined_results.json",
    "agg_results_on_metrics.csv",
    "agg_results_on_datapoints.csv",
)


def _read_summaries(experiment_dir: Path) -> dict[str, str]:
    return {
        name: (experiment_dir / name).read_text()
        for name in SUMMARY_FILES
        if (experiment_dir / name).is_file()
    }


def _json_or_none(text: str) -> Any:
    """``text`` parsed, or None for a file cut off while it was written."""
    try:
        return json.loads(text)
    except ValueError:
        return None


def merge_summaries(experiment_dir: Path, earlier: dict[str, str], cells: Sequence[str]) -> None:
    """Put back the entries of the cells a run did not evaluate.

    The testbed rewrites the summary files with the cells it evaluated only, so a run narrowed
    with --agents, --versions or --skip would leave the table's summary with just those cells.
    Entries of ``cells`` come from the new files; those of every other cell from ``earlier``.
    """
    import io

    import pandas as pd

    from agents_gym_core.runtime.runners.result_aggregator import final_scores_json

    keys = {cell.replace(":", "__", 1) for cell in cells}
    for name, before in earlier.items():
        path = experiment_dir / name
        if not path.is_file() or (after := path.read_text()) == before:
            continue  # not rewritten by this run
        # A file is written back only when it keeps entries of other cells.
        if name == "final_score_per_agent.json":
            if (old_scores := _json_or_none(before)) is None:
                continue
            merged, kept = json.loads(after), False
            for loop_key, agents in old_scores.items():
                for agent_key, scores in agents.items():
                    if agent_key not in keys and agent_key not in merged.get(loop_key, {}):
                        merged.setdefault(loop_key, {})[agent_key] = scores
                        kept = True
            if kept:
                path.write_text(final_scores_json(merged))
        elif name == "combined_results.json":
            if (old_rows := _json_or_none(before)) is None:
                continue
            kept_rows = [
                row
                for row in old_rows
                if f"{row.get('agent')}__{row.get('agent_version')}" not in keys
            ]
            if kept_rows:
                path.write_text(json.dumps(json.loads(after) + kept_rows, indent=2))
        else:
            # A last line without its newline was cut off while the file was written.
            before = before if before.endswith("\n") else before[: before.rfind("\n") + 1]
            try:
                old = pd.read_csv(io.StringIO(before))
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue
            if not {"agent", "agent_version"} <= set(old.columns):
                continue
            old = old[~(old["agent"] + "__" + old["agent_version"]).isin(keys)]
            if old.empty:
                continue
            try:
                new = pd.read_csv(io.StringIO(after))
            except pd.errors.EmptyDataError:
                new = pd.DataFrame()
            pd.concat([new, old], ignore_index=True).to_csv(path, index=False)


def execute(runs: list[Run], output_dir: Path, dry_run: bool = False) -> int:
    """Run every group; a failing group does not stop the others, and the first failure's
    exit code is returned."""
    failed = 0
    for run in runs:
        env = " ".join(f"{k}={v}" for k, v in run.env.items())
        print(f"\n# {run.label}: {run.summary}")
        print(f"{env} {shlex.join(run.argv)}")
        if dry_run:
            continue
        earlier = _read_summaries(output_dir / run.label)
        code = subprocess.run(run.argv, env={**os.environ, **run.env}).returncode
        merge_summaries(output_dir / run.label, earlier, run.cells)
        if raised := evaluator_exceptions(output_dir / run.label, run.cells):
            print(
                f"  {raised} evaluation{'s' if raised != 1 else ''} failed because the evaluator "
                "raised; they are scored zero for now. Rerun the same command: fill mode "
                "evaluates them again.",
                file=sys.stderr,
            )
            code = code or 1
        elif not code and (failures := endpoint_failures(output_dir / run.label, run.cells)):
            counts = ", ".join(f"{title} x{n}" for title, n in failures.most_common())
            total = failures.total()
            print(
                f"  {total} run{'s' if total != 1 else ''} failed because the model endpoint "
                f"had an outage ({counts}); they are scored zero for now. "
                "Rerun the same command once the endpoint works: only those runs run again.",
                file=sys.stderr,
            )
            code = 1
        elif not code:
            print(f"  scores: {output_dir}/{run.label}/final_score_per_agent.json")
        failed = failed or code
    return failed


def markdown() -> str:
    """The tables, as listed in docs/REPRODUCING.md."""
    lines = [
        "| Command | Paper table | System model | Agents | Rows (version) |",
        "|---|---|---|---|---|",
    ]
    for t in TABLES:
        rows = "<br>".join(f"{label}: `{v}`" for label, v in t.rows)
        runs = "".join(f"; {a}: {n} runs" for a, n in t.runs.items())
        agents = ", ".join(t.agents) + runs
        model = (
            f"`{t.model}`" if t.model else f"your endpoint, `{QWEN_122B_MODEL_ENV}` or `--model`"
        )
        lines.append(f"| `gym reproduce {t.name}` | {t.title} | {model} | {agents} | {rows} |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None, *, prog: str = "gym reproduce") -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog=prog,
        description="Evaluate the paper's prompts on the test splits, one table at a time.",
    )
    parser.add_argument("table", nargs="?", choices=list(TABLES_BY_NAME), metavar="table")
    parser.add_argument("--list", action="store_true", help="list the tables")
    parser.add_argument("--agents", nargs="+", help="only these agents of the table")
    parser.add_argument("--versions", nargs="+", help="only these versions (rows)")
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        metavar="AGENT:VERSION",
        help="leave out these cells, e.g. ones already evaluated",
    )
    parser.add_argument("--model", help="system model (default: the table's)")
    parser.add_argument("--runs", type=int, help="runs per cell (default: the table's, mostly 5)")
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--experiment-name", help="default: the table's name")
    parser.add_argument("--dry-run", action="store_true", help="print the commands only")
    args = parser.parse_args(argv)
    if args.runs is not None and args.runs < 1:
        parser.error("--runs must be at least 1")

    if args.list or not args.table:
        if not args.table and not args.list:
            print("gym reproduce: name a table:", file=sys.stderr)
        for t in TABLES:
            print(f"  {t.name:<22} {t.title}")
        return 0 if args.list else 2
    table = TABLES_BY_NAME[args.table]
    try:
        runs, missing = plan(
            table,
            model=args.model,
            agents=args.agents,
            versions=args.versions,
            runs=args.runs,
            skip=args.skip,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            max_concurrency=args.max_concurrency,
        )
    except ValueError as error:
        parser.error(str(error))
    print(f"{table.title}" + (f"\n  {table.note}" if table.note else ""))
    if missing:
        print(f"  no such version, not run: {', '.join(missing)}")
    if not runs:
        print("  nothing to run")
        return 0
    return execute(runs, args.output_dir, args.dry_run)
