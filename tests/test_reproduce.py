"""`gym reproduce`: each paper table plans the testbed runs with the paper's settings."""

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from agents_gym_core.reproduce import (
    QWEN_122B_MODEL_ENV,
    TABLES,
    TABLES_BY_NAME,
    Table,
    markdown,
    plan,
)
from agents_gym_core.runtime.run_utils import (
    RunMode,
    get_datapoint_dir,
    should_run_datapoint,
    should_run_evaluator,
)

REPO = Path(__file__).resolve().parents[1]

# Cells a table names that have no prompt; each table's note (or REPRODUCING) explains why.
GAPS = {
    "main-mini": {"aime_2025_agent:gepa-merge__gpt-4.1-mini-test"},
    "tool-ablation": {
        "hotpotqa:reapo__gpt-4.1-mini__score-only-test",
        "livebench_math:reapo__gpt-4.1-mini__score-only-test",
    },
}


def _specs(argv: tuple[str, ...]) -> list[str]:
    start = argv.index("--agents") + 1
    end = next(i for i in range(start, len(argv)) if argv[i].startswith("--"))
    return list(argv[start:end])


def _flag(argv: tuple[str, ...], name: str) -> str:
    return argv[argv.index(name) + 1]


# Tables without a public default system model take the caller's endpoint.
MODEL = "my-qwen3.5-122b"


@pytest.mark.parametrize("table", TABLES, ids=lambda t: t.name)
def test_every_cell_is_a_version_or_a_known_gap(table: Table) -> None:
    runs, missing = plan(table, model=table.model or MODEL)
    assert set(missing) == GAPS.get(table.name, set())
    cells = {f"{a}:{v}" for a in table.agents for _, v in table.rows}
    assert sorted(s for r in runs for s in _specs(r.argv)) == sorted(cells - set(missing))
    for run in runs:
        assert run.env["UIPATH_AGENTS_GYM_TEMPERATURE"] == "1.0"
        assert _flag(run.argv, "--models") == (table.model or MODEL)
        assert _flag(run.argv, "--mode") == "fill_skip_run_errors"


def test_reproduction_preserves_scored_failures_when_resuming(tmp_path: Path) -> None:
    (run,), _ = plan(TABLES_BY_NAME["agents"])
    mode = cast(RunMode, _flag(run.argv, "--mode"))
    datapoint_dir = get_datapoint_dir(tmp_path, "failed", 1)
    results_dir = datapoint_dir / "results"
    results_dir.mkdir(parents=True)
    (datapoint_dir / "output.json").write_text("[]")
    (results_dir / "judge.json").write_text(
        json.dumps({"score": 0.0, "agent_output": {"type": "error"}})
    )

    assert should_run_datapoint(tmp_path, "failed", "fill", 1)
    assert should_run_evaluator(tmp_path, "failed", "judge", "fill", 1)
    assert not should_run_datapoint(tmp_path, "failed", mode, 1)
    assert not should_run_evaluator(tmp_path, "failed", "judge", mode, 1)
    assert should_run_datapoint(tmp_path, "missing", mode, 1)
    assert should_run_evaluator(tmp_path, "missing", "judge", mode, 1)


@pytest.mark.parametrize("table", TABLES, ids=lambda t: t.name)
def test_each_table_is_scored_as_in_the_paper(table: Table) -> None:
    runs, _ = plan(table, model=table.model or MODEL)
    # Eqs. 1-2 of the paper: the evaluator mean per datapoint, then the mean over datapoints.
    assert {_flag(r.argv, "--aggregation_mode") for r in runs} == {"datapoint_run"}


def test_the_122b_table_needs_the_callers_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(QWEN_122B_MODEL_ENV, raising=False)
    with pytest.raises(ValueError, match=f"{QWEN_122B_MODEL_ENV} or pass --model"):
        plan(TABLES_BY_NAME["self-opt-122b"])


def test_the_122b_table_takes_the_model_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(QWEN_122B_MODEL_ENV, MODEL)
    runs, _ = plan(TABLES_BY_NAME["self-opt-122b"])
    assert {_flag(r.argv, "--models") for r in runs} == {MODEL}


def test_enterprise_main_table_uses_each_agents_loop() -> None:
    (agents,), _ = plan(TABLES_BY_NAME["agents"])
    assert "--loops" not in agents.argv and _flag(agents.argv, "--num_runs") == "5"


def test_forced_final_answer_runs_once_per_call_limit() -> None:
    runs, _ = plan(TABLES_BY_NAME["agents-forced"])
    by_limit = {r.env["UIPATH_AGENTS_GYM_LOOP_RECURSION_LIMIT"]: r for r in runs}
    assert {a.split(":")[0] for a in _specs(by_limit["12"].argv)} == {"sql", "meeting_management"}
    assert "sql" not in {a.split(":")[0] for a in _specs(by_limit["6"].argv)}
    assert all(_flag(r.argv, "--loops") == "basic:auto-graceful" for r in runs)
    assert {r.label for r in runs} == {"agents-forced_6calls", "agents-forced_12calls"}


def test_qwen_agents_use_each_agents_own_loop() -> None:
    (run,), _ = plan(TABLES_BY_NAME["qwen-agents"])
    assert "--loops" not in run.argv
    assert run.env["UIPATH_AGENTS_GYM_LOOP_RECURSION_LIMIT"] == "20"
    assert run.label == "qwen-agents"


def test_aime_transfer_gets_its_own_ten_run_experiment() -> None:
    runs, _ = plan(TABLES_BY_NAME["transfer"])
    by_label = {r.label: r for r in runs}
    assert _flag(by_label["transfer_10runs"].argv, "--num_runs") == "10"
    assert {a.split(":")[0] for a in _specs(by_label["transfer_10runs"].argv)} == {
        "aime_2025_agent"
    }
    assert "aime_2025_agent:baseline-test" not in _specs(by_label["transfer"].argv)


def test_agents_versions_and_skip_narrow_a_table() -> None:
    table = TABLES_BY_NAME["transfer"]
    done = "hover:reapo__gpt-4.1-mini-test"
    (run,), _ = plan(table, agents=["hover", "ifbench"], skip=[done], runs=1)
    assert _specs(run.argv) == [
        "hover:baseline-test",
        "hover:gepa__gpt-4.1-mini-test",
        "ifbench:baseline-test",
        "ifbench:gepa__gpt-4.1-mini-test",
        "ifbench:reapo__gpt-4.1-mini-test",
    ]
    assert run.label == "transfer_1runs"
    (run,), _ = plan(table, agents=["hover"], versions=["gepa__gpt-4.1-mini-test"], model="m")
    assert (
        _specs(run.argv) == ["hover:gepa__gpt-4.1-mini-test"] and _flag(run.argv, "--models") == "m"
    )
    with pytest.raises(ValueError, match="no agents"):
        plan(table, agents=["sql"])
    with pytest.raises(ValueError, match="outside this run"):
        plan(table, agents=["ifbench"], skip=[done])


def test_cli_prints_the_commands_without_running_them(tmp_path: Path) -> None:
    run = subprocess.run(
        [sys.executable, "-m", "agents_gym_core.sdk.cli", "reproduce", "qwen-agents",
         "--dry-run", "--output-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=120, check=True,
    )  # fmt: skip
    assert "--agents claim_history:baseline-test" in run.stdout
    assert "--experiment_name qwen-agents" in run.stdout and not any(tmp_path.iterdir())
    listing = subprocess.run(
        [sys.executable, "-m", "agents_gym_core.sdk.cli", "reproduce", "--list"],
        capture_output=True, text=True, timeout=120, check=True,
    ).stdout  # fmt: skip
    assert all(t.name in listing for t in TABLES)


def test_reproducing_lists_the_tables() -> None:
    assert markdown() in (REPO / "docs" / "REPRODUCING.md").read_text()
