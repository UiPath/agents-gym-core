"""In an optimization run the tools evaluate only the target version and hide its other splits."""

import json
from pathlib import Path

import pytest

from agents_gym_core.paths import AGENTS_DIR
from agents_gym_core.prompt_optimization.mcp_tools import (
    helpers,
    tools_agents,
    tools_datapoints,
    tools_experiments,
    tools_runner,
)
from agents_gym_core.prompt_optimization.mcp_tools.helpers import (
    TARGET_ENV,
    record_started_experiment,
)

EVAL_SETS = AGENTS_DIR / "claim_history" / "evaluations" / "eval-sets"


def _names(file: str) -> list[str]:
    return [e["name"] for e in json.loads((EVAL_SETS / file).read_text())["evaluations"]]


TRAIN = _names("evaluation-set-claim_history_train.json")
TEST = _names("evaluation-set-claim_history_test.json")


@pytest.fixture
def pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TARGET_ENV, "claim_history:train")


def test_only_the_target_split_is_readable(pinned: None) -> None:
    train = tools_agents.read_agent_file(
        "claim_history:train", "evaluations/eval-sets/evaluation-set-claim_history_train.json"
    )
    assert TRAIN[0] in train
    # The test split, and the base set that contains it, are hidden.
    for hidden in ("evaluation-set-claim_history_test.json", "evaluation-set-claim_history.json"):
        reply = tools_agents.read_agent_file("claim_history", f"evaluations/eval-sets/{hidden}")
        assert reply.startswith("Access denied"), reply
        assert TEST[0] not in reply


def test_datapoint_lookup_never_returns_or_lists_other_splits(pinned: None) -> None:
    assert (
        json.loads(tools_datapoints.get_datapoint("claim_history:train", TRAIN[0]))["name"]
        == (TRAIN[0])
    )
    reply = tools_datapoints.get_datapoint("claim_history:train", TEST[0])
    assert reply.startswith(f"Datapoint '{TEST[0]}' not found")
    assert not any(name in reply.split("\n", 1)[1] for name in TEST)


def test_unpinned_runs_are_unrestricted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TARGET_ENV, raising=False)
    test_file = "evaluations/eval-sets/evaluation-set-claim_history_test.json"
    assert TEST[0] in tools_agents.read_agent_file("claim_history", test_file)


def test_a_pinned_run_cannot_read_or_edit_another_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TARGET_ENV, "calculator:base")
    prompt_file = AGENTS_DIR / "claim_history" / "prompts" / "system_prompt.md"
    before = prompt_file.read_text()
    for reply in (
        tools_agents.read_agent_file("claim_history", "agent.yaml"),
        tools_agents.get_prompt("claim_history"),
        tools_agents.summarize_agent("claim_history"),
        tools_agents.update_prompt("claim_history", "overwritten"),
        tools_datapoints.get_datapoint("claim_history", TRAIN[0]),
    ):
        assert reply.startswith("Access denied: this optimization run is limited to 'calculator'")
    assert prompt_file.read_text() == before
    # The target itself is still readable.
    assert "calculator" in tools_agents.read_agent_file("calculator", "agent.yaml")


def test_runs_are_limited_to_the_target_spec(pinned: None, monkeypatch: pytest.MonkeyPatch) -> None:
    for spec in ("claim_history:test", "claim_history", "claim_history:train calculator"):
        assert tools_runner.run_agent_async(spec).startswith("Refused"), spec
    # Extra testbed flags cannot ride in on the other arguments.
    run = tools_runner.run_agent_async
    assert run("claim_history:train", loops="basic --agents claim_history:test").startswith("Ref")
    assert run("claim_history:train", datapoints="--agents claim_history:test").startswith("Ref")
    cmd = tools_runner._build_cmd(["claim_history:train"], 0, "--agents=claim_history:test", "e")
    assert cmd.count("--agents") == 1 and "--models=--agents=claim_history:test" in cmd

    def no_process(*args: object, **kwargs: object) -> None:
        raise OSError("not started in tests")

    monkeypatch.setattr(tools_runner.subprocess, "Popen", no_process)
    reply = tools_runner.run_agent_async("claim_history:train", experiment_name="t_target")
    assert reply == "Error starting agent run: not started in tests"


def _experiment(root: Path, name: str, version: str) -> None:
    run = root / name / "claim_history" / version / "basic" / "dp" / "run_1"
    (run / "results").mkdir(parents=True)
    (run / "output.json").write_text("[]")
    (run / "results" / "judge.json").write_text(json.dumps({"datapoint_name": "dp", "score": 1}))
    (root / name / "combined_results.json").write_text("[]")


def test_only_experiments_the_run_started_are_readable(
    pinned: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    experiments = tmp_path / "experiments"
    monkeypatch.setattr(tools_experiments, "experiments_dir", lambda: experiments)
    monkeypatch.setattr(tools_runner, "experiments_dir", lambda: experiments)
    monkeypatch.setattr(helpers, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tools_runner, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("COACH_OUTPUT_DIR", "tmp_workspace")
    _experiment(experiments, "react_opt_round_000_001", "train")
    _experiment(experiments, "other_eval", "test")
    record_started_experiment("react_opt_round_000_001")
    # The optimizer can neither write the list of started experiments nor smuggle a name in.
    registry = helpers.STARTED_EXPERIMENTS_FILE
    for name in (registry, registry.upper(), f"round_000/../{registry}", f"./{registry}"):
        reply = tools_runner.save_optimizer_log("tmp_workspace", name, '["other_eval"]')
        assert reply.startswith("Refused"), reply
    reply = tools_runner.run_agent_async("claim_history:train", experiment_name="x\nother_eval")
    assert reply.startswith("Refused"), reply
    # A run named like an existing experiment in another case gets a fresh number, and an
    # experiment that also holds another version's results stays hidden.
    _experiment(experiments, "other_eval_001", "test")
    assert tools_runner._next_experiment_name("OTHER_EVAL") == "OTHER_EVAL_002"
    record_started_experiment("other_eval_001")
    assert tools_experiments.get_full_results("other_eval_001").startswith("Access denied")

    listing = tools_experiments.list_experiments()
    assert "react_opt_round_000_001" in listing and "other_eval" not in listing
    assert "Per-Datapoint" in tools_experiments.get_full_results("react_opt_round_000_001")
    for reply in (
        tools_experiments.list_experiments("other_eval"),
        tools_experiments.get_full_results("other_eval"),
        tools_experiments.get_experiment_results("other_eval", "claim_history"),
        tools_experiments.get_agent_messages("other_eval", "claim_history", "dp"),
        # a sub-path of an experiment, or its absolute path, is no way around it
        tools_experiments.get_experiment_results("other_eval/claim_history", "test"),
        tools_experiments.get_agent_messages("other_eval/claim_history", "test", "dp"),
    ):
        assert reply.startswith("Access denied"), reply
    absolute = str(experiments / "other_eval")
    for reply in (
        tools_experiments.list_experiments(absolute),
        tools_experiments.get_full_results(absolute),
        tools_experiments.get_agent_messages(absolute, "claim_history", "dp"),
        tools_experiments.get_agent_messages("react_opt_round_000_001", "../other_eval", "dp"),
    ):
        assert reply.startswith("Invalid"), reply

    monkeypatch.delenv(TARGET_ENV)
    assert "Per-Datapoint" in tools_experiments.get_full_results("other_eval")


def test_another_spelling_of_a_hidden_file_is_still_hidden(pinned: None) -> None:
    for path in (
        "evaluations/eval-sets/EVALUATION-SET-claim_history_test.json",
        "evaluations//eval-sets/./evaluation-set-claim_history_test.json",
    ):
        assert TEST[0] not in tools_agents.read_agent_file("claim_history", path)


def test_an_unknown_skill_fails_before_the_prompts_are_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agents_gym_core.prompt_optimization import main, reset_prompts

    calls: list[str] = []
    monkeypatch.setattr(reset_prompts, "reset_prompts", lambda name, **_: calls.append(name))
    monkeypatch.setattr("sys.argv", ["gym optimize", "calculator", "-s", "unknown", "--reset"])
    with pytest.raises(SystemExit):
        main.main()
    assert calls == []


def test_other_spellings_of_the_target_agent_are_still_pinned(pinned: None) -> None:
    test_file = "evaluations/eval-sets/evaluation-set-claim_history_test.json"
    for name in ("claim_history/prompts/..", "claim_history/../claim_history", "CLAIM_HISTORY"):
        assert TEST[0] not in tools_agents.read_agent_file(name, test_file), name
        listing = tools_datapoints.get_datapoint(name, TEST[0]).split("\n", 1)[-1]  # after the echo
        assert not any(test_name in listing for test_name in TEST), name


def test_round_stats_are_read_once_when_an_old_workspace_is_continued(
    pinned: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(helpers, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(tools_runner, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("COACH_OUTPUT_DIR", "tmp_workspace")
    workspace = tmp_path / "tmp_workspace"
    (workspace / "round_000").mkdir(parents=True)
    (workspace / "round_000" / "round_stats.json").write_text(
        '{"experiment_name": "react_opt_round_000_001"}'
    )
    helpers.init_started_experiments(workspace)  # what the optimizer does before it starts
    assert helpers.started_experiments() == {"react_opt_round_000_001"}

    # the optimizer cannot write round_stats.json, under any spelling
    for name in (
        "round_001/round_stats.json",
        "round_001/ROUND_STATS.JSON",
        "round_001/round_stats.json.",
    ):
        reply = tools_runner.save_optimizer_log("tmp_workspace", name, '{"experiment_name": "x"}')
        assert reply == "Refused: round_stats.json is written by run_agent_async.", reply
    # and one that appears some other way afterwards is not trusted
    (workspace / "round_001").mkdir()
    (workspace / "round_001" / "round_stats.json").write_text('{"experiment_name": "other_eval"}')
    helpers.init_started_experiments(workspace)
    assert helpers.started_experiments() == {"react_opt_round_000_001"}


def test_reset_dry_run_succeeds_for_a_known_agent() -> None:
    from agents_gym_core.prompt_optimization.reset_prompts import reset_prompts

    assert reset_prompts("hover", dry_run=True)
    assert not reset_prompts("no_such_agent", dry_run=True)


def test_a_refused_workspace_leaves_the_prompts_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from agents_gym_core.prompt_optimization import main, reset_prompts

    (tmp_path / "tmp_existing").mkdir()
    monkeypatch.setattr(main, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(main, "compute_output_dir", lambda *a, **k: "tmp_existing")
    resets: list[str] = []

    def fake_reset(name: str, dry_run: bool = False, **kwargs: object) -> bool:
        assert kwargs["directory_provider"] is helpers.directory_provider()
        if not dry_run:  # the up-front check only reads
            resets.append(name)
        return True

    monkeypatch.setattr(reset_prompts, "reset_prompts", fake_reset)
    monkeypatch.setattr("sys.argv", ["gym optimize", "calculator", "--reset"])
    with pytest.raises(SystemExit, match="pass -c to continue it or -o"):
        main.main()
    assert resets == []


def test_check_evals_is_limited_to_the_target_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TARGET_ENV, "calculator:base")
    reply = tools_datapoints.check_evals("claim_history", source="core")
    assert reply.startswith("Access denied: this optimization run is limited to 'calculator'")
    assert tools_datapoints.check_evals("../claim_history", source="core").startswith("Agent ")
    # With no agent named, a pinned run checks only its target.
    assert "claim_history" not in tools_datapoints.check_evals("", source="core")


def test_the_eval_set_checker_refuses_a_path_as_agent_name(tmp_path: Path) -> None:
    from agents_gym_core.prompt_optimization.check_eval_sets import _discover_agent_dirs

    (tmp_path / "agents").mkdir()
    with pytest.raises(SystemExit):
        _discover_agent_dirs(tmp_path / "agents", "../agents")
