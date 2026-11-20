"""Synthetic containment regressions for Core's MCP agent-file helpers."""

import ast
import json
from pathlib import Path

import pytest
import yaml

from agents_gym_core.prompt_optimization.mcp_tools import helpers, tools_agents, tools_datapoints
from agents_gym_core.utils.directory_provider import CoreDirectoryProvider


@pytest.fixture
def agent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv(helpers.TARGET_ENV, raising=False)
    monkeypatch.delenv("COACH_OUTPUT_DIR", raising=False)
    provider = CoreDirectoryProvider()
    monkeypatch.setattr(provider, "get_project_root", lambda: tmp_path)
    monkeypatch.setattr(provider, "get_agents_dir", lambda: tmp_path / "core-agents")
    monkeypatch.setattr(helpers, "DIRECTORY_PROVIDER", provider)
    monkeypatch.setattr(tools_agents, "record_prompt_update", lambda *_: None)
    for source in ("gym", "core"):
        agent = helpers.directory_provider().agent_roots(source)[0] / "example"
        (agent / "prompts").mkdir(parents=True)
        (agent / "prompts/system.md").write_text("Synthetic prompt.")
        (agent / "agent.yaml").write_text(
            yaml.safe_dump(
                {
                    "metadata": {"description": "Synthetic agent"},
                    "agent": {
                        "system_prompt": f"{source}::prompts/system.md",
                        "user_prompt": "prompts/system.md",
                        "signature_prompts": {"reason": "prompts/system.md"},
                    },
                }
            )
        )
    return helpers.directory_provider().agent_roots("gym")[0] / "example"


@pytest.mark.parametrize("source", ["gym", "core", "all"])
def test_regular_files_and_prompts(agent_dir: Path, source: str) -> None:
    assert tools_agents.read_agent_file("example:v1", "prompts/system.md", source).endswith(
        "\n\nSynthetic prompt."
    )
    assert tools_agents.get_prompt("example", source=source) == "Synthetic prompt."
    assert "Synthetic prompt." in tools_agents.summarize_agent("example", source)
    assert tools_agents.update_prompt("example:v1", "Updated.", source=source).startswith("Updated")
    assert tools_agents.get_prompt("example", source=source) == "Updated."


@pytest.mark.parametrize("source", ["gym", "core", "all"])
@pytest.mark.parametrize("target", ["outside", "registry", "internal"])
def test_agent_directory_aliases(agent_dir: Path, tmp_path: Path, source: str, target: str) -> None:
    root = helpers.directory_provider().agent_roots("gym" if source == "all" else source)[0]
    outside = tmp_path / "outside-agent"
    outside.mkdir()
    (outside / "agent.yaml").write_text("metadata: {}\n")
    (outside / "private.txt").write_text("OUTSIDE_AGENT")
    destination = {"outside": outside, "registry": root, "internal": root / "example"}[target]
    (root / "alias").symlink_to(destination, target_is_directory=True)
    if target == "internal":
        assert helpers.resolve_agent_dir("alias:v1", source) is not None
        assert tools_agents.get_prompt("alias", source=source) == "Synthetic prompt."
    else:
        assert helpers.resolve_agent_dir("alias", source) is None
        for reply in (
            tools_agents.read_agent_file("alias", "private.txt", source),
            tools_agents.get_agent_files("alias", source),
            tools_agents.summarize_agent("alias", source),
        ):
            assert "not found" in reply and "OUTSIDE_AGENT" not in reply
            assert str(tmp_path) not in reply


@pytest.mark.parametrize(
    "relative",
    [
        "",
        "../agent.yaml",
        "prompts/../agent.yaml",
        "/outside",
        "C:secret",
        "C:/secret",
        "C:\\secret",
        "\\\\host\\share",
        "//host/share",
        "bad\x00name",
    ],
)
def test_invalid_paths_are_rejected_before_resolution(
    agent_dir: Path, monkeypatch: pytest.MonkeyPatch, relative: str
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> Path:
        pytest.fail("invalid paths must not reach filesystem resolution")

    monkeypatch.setattr(Path, "resolve", unexpected)
    assert helpers.resolve_agent_file(agent_dir, relative) is None
    assert helpers.resolve_prompt_path(agent_dir, f"core::{relative}") is None
    assert tools_agents.read_agent_file("example", relative).startswith("Invalid")


@pytest.mark.parametrize("namespace", ["", "..", "../core", "C:secret", "C:\\core", "bad\x00name"])
def test_invalid_prompt_namespaces_do_not_bypass_validation(
    agent_dir: Path, monkeypatch: pytest.MonkeyPatch, namespace: str
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> Path:
        pytest.fail("invalid namespaces must not reach filesystem resolution")

    monkeypatch.setattr(Path, "resolve", unexpected)
    assert helpers.resolve_prompt_path(agent_dir, f"{namespace}::prompts/system.md") is None


@pytest.mark.parametrize("namespace", ["gym", "core", "agents_gym_core"])
def test_valid_prompt_namespaces_remain_supported(agent_dir: Path, namespace: str) -> None:
    assert (
        helpers.resolve_prompt_path(agent_dir, f"{namespace}::prompts/system.md")
        == (agent_dir / "prompts/system.md").resolve()
    )


@pytest.mark.parametrize("kind", ["file", "directory", "loop"])
def test_file_escapes_and_loops_are_not_read_or_listed(
    agent_dir: Path, tmp_path: Path, kind: str
) -> None:
    outside = tmp_path / "outside-files"
    outside.mkdir()
    (outside / "private.md").write_text("OUTSIDE_FILE")
    link = agent_dir / "rejected-alias"
    link.symlink_to({"file": outside / "private.md", "directory": outside, "loop": link}[kind])
    relative = "rejected-alias/private.md" if kind == "directory" else "rejected-alias"
    reply = tools_agents.read_agent_file("example", relative)
    assert reply.startswith(("Refused", "File not found"))
    for result in (
        reply,
        tools_agents.get_agent_files("example"),
        tools_agents.read_agent_file("example", "missing.md"),
    ):
        assert "OUTSIDE_FILE" not in result
        assert str(tmp_path) not in result
    assert "rejected-alias" not in tools_agents.get_agent_files("example")
    assert "rejected-alias" not in tools_agents.read_agent_file("example", "missing.md")


def test_contained_file_and_directory_aliases_remain_usable(agent_dir: Path) -> None:
    (agent_dir / "alias.md").symlink_to("prompts/system.md")
    (agent_dir / "prompt-alias").symlink_to("prompts", target_is_directory=True)
    for relative in ("alias.md", "prompt-alias/system.md"):
        assert tools_agents.read_agent_file("example", relative).endswith("Synthetic prompt.")
    assert "alias.md" in tools_agents.get_agent_files("example")
    assert "alias.md" in tools_agents.read_agent_file("example", "missing.md")
    assert tools_agents.read_agent_file("example", "prompts") == "Not a regular file: prompts"


def test_external_file_alias_is_not_read_or_sized(
    agent_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outside = tmp_path / "external.txt"
    outside.write_text("SYNTHETIC_EXTERNAL")
    (agent_dir / "external-alias").symlink_to(outside)
    original_stat = Path.stat
    original_read = Path.read_text

    class ResolutionStat:
        @property
        def st_size(self) -> int:
            raise AssertionError("external targets must not be sized")

    def guarded_stat(path: Path, *, follow_symlinks: bool = True):
        # Python 3.11/3.12 resolve() checks stat() for loops but never uses its size.
        if path == outside:
            original_stat(path, follow_symlinks=follow_symlinks)
            return ResolutionStat()
        return original_stat(path, follow_symlinks=follow_symlinks)

    def guarded_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        assert path != outside, "external targets must not be read"
        return original_read(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "read_text", guarded_read)
    assert tools_agents.read_agent_file("example", "external-alias").startswith("Refused")
    assert "external-alias" not in tools_agents.get_agent_files("example")
    assert "external-alias" not in tools_agents.read_agent_file("example", "missing")
    assert tools_agents.read_agent_file("example", "prompts/system.md").endswith(
        "Synthetic prompt."
    )


@pytest.mark.parametrize("internal", [False, True])
def test_config_aliases_apply_to_reads_and_writes(
    agent_dir: Path, tmp_path: Path, internal: bool
) -> None:
    target = (agent_dir if internal else tmp_path) / "config.yaml"
    target.write_text("metadata:\n  description: CONFIG_SENTINEL\n")
    (agent_dir / "agent.yaml").unlink()
    (agent_dir / "agent.yaml").symlink_to(target)
    result = tools_agents.update_agent_yaml("example", "metadata.description", '"updated"', "gym")
    if internal:
        assert result.startswith("Updated")
        assert yaml.safe_load(target.read_text())["metadata"]["description"] == "updated"
        assert (agent_dir / "agent.yaml").is_symlink()
    else:
        assert not result.startswith("Updated")
        assert "CONFIG_SENTINEL" not in tools_agents.summarize_agent("example", "gym")
        assert "CONFIG_SENTINEL" in target.read_text()
        assert helpers.load_agent_config(agent_dir)[1] is not None


@pytest.mark.parametrize("prompt_type", ["system", "user", "reason"])
@pytest.mark.parametrize(
    "kind", ["traversal", "absolute", "windows", "nul", "file", "directory", "loop"]
)
def test_prompt_read_preview_and_update_share_containment(
    agent_dir: Path, tmp_path: Path, prompt_type: str, kind: str
) -> None:
    victim = tmp_path / "victim.md"
    victim.write_text("OUTSIDE_PROMPT")
    (agent_dir / "linked.md").symlink_to(victim)
    (agent_dir / "linked-dir").symlink_to(tmp_path, target_is_directory=True)
    (agent_dir / "loop.md").symlink_to("loop.md")
    references = {
        "traversal": "../../victim.md",
        "absolute": str(victim),
        "windows": "C:\\victim.md",
        "nul": "bad\x00name",
        "file": "linked.md",
        "directory": "linked-dir/victim.md",
        "loop": "loop.md",
    }
    ref = f"core::{references[kind]}"
    config = yaml.safe_load((agent_dir / "agent.yaml").read_text())
    if prompt_type == "reason":
        config["agent"]["signature_prompts"][prompt_type] = ref
    else:
        config["agent"][f"{prompt_type}_prompt"] = ref
    (agent_dir / "agent.yaml").write_text(yaml.safe_dump(config))
    for result in (
        tools_agents.get_prompt("example", prompt_type),
        tools_agents.update_prompt("example", "OVERWRITE", prompt_type),
        tools_agents.summarize_agent("example"),
    ):
        assert "OUTSIDE_PROMPT" not in result
        assert str(tmp_path) not in result
    assert victim.read_text() == "OUTSIDE_PROMPT"
    assert (agent_dir / "prompts/system.md").read_text() == "Synthetic prompt."


def test_internal_prompt_alias_updates_target_atomically(agent_dir: Path) -> None:
    (agent_dir / "alias.md").symlink_to("prompts/system.md")
    tools_agents.update_agent_yaml("example", "agent.system_prompt", '"core::alias.md"')
    (agent_dir / "prompts/system.md").chmod(0o640)
    assert tools_agents.update_prompt("example", "Updated target.").startswith("Updated")
    assert (agent_dir / "alias.md").is_symlink()
    assert (agent_dir / "prompts/system.md").read_text() == "Updated target."
    assert (agent_dir / "prompts/system.md").stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize("kind", ["file", "directory", "internal-file", "internal-directory"])
def test_evaluator_aliases_are_contained_even_in_fallbacks(
    agent_dir: Path, tmp_path: Path, kind: str
) -> None:
    internal = kind.startswith("internal")
    target_dir = (agent_dir if internal else tmp_path) / "judge-files"
    target_dir.mkdir()
    target = target_dir / "evaluator-judge.json"
    target.write_text(json.dumps({"id": "JUDGE_SENTINEL", "evaluatorConfig": {"prompt": "old"}}))
    directory = agent_dir / "evaluations/evaluators"
    directory.parent.mkdir()
    if kind.endswith("directory"):
        directory.symlink_to(target_dir, target_is_directory=True)
    else:
        directory.mkdir()
        (directory / target.name).symlink_to(target)
    result = tools_datapoints.update_evaluator_prompt("example", "JUDGE_SENTINEL", "new")
    assert result.startswith("Updated") is internal
    assert (json.loads(target.read_text())["evaluatorConfig"]["prompt"] == "new") is internal
    for result in (
        tools_agents.inspect_agent("example"),
        tools_datapoints.update_evaluator_prompt("example", "missing", "new"),
    ):
        assert ("JUDGE_SENTINEL" in result) is internal
        assert str(tmp_path) not in result


@pytest.mark.parametrize(
    "kind", ["absolute", "traversal", "windows", "nul", "file", "directory", "root", "internal"]
)
def test_tool_data_reads_and_extra_listing_are_contained(
    agent_dir: Path, tmp_path: Path, kind: str
) -> None:
    outside = tmp_path / "outside-tools"
    outside.mkdir()
    (outside / "data.json").write_text('{"TOOL_DATA_SENTINEL": 1}')
    root = agent_dir / "tools_data"
    if kind == "root":
        root.symlink_to(tmp_path, target_is_directory=True)
        name = "outside-tools"
    else:
        tool = root / "tool"
        tool.mkdir(parents=True)
        name = "tool"
        if kind == "absolute":
            name = str(outside)
        elif kind == "traversal":
            name = "../../../outside-tools"
        elif kind == "windows":
            name = "C:\\outside-tools"
        elif kind == "nul":
            name = "bad\x00name"
        elif kind == "file":
            (tool / "data.json").symlink_to(outside / "data.json")
        elif kind == "directory":
            tool.rmdir()
            tool.symlink_to(outside, target_is_directory=True)
        else:
            (tool / "data.json").write_text('{"TOOL_DATA_SENTINEL": 1}')
    (agent_dir / "agent.yaml").write_text(
        yaml.safe_dump({"agent": {"tools": [{"tool_name": name}]}})
    )
    assert ("key: TOOL_DATA_SENTINEL" in tools_agents.inspect_tools("example")) is (
        kind == "internal"
    )
    (agent_dir / "agent.yaml").write_text(yaml.safe_dump({"agent": {"tools": [{}]}}))
    assert ("[extra]" in tools_agents.inspect_tools("example")) is (kind == "internal")


@pytest.fixture
def eval_agent(agent_dir: Path) -> Path:
    directory = agent_dir / "evaluations/eval-sets"
    directory.mkdir(parents=True)
    (directory / "evaluation-set-default.json").write_text(
        json.dumps({"evaluations": [{"name": "synthetic-train", "inputs": {"value": "original"}}]})
    )
    return agent_dir


@pytest.mark.parametrize(
    "name", ["", ".", "..", "../outside", "/outside", "C:outside", "a\\b", "a\x00b"]
)
def test_eval_set_suffix_is_not_a_path(eval_agent: Path, name: str) -> None:
    assert helpers.find_eval_set_file(eval_agent, name) is None


@pytest.mark.parametrize("kind", ["exact", "partial", "directory", "internal-directory"])
def test_eval_set_reads_writes_and_fallbacks_are_contained(
    eval_agent: Path, tmp_path: Path, kind: str
) -> None:
    internal = kind == "internal-directory"
    directory = eval_agent / "evaluations/eval-sets"
    target_dir = (eval_agent if internal else tmp_path) / "linked-eval-sets"
    target_dir.mkdir()
    target = target_dir / "evaluation-set-default.json"
    target.write_text(
        json.dumps(
            {
                "evaluations": [{"name": "fixture", "inputs": {"value": "EVAL_SENTINEL"}}],
                "evaluatorRefs": ["EVAL_SENTINEL"],
            }
        )
    )
    (directory / "evaluation-set-default.json").unlink()
    if kind.endswith("directory"):
        directory.rmdir()
        directory.symlink_to(target_dir, target_is_directory=True)
        suffix = "default"
    else:
        suffix = "default" if kind == "exact" else "partial"
        (directory / f"evaluation-set-long-{suffix}.json").symlink_to(target)
    assert (helpers.find_eval_set_file(eval_agent, suffix) is not None) is internal
    for result in (
        tools_datapoints.get_datapoint("example", "fixture"),
        tools_agents.summarize_agent("example"),
        tools_agents.inspect_agent("example"),
    ):
        assert ("EVAL_SENTINEL" in result) is internal
    result = tools_datapoints.update_datapoint_inputs(
        "example", "fixture", '{"value": "changed"}', eval_set_name=suffix
    )
    assert result.startswith("Updated") is internal
    assert (
        json.loads(target.read_text())["evaluations"][0]["inputs"]["value"] == "changed"
    ) is internal


def test_eval_set_internal_alias_and_mutator_helpers(eval_agent: Path) -> None:
    directory = eval_agent / "evaluations/eval-sets"
    target = directory / "evaluation-set-default.json"
    (directory / "evaluation-set-alias.json").symlink_to(target.name)
    assert helpers.find_eval_set_file(eval_agent, "fault") == target
    assert helpers.find_eval_set_file(eval_agent, "alias") == target
    assert tools_datapoints.update_evaluation_criterias(
        "example", "synthetic-train", "judge", '{"answer": "ok"}', eval_set_name="alias"
    ).startswith("Updated")
    (directory / "evaluation-set-directory.json").mkdir()
    (directory / "evaluation-set-loop.json").symlink_to("evaluation-set-loop.json")
    assert helpers.find_eval_set_file(eval_agent, "directory") is None
    assert helpers.find_eval_set_file(eval_agent, "loop") is None


def test_non_target_eval_set_identity_and_prompt_protection(
    eval_agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = eval_agent / "evaluations/eval-sets"
    hidden = directory / "evaluation-set-test.json"
    hidden.write_text('{"evaluations": [{"name": "HELD_OUT_SYNTHETIC"}]}')
    (directory / "evaluation-set-alias.json").symlink_to(hidden.name)
    (eval_agent / "hard-alias.json").hardlink_to(hidden)
    (eval_agent.parent / "agent-alias").symlink_to(eval_agent, target_is_directory=True)
    config = yaml.safe_load((eval_agent / "agent.yaml").read_text())
    config["agent"]["evaluation_sets"] = ["core::evaluations/eval-sets/evaluation-set-test.json"]
    config["agent"]["system_prompt"] = "core::hard-alias.json"
    (eval_agent / "agent.yaml").write_text(yaml.safe_dump(config))
    (eval_agent / "versions.yaml").write_text(
        yaml.safe_dump(
            {
                "train": {
                    "agent": {
                        "evaluation_sets": [
                            "core::evaluations/eval-sets/evaluation-set-default.json"
                        ]
                    }
                }
            }
        )
    )
    monkeypatch.setenv(helpers.TARGET_ENV, "example:train")
    assert hidden in helpers.hidden_eval_sets(eval_agent)
    for agent in ("example", "agent-alias"):
        for relative in (
            "hard-alias.json",
            "evaluations/eval-sets/evaluation-set-test.json",
            "evaluations/eval-sets/evaluation-set-alias.json",
        ):
            assert tools_agents.read_agent_file(agent, relative).startswith("Access denied")
        assert tools_agents.get_prompt(agent).startswith("Access denied")
        assert tools_agents.update_prompt(agent, "changed").startswith("Access denied")
        assert tools_datapoints.update_datapoint_inputs(
            agent, "HELD_OUT_SYNTHETIC", "{}", eval_set_name="test"
        ).startswith("Access denied")
        for result in (
            tools_datapoints.get_datapoint(agent, "missing"),
            tools_agents.summarize_agent(agent),
            tools_agents.inspect_agent(agent),
            tools_agents.read_agent_file(agent, "missing"),
        ):
            assert "HELD_OUT_SYNTHETIC" not in result
        assert "synthetic-train" in tools_datapoints.get_datapoint(agent, "synthetic-train")


def test_other_versions_prompts_are_hidden_from_the_target(
    agent_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = "prompts/gepa__x/system.md"
    (agent_dir / "prompts/gepa__x").mkdir()
    (agent_dir / other).write_text("OTHER_VERSION_PROMPT")
    (agent_dir / "alias.md").symlink_to(agent_dir / other)
    assert "OTHER_VERSION_PROMPT" in tools_agents.read_agent_file("example", other)

    monkeypatch.setenv(helpers.TARGET_ENV, "example:train")
    listing = tools_agents.get_agent_files("example")
    assert "prompts/system.md" in listing
    assert "gepa__x" not in listing and "alias.md" not in listing
    for relative in (other, "alias.md"):
        assert tools_agents.read_agent_file("example", relative).startswith("Access denied")
    assert tools_agents.get_prompt("example") == "Synthetic prompt."


def test_escaping_versions_alias_does_not_unhide_base_sets(
    eval_agent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = {"agent": {"evaluation_sets": ["evaluations/eval-sets/evaluation-set-default.json"]}}
    (eval_agent / "agent.yaml").write_text(yaml.safe_dump(config))
    outside = tmp_path / "versions.yaml"
    outside.write_text("train: {}\n")
    (eval_agent / "versions.yaml").symlink_to(outside)
    monkeypatch.setenv(helpers.TARGET_ENV, "example:train")
    assert helpers.target_eval_sets(eval_agent, "train") == set()
    assert tools_datapoints.get_datapoint("example", "synthetic-train").startswith("Datapoint")


def test_eval_alias_outside_eval_root_stays_hidden_inside_agent(
    eval_agent: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hidden = eval_agent / "held-out.json"
    hidden.write_text('{"evaluations": [{"name": "SYNTHETIC_HELD_OUT"}]}')
    directory = eval_agent / "evaluations/eval-sets"
    (directory / "evaluation-set-test.json").symlink_to(hidden)
    (eval_agent / "agent.yaml").write_text(
        yaml.safe_dump(
            {"agent": {"evaluation_sets": ["evaluations/eval-sets/evaluation-set-default.json"]}}
        )
    )
    monkeypatch.setenv(helpers.TARGET_ENV, "example:base")
    assert helpers.find_eval_set_file(eval_agent, "test") is None
    assert tools_agents.read_agent_file("example", "held-out.json").startswith("Access denied")
    assert "held-out.json" not in tools_agents.get_agent_files("example")


def test_file_sizes_binary_reads_and_generic_errors(
    agent_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (agent_dir / "binary.dat").write_bytes(b"\xff\xfe")
    (agent_dir / "large.txt").write_bytes(b"x" * 500_001)
    assert tools_agents.read_agent_file("example", "binary.dat").startswith("Cannot read binary")
    assert tools_agents.read_agent_file("example", "large.txt").startswith("File too large")

    def denied(*_args: object, **_kwargs: object) -> str:
        raise PermissionError(f"synthetic denial: {tmp_path}/private-host-path")

    monkeypatch.setattr(Path, "read_text", denied)
    for result in (
        tools_agents.read_agent_file("example", "prompts/system.md"),
        helpers.load_agent_config(agent_dir)[1],
        helpers.load_eval_set(agent_dir / "example.json")[1],
        helpers.load_json_safe(agent_dir / "example.json")[1],
    ):
        assert result is not None
        assert "Error reading" in result
        assert str(tmp_path) not in result


def test_started_experiment_registry_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(helpers, "repo_root", lambda: tmp_path)
    monkeypatch.setenv("COACH_OUTPUT_DIR", "workspace")
    workspace = tmp_path / "workspace"
    (workspace / "round_000").mkdir(parents=True)
    (workspace / "round_000/round_stats.json").write_text('{"experiment_name": "first"}')
    helpers.init_started_experiments(workspace)
    helpers.record_started_experiment("second")
    (workspace / "round_000/round_stats.json").write_text('{"experiment_name": "not-trusted"}')
    helpers.init_started_experiments(workspace)
    assert helpers.started_experiments() == {"first", "second"}


def test_server_retains_twenty_tools_without_mutator_exposure() -> None:
    server = Path(helpers.__file__).parents[1] / "mcp_server.py"
    module = ast.parse(server.read_text())
    exposed = {
        node.name
        for node in module.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and any(
            isinstance(d, ast.Call)
            and isinstance(d.func, ast.Attribute)
            and isinstance(d.func.value, ast.Name)
            and d.func.value.id == "mcp"
            and d.func.attr == "tool"
            for d in node.decorator_list
        )
    }
    assert len(exposed) == 20
    assert not exposed & {
        "update_agent_yaml",
        "update_datapoint_inputs",
        "update_evaluation_criterias",
        "update_evaluator_prompt",
    }
