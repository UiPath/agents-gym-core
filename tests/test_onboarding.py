"""Standalone-agent contracts: explicit selection, isolated tools and real workers."""

import argparse
import asyncio
import importlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import BaseModel, ConfigDict, ValidationError
from uipath.eval.models.evaluation_set import EvaluationSet

from agents_gym_core.loops.callable_loop import CallableAgent, create_callable_graph
from agents_gym_core.project import PROJECT_FILE, ProjectDirectoryProvider
from agents_gym_core.prompt_optimization.mcp_tools import helpers, tools_agents, tools_runner
from agents_gym_core.prompt_optimization.mcp_tools.mcp_client import _build_connection
from agents_gym_core.prompt_optimization.reset_prompts import reset_prompts
from agents_gym_core.sdk import cli
from agents_gym_core.sdk.onboarding import new_project, prepare_jsonl

MODEL = "gpt-4.1-mini-2025-04-14"
NAME = "my_agent"


def _env(home: Path, port: int | None = None) -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "AGENTS_GYM_SKIP_DOTENV": "1"}
    if port is not None:
        env |= {
            "OPENAI_API_KEY": "sk-stub",
            "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "AGENTS_GYM_LLM_BACKEND": "langchain",
        }
    return env


def _gym(
    root: Path, *args: str, port: int | None = None, project: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "agents_gym_core.sdk.cli",
            *(["--project", str(root)] if project else []),
            *args,
        ],
        cwd=root,
        env=_env(root, port),
        capture_output=True,
        text=True,
        timeout=180,
    )


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "project with spaces"
    new_project(argparse.Namespace(name=NAME, directory=str(root)))
    return root


def test_router_selects_provider_once_without_mutating_process(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    before_env, before_path, before_argv = dict(os.environ), sys.path.copy(), sys.argv.copy()
    with patch.object(Path, "read_text", autospec=True, side_effect=Path.read_text) as read:
        provider = ProjectDirectoryProvider(project)
        assert provider.get_agents_dir() == project / "agents"
        assert provider.agent_roots("all") == (project / "agents",)
        assert read.call_count == 1
    real_import = cli.importlib.import_module
    monkeypatch.setattr(
        cli.importlib,
        "import_module",
        lambda name: (
            types.SimpleNamespace(main=lambda **kwargs: calls.append(kwargs))
            if name == "agents_gym_core.testbed_core"
            else real_import(name)
        ),
    )
    arguments = ["--agents", "my_agent:train", "--models=name with spaces", "--", "-h"]
    cli.main(["--project", str(project), "eval", *arguments])
    assert calls[0]["argv"] == arguments
    assert calls[0]["prog"] == "gym eval"
    assert calls[0]["directory_provider"].root == project
    assert (os.environ, sys.path, sys.argv) == (before_env, before_path, before_argv)
    (project / PROJECT_FILE).write_text('{"version": 2}')
    with pytest.raises(SystemExit):
        cli.main(["--project", str(project), "eval"])


def test_trailing_underscore_is_rejected_before_creating_project(tmp_path: Path) -> None:
    root = tmp_path / "invalid"
    with pytest.raises(SystemExit, match="trailing '_'"):
        new_project(argparse.Namespace(name="a_", directory=str(root)))
    assert not root.exists()


@pytest.mark.parametrize("module", ["main", "mcp_server"])
@pytest.mark.parametrize("subclass", [False, True])
def test_optimizer_rejects_providers_that_workers_cannot_reconstruct(
    project: Path, module: str, subclass: bool
) -> None:
    provider = (
        type("CustomProjectProvider", (ProjectDirectoryProvider,), {})(project)
        if subclass
        else helpers.DIRECTORY_PROVIDER
    )
    entry = importlib.import_module(f"agents_gym_core.prompt_optimization.{module}")
    with pytest.raises(TypeError, match="exact ProjectDirectoryProvider or None"):
        entry.main([], directory_provider=provider)


def test_reproduction_requires_bundled_agents_even_in_project_directory(project: Path) -> None:
    rejected = _gym(project, "reproduce", "main-mini", "--dry-run")
    assert rejected.returncode != 0 and "omit --project" in rejected.stderr
    ordinary = _gym(project, "reproduce", "main-mini", "--dry-run", project=False)
    assert ordinary.returncode == 0, ordinary.stderr
    assert "aime_2025_agent:baseline-test" in ordinary.stdout and NAME not in ordinary.stdout
    assert not (project / "experiments").exists()


async def test_mcp_worker_uses_selected_project_and_hides_test_data(project: Path) -> None:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    with helpers.optimization_context(ProjectDirectoryProvider(project)):
        connection = _build_connection()
    connection.env = {**_env(project), "COACH_TARGET_AGENT": f"{NAME}:train"}
    async with stdio_client(connection) as (reader, writer):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            prompt = await session.call_tool("get_prompt", {"agent_name": NAME})
            assert "Answer the question concisely" in str(prompt.content)
            hidden = await session.call_tool(
                "read_agent_file", {"agent_name": NAME, "relative_path": "data/test.jsonl"}
            )
            assert "Access denied" in str(hidden.content)


async def test_tools_preserve_nested_concurrent_and_thread_scopes(
    project: Path, tmp_path: Path
) -> None:
    second = tmp_path / "second"
    new_project(argparse.Namespace(name=NAME, directory=str(second)))
    (second / "agents" / NAME / "prompts" / "system_prompt.md").write_text("second")
    first_provider, second_provider = (
        ProjectDirectoryProvider(project),
        ProjectDirectoryProvider(second),
    )
    default = helpers.directory_provider()

    async def read(provider: ProjectDirectoryProvider) -> str:
        with helpers.optimization_context(provider):
            await asyncio.sleep(0)
            return await asyncio.to_thread(tools_agents.get_prompt, NAME)

    with helpers.optimization_context(first_provider):
        assert helpers.experiments_dir() == project / "experiments"
        with pytest.raises(RuntimeError):
            with helpers.optimization_context(second_provider):
                assert tools_agents.get_prompt(NAME) == "second"
                raise RuntimeError("interrupted invocation")
        assert helpers.repo_root() == project
        first, second_text = await asyncio.gather(read(first_provider), read(second_provider))
        assert "Answer the question concisely" in first and second_text == "second"
    assert helpers.directory_provider() is default


def test_baseline_and_optimizer_workers_use_their_own_splits(
    project: Path, stub: tuple[int, Path]
) -> None:
    port, log = stub
    assert _gym(project, "init", NAME).returncode == 0
    result = _gym(
        project,
        "eval",
        "--agents",
        f"{NAME}:baseline-test",
        "--models",
        MODEL,
        "--num_runs",
        "1",
        "--experiment_name",
        "baseline",
        port=port,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    costs = list((project / "experiments" / "baseline").rglob("cost.json"))
    assert len(costs) == 2 and all(json.loads(p.read_text())["num_llm_calls"] == 1 for p in costs)
    baseline = [json.loads(line)["text"] for line in log.read_text().splitlines()]
    assert any("capital of Italy" in text for text in baseline)
    assert all("capital of France" not in text for text in baseline)
    log.write_text("")
    provider = ProjectDirectoryProvider(project)
    with patch.dict(
        os.environ, {**_env(project, port), "COACH_TARGET_AGENT": f"{NAME}:train"}, clear=True
    ):
        with helpers.optimization_context(provider):
            reply = tools_runner.run_agent_async(
                f"{NAME}:train", model=MODEL, experiment_name="train"
            )
            assert reply.startswith("Started run 'run_"), reply
            run = next(reversed(tools_runner._runs.values()))
            assert run["process"].wait(timeout=120) == 0
    train = [json.loads(line)["text"] for line in log.read_text().splitlines()]
    assert any("capital of France" in text for text in train)
    assert all("capital of Italy" not in text for text in train)
    prompt = project / "agents" / NAME / "prompts" / "system_prompt.md"
    prompt.write_text("edited")
    assert reset_prompts(NAME, directory_provider=provider)
    assert "Answer the question concisely" in prompt.read_text()


def test_optimizer_selects_local_train_and_project_workspace(
    project: Path, stub: tuple[int, Path]
) -> None:
    port, _ = stub
    assert _gym(project, "init", NAME).returncode == 0
    run = _gym(
        project,
        "optimize",
        NAME,
        "--optimizer-model",
        MODEL,
        "-m",
        MODEL,
        "-r",
        "1",
        "--recursion-limit",
        "4",
        "-n",
        "onboarding",
        "-o",
        port=port,
    )
    assert run.returncode == 1 and "Stopped at --recursion-limit 4" in run.stderr, run.stderr
    assert "Optimizing my_agent:train" in run.stdout
    (workspace,) = project.glob("tmp_onboarding_*")
    assert (workspace / "run_metadata.json").is_file()


@pytest.mark.parametrize("key", ["id", "inputs"])
def test_split_overlap_is_rejected_before_writing(project: Path, key: str) -> None:
    agent = project / "agents" / NAME
    row = json.loads((agent / "data/train.jsonl").read_text().splitlines()[0])
    case = {"id": "new", "inputs": {"question": "new"}, "expected_output": {"answer": "x"}}
    case[key] = row[key]
    (agent / "data/test.jsonl").write_text(json.dumps(case))
    with pytest.raises(ValueError, match="duplicate"):
        prepare_jsonl(agent, {})
    assert not (agent / "evaluations").exists()


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_adapter_receives_current_prompts_and_validates_output(
    tmp_path: Path, asynchronous: bool
) -> None:
    class Inputs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        question: str

    class Outputs(BaseModel):
        answer: str

    path = tmp_path / "adapter.py"
    path.write_text(f"""{"async " if asynchronous else ""}def run(**kwargs):
    assert kwargs["system_prompt"] == "revised"
    assert kwargs["user_prompt"] == "{{question}}"
    return {{"answer": kwargs["inputs"]["question"]}}
""")
    agent = CallableAgent(
        name="external",
        system_prompt="revised",
        user_prompt="{question}",
        input_schema=Inputs,
        output_schema=Outputs,
        evaluation_set=EvaluationSet(id="external", name="external"),
        entrypoint=f"{path}:run",
    )
    model = FakeListChatModel(responses=["unused"])
    graph = create_callable_graph(agent, model).compile()
    assert await graph.ainvoke({"question": "q"}) == {"answer": "q"}
    agent.output_schema = type("WrongOutput", (BaseModel,), {"__annotations__": {"wrong": str}})
    with pytest.raises(ValidationError):
        await create_callable_graph(agent, model).compile().ainvoke({"question": "q"})
