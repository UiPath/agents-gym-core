"""The paper's skill prompts are pinned, and the tools ablation removes the paper's tools.

Skills: ``full`` is the paper's sigma_Full (the default), ``trimmed`` its sigma_Trimmed (skill
prompt ablation), ``tool-free`` its sigma_Ablate (the simplified skill of the score-only and
macro tools ablations). Tools ablation: Full, Score-Only (only the aggregate score of each run) and Macro
(keeps get_full_results); both restricted configurations also hide the eval-set files.
"""

import hashlib
from typing import Any

import pytest
from langchain_core.tools import BaseTool, StructuredTool

from agents_gym_core.prompt_optimization.main import build_parser
from agents_gym_core.prompt_optimization.react_optimizer.optimizer import apply_ablation
from agents_gym_core.prompt_optimization.react_optimizer.prompts import (
    SKILLS_DIR,
    _load_skill,
    skill_names,
)

# sha256 of the paper's three skill prompts (sigma_Full, sigma_Trimmed, sigma_Ablate).
PAPER_SKILLS = {
    "full": "911b9ae35d0aba9c039a82f357eca053d2b5e5a62d3e38f5d9b29c2e66cb6f83",
    "trimmed": "a5eeb967197a4dbaec1b01c4faeb138b3004f0243dc9993835c68dc2fcc0854b",
    "tool-free": "c7ce9187a0f970fad91201ab64388d13ab3107161315fd8da4333913a95c4d9b",
}

OPTIMIZER_TOOLS = [
    "get_agent_files", "read_agent_file", "summarize_agent", "inspect_agent", "inspect_tools",
    "get_prompt", "update_prompt", "check_evals", "get_datapoint",
    "run_agent_async", "get_run_status", "wait_for_run", "save_optimizer_log",
    "read_optimizer_log", "get_full_results", "list_experiments", "get_experiment_results",
    "get_agent_messages", "get_usage_stats", "reset_usage_stats",
]  # fmt: skip


def test_the_skills_are_pinned() -> None:
    assert skill_names() == sorted(PAPER_SKILLS)
    for name, digest in PAPER_SKILLS.items():
        assert hashlib.sha256((SKILLS_DIR / f"{name}.md").read_bytes()).hexdigest() == digest


def test_full_is_the_default_skill() -> None:
    args = build_parser().parse_args(["sql"])

    assert args.skill == "full"
    assert args.ablation == "full"


def test_an_unknown_skill_lists_the_available_ones() -> None:
    with pytest.raises(FileNotFoundError, match="available: full, tool-free, trimmed"):
        _load_skill("unknown")


def _tool(name: str) -> BaseTool:
    async def call(relative_path: str = "") -> str:
        return f"{name} read {relative_path}"

    return StructuredTool.from_function(coroutine=call, name=name, description=name)


def _names(tools: list[BaseTool]) -> set[str]:
    return {tool.name for tool in tools}


@pytest.mark.parametrize(
    ("ablation", "removed"),
    [
        ("full", set()),
        (
            "score-only",
            {
                "get_agent_messages",
                "get_datapoint",
                "get_experiment_results",
                "get_full_results",
                "check_evals",
            },
        ),
        (
            "macro",
            {"get_agent_messages", "get_datapoint", "get_experiment_results", "check_evals"},
        ),
    ],
)
def test_each_ablation_removes_its_results_analysis_tools(ablation: str, removed: set[str]) -> None:
    tools = apply_ablation([_tool(name) for name in OPTIMIZER_TOOLS], ablation)

    assert _names(tools) == set(OPTIMIZER_TOOLS) - removed


@pytest.mark.parametrize(
    "path",
    [
        "evaluations/./eval-sets/evaluation-set-sql_train.json",
        "./evaluations//eval-sets/evaluation-set-sql_train.json",
        "prompts/../evaluations/eval-sets/evaluation-set-sql_train.json",
        "Evaluations/Eval-Sets/evaluation-set-sql_train.json",
        "evaluations\\eval-sets\\evaluation-set-sql_train.json",
        "evaluations/eval-sets",
    ],
)
async def test_restricted_ablations_hide_the_eval_sets_however_spelled(path: str) -> None:
    tools = {tool.name: tool for tool in apply_ablation([_tool("read_agent_file")], "score-only")}
    read: Any = tools["read_agent_file"]

    assert (await read.ainvoke({"relative_path": path})).startswith("Access denied")


@pytest.mark.parametrize(
    ("ablation", "blocked"), [("full", False), ("score-only", True), ("macro", True)]
)
async def test_restricted_ablations_hide_the_eval_sets(ablation: str, blocked: bool) -> None:
    tools = {tool.name: tool for tool in apply_ablation([_tool("read_agent_file")], ablation)}
    read: Any = tools["read_agent_file"]

    eval_set = await read.ainvoke(
        {"relative_path": "evaluations/eval-sets/evaluation-set-sql_train.json"}
    )
    other = await read.ainvoke({"relative_path": "prompts/system_prompt.md"})

    assert eval_set.startswith("Access denied") is blocked
    assert other == "read_agent_file read prompts/system_prompt.md"
