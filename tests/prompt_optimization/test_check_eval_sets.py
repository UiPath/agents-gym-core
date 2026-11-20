"""The ``check_evals`` checker takes each criterion's required field from its evaluator type."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from agents_gym_core.prompt_optimization.check_eval_sets import Severity, check_agent
from agents_gym_core.prompt_optimization.mcp_tools.tools_datapoints import check_evals

AGENTS_DIR = Path(__file__).resolve().parents[2] / "src" / "agents_gym_core" / "agents"
PAPER_ENTERPRISE_AGENTS = [
    "claim_history",
    "customer_support_triage_agent",
    "meeting_management",
    "product_questions_agent",
    "sql",
]
# Enterprise agents that ship with the repository but are not used in the paper.
OTHER_ENTERPRISE_AGENTS = [
    "athena_chatbot",
    "for_loop",
    "hts_classifier_agent",
    "loan",
    "product_recommendation",
    "salesforce_agent",
    "search_agent",
    "semantic_analysis",
    "servicenow_work_order_accelerator",
]
TRAIN_SET = "evaluation-set-claim_history_train.json"


def _failures(agent_dir: Path, check_evaluators: bool) -> list[str]:
    return [
        i.message for i in check_agent(agent_dir, check_evaluators) if i.severity is Severity.FAIL
    ]


@pytest.fixture
def claim_history_copy(tmp_path: Path) -> Path:
    """A copy of claim_history's evaluations (built-in tool-call + LLM-judge + custom evaluators)."""
    agent_dir = tmp_path / "claim_history"
    shutil.copytree(AGENTS_DIR / "claim_history" / "evaluations", agent_dir / "evaluations")
    return agent_dir


@pytest.mark.parametrize("check_evaluators", [False, True])
@pytest.mark.parametrize(
    ("evaluator_id", "field"),
    [
        ("claim_history_ToolCallCountEvaluator", "toolCallsCount"),
        ("claim_history_ToolCallArgsEvaluator", "toolCalls"),
    ],
)
def test_missing_tool_call_field_fails_in_both_modes(
    claim_history_copy: Path, check_evaluators: bool, evaluator_id: str, field: str
) -> None:
    path = claim_history_copy / "evaluations" / "eval-sets" / TRAIN_SET
    data = json.loads(path.read_text())
    first = data["evaluations"][0]
    del first["evaluationCriterias"][evaluator_id][field]
    path.write_text(json.dumps(data))

    failures = _failures(claim_history_copy, check_evaluators)

    assert len(failures) == 1, failures
    assert f'"{first["name"]}"' in failures[0]
    assert f'["{evaluator_id}"] missing {field}' in failures[0]


@pytest.mark.parametrize("agent", PAPER_ENTERPRISE_AGENTS + OTHER_ENTERPRISE_AGENTS)
def test_enterprise_agents_pass_cli_and_mcp_tool(agent: str) -> None:
    for flags in ([], ["--check-evaluators"]):
        cli = subprocess.run(
            [sys.executable, "-m", "agents_gym_core.prompt_optimization.check_eval_sets"]
            + ["--core", "--agent", agent, *flags],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert cli.returncode == 0, cli.stdout + cli.stderr
    report = check_evals(agent_name=agent, source="core")
    assert "Errors:             0" in report, report
    assert "Exit code" not in report, report
