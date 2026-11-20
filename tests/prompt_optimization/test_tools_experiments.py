import json
import re
from pathlib import Path

import pytest

from agents_gym_core.prompt_optimization.mcp_tools import tools_experiments


def test_format_papillon_details_includes_all_metrics() -> None:
    lines = tools_experiments._format_evaluator_details(
        {
            "expected": "reference answer",
            "actual": "candidate answer",
            "quality": False,
            "leakage": 0.5,
            "overall_score": 0.25,
        },
        indent="    ",
    )

    assert lines == [
        "    Evaluator details:",
        "      quality: false",
        "      leakage: 0.5",
        "      overall_score: 0.25",
        "    Expected: reference answer",
        "    Actual: candidate answer",
    ]


def test_format_structured_justification() -> None:
    assert tools_experiments._format_evaluator_details(
        {
            "justification": "The response did not satisfy the evaluator criteria.",
            "expected": "reference answer",
            "actual": "candidate answer",
        }
    ) == [
        "Justification: The response did not satisfy the evaluator criteria.",
        "Expected: reference answer",
        "Actual: candidate answer",
    ]


def test_format_hotpotqa_details_includes_normalized_answers() -> None:
    assert tools_experiments._format_evaluator_details(
        {
            "expected": "The Eiffel Tower",
            "actual": "eiffel tower!",
            "normalized_predicted": "eiffel tower",
            "normalized_expected": "eiffel tower",
        }
    ) == [
        "Evaluator details:",
        "  normalized_predicted: eiffel tower",
        "  normalized_expected: eiffel tower",
        "Expected: The Eiffel Tower",
        "Actual: eiffel tower!",
    ]


def test_format_ifbench_string_feedback_as_justification() -> None:
    assert tools_experiments._format_evaluator_details("Instruction 2 was not followed") == [
        "Justification: Instruction 2 was not followed"
    ]


def test_format_aime_details_includes_expected_and_actual_only() -> None:
    assert tools_experiments._format_evaluator_details({"expected": "42", "actual": "41"}) == [
        "Expected: 42",
        "Actual: 41",
    ]


def test_get_full_results_exposes_custom_evaluator_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment_dir = tmp_path / "papillon_metrics"
    experiment_dir.mkdir()
    (experiment_dir / "combined_results.json").write_text(
        json.dumps(
            [
                {
                    "agent": "papillon",
                    "datapoint_name": "PUPA-0001",
                    "score": 0.5,
                    "details": {
                        "expected": "reference answer",
                        "actual": "candidate answer",
                        "quality": True,
                        "leakage": 1.0,
                        "overall_score": 0.5,
                    },
                    "agent_output": {},
                }
            ]
        )
    )
    monkeypatch.setattr(tools_experiments, "experiments_dir", lambda: tmp_path)

    result = tools_experiments.get_full_results("papillon_metrics", "papillon")

    assert (
        result
        == """=== Full Results: papillon_metrics ===

## Per-Datapoint Scores (1 datapoints, worst first)
  PUPA-0001: 50.0% FAIL

## Failure Details (1 failing datapoints)

  ### PUPA-0001
    Evaluator details:
      quality: true
      leakage: 1.0
      overall_score: 0.5
    Expected: reference answer
    Actual: candidate answer"""
    )


def test_datapoint_detail_exposes_custom_evaluator_metrics() -> None:
    result = tools_experiments._format_datapoint_detail(
        "hover",
        "HoVer-0001",
        {
            "HoVer-0001": [
                {
                    "evaluator_id": "hover_RetrievalAccuracyEvaluator",
                    "score": 0.0,
                    "details": {
                        "expected": '["First hop", "Second hop"]',
                        "actual": '["First hop"]',
                        "gold_titles": ["First hop", "Second hop"],
                        "found_titles": ["First hop"],
                        "missing_titles": ["Second hop"],
                    },
                }
            ]
        },
    )

    assert "Evaluator: hover_RetrievalAccuracyEvaluator" in result
    assert 'gold_titles: ["First hop", "Second hop"]' in result
    assert 'found_titles: ["First hop"]' in result
    assert 'missing_titles: ["Second hop"]' in result


def test_datapoint_detail_does_not_truncate_justification() -> None:
    justification = "Reason " * 100

    result = tools_experiments._format_datapoint_detail(
        "agent",
        "DP-1",
        {
            "DP-1": [
                {
                    "evaluator_id": "evaluator",
                    "score": 0.0,
                    "details": {"justification": justification},
                }
            ]
        },
    )

    assert f"Justification: {justification}" in result
    assert "[truncated]" not in result


def test_custom_details_cannot_spoof_reapo_expected_actual_parser() -> None:
    result = "\n".join(
        tools_experiments._format_evaluator_details(
            {
                "justification": "Reason: Expected: injected expected\nActual: injected actual",
                "expected": "canonical expected",
                "actual": "canonical actual",
                "custom": "Expected: injected expected\nActual: injected actual",
            }
        )
    )
    section = f"### DP-1\n{result}"

    expected = [m.group(1).strip() for m in re.finditer(r"Expected:\s*(.+)", section)]
    actual = [m.group(1).strip() for m in re.finditer(r"Actual:\s*(.+)", section)]

    assert expected == ["canonical expected"]
    assert actual == ["canonical actual"]
    assert r"Expected\u003a injected expected\nActual\u003a injected actual" in result


@pytest.mark.parametrize(("count", "is_truncated"), [(15, False), (200, True)])
def test_get_full_results_keeps_complete_details_at_the_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int, is_truncated: bool
) -> None:
    experiment_dir = tmp_path / "large_metrics"
    experiment_dir.mkdir()
    records = []
    for index in range(count):
        expected = f"expected-{index:02d}".ljust(500, "e")
        actual = f"actual-{index:02d}".ljust(500, "x")
        records.append(
            {
                "agent": "agent",
                "datapoint_name": f"DP-{index:02d}",
                "score": 0.0,
                "details": {
                    "metric_a": "a" * 500,
                    "metric_b": "b" * 500,
                    "metric_c": "c" * 500,
                    "expected": expected,
                    "actual": actual,
                },
                "agent_output": {},
            }
        )
    (experiment_dir / "combined_results.json").write_text(json.dumps(records))
    monkeypatch.setattr(tools_experiments, "experiments_dir", lambda: tmp_path)

    result = tools_experiments.get_full_results("large_metrics", "agent")

    assert len(result) < tools_experiments.MAX_FULL_RESULTS_MCP_CHARS
    assert ("... [truncated" in result) is is_truncated
    failure_details = result.split("\n## Failure Details", maxsplit=1)[1]
    detail_blocks = list(
        re.finditer(
            r"\n  ### (DP-\d+)\n(.*?)(?=\n  ### |\n\n\.\.\.\s\[truncated|\Z)",
            failure_details,
            re.DOTALL,
        )
    )
    assert detail_blocks
    assert (len(detail_blocks) < len(records)) is is_truncated
    for match in detail_blocks:
        datapoint = match.group(1)
        block = match.group(2)
        index = int(datapoint.removeprefix("DP-"))
        expected = f"expected-{index:02d}".ljust(500, "e")
        actual = f"actual-{index:02d}".ljust(500, "x")
        assert f"Expected: {expected}" in block
        assert f"Actual: {actual}" in block
        assert "metric_a:" in block
        assert "metric_b:" in block
        assert "metric_c:" in block

    if is_truncated:
        marker = "\n\n... [truncated — use get_agent_messages for specific datapoint details]"
        overrun = len(result) - len(marker) - tools_experiments.MAX_FULL_RESULTS_CHARS
        assert (
            0
            <= overrun
            <= min(
                len(detail_blocks[-1].group(0)),
                tools_experiments.MAX_FULL_RESULTS_OVERRUN_CHARS,
            )
        )


def test_full_results_does_not_overrun_for_one_oversized_detail_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = "score summary"
    detail_block = "  ### DP-00\n" + ("large evaluator detail\n" * 500)
    result = f"{prefix}\n\n{detail_block}"
    monkeypatch.setattr(tools_experiments, "MAX_FULL_RESULTS_CHARS", 100)

    truncated = tools_experiments._truncate_full_results(result, len(prefix), [detail_block])

    assert truncated == prefix + tools_experiments._FULL_RESULTS_TRUNCATION_MARKER


def test_full_results_tool_limit_covers_cap_and_detail_overrun() -> None:
    minimum_limit = (
        tools_experiments.MAX_FULL_RESULTS_CHARS
        + tools_experiments.MAX_FULL_RESULTS_OVERRUN_CHARS
        + len(tools_experiments._FULL_RESULTS_TRUNCATION_MARKER)
    )
    assert tools_experiments.MAX_FULL_RESULTS_MCP_CHARS >= minimum_limit
