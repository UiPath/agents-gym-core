"""``--run_report`` executes the report notebook: it needs a Jupyter kernel (ipykernel)."""

import json
from pathlib import Path

from agents_gym_core.reporting.run_report import run_notebook


def _result(agent: str, datapoint: str, run: int, score: float) -> dict:
    loop_key = "basic__auto__model:test-model"
    return {
        "experiment_unique_identifier": f"{agent}__base_{loop_key}_{datapoint}:run_{run}",
        "agent": agent,
        "agent_version": "base",
        "loop_name": "basic",
        "loop_version": "auto",
        "loop_key": loop_key,
        "loop_models": {"model": "test-model"},
        "datapoint_name": datapoint,
        "score": score,
        "evaluator_id": "ExactMatchEvaluator",
        "agent_output": {"answer": "42"},
        "evaluation_criteria": {"expectedOutput": {"answer": "42"}},
        "details": {"expected": "42", "actual": "42"},
        "evaluator_weights": {"ExactMatchEvaluator": 1.0},
        "run_number": run,
        "split": "all_data",
    }


def test_the_report_notebook_runs(tmp_path: Path) -> None:
    results = [
        _result(agent, f"dp-{d}", run, (d + run) % 2)
        for agent in ("agent_a", "agent_b")
        for d in range(3)
        for run in (1, 2)
    ]
    combined = tmp_path / "combined_results.json"
    combined.write_text(json.dumps(results))
    output = tmp_path / "report_executed.ipynb"

    run_notebook(results_paths=[str(combined)], output_path=str(output), export_html=True)

    cells = json.loads(output.read_text())["cells"]
    errors = [o for c in cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors[0]
    assert output.with_suffix(".html").is_file()
