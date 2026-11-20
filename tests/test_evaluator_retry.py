"""``fill`` retries evaluators that raised, and re-evaluates unreadable cached results."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

from agents_gym_core.hydra_agent import CoreAgentRegistry
from agents_gym_core.runtime import run_utils
from agents_gym_core.runtime.run_utils import get_datapoint_dir, should_run_evaluator
from agents_gym_core.runtime.runners.evaluation_manager import EvaluationManager
from agents_gym_core.runtime.runtime_config import RuntimeConfig
from agents_gym_core.utils.gym_types import Agent


def _write_result(experiment_dir: Path, details: Any, agent_output: Any = None) -> None:
    results = get_datapoint_dir(experiment_dir, "dp", 1) / "results"
    results.mkdir(parents=True)
    (results.parent / "output.json").write_text("[]")  # the agent run is on disk
    output = {"result": "ok"} if agent_output is None else agent_output
    result = {"score": 0.0, "details": details, "agent_output": output}
    (results / "judge.json").write_text(json.dumps(result))


def test_fill_retries_an_evaluator_that_raised(tmp_path: Path) -> None:
    _write_result(tmp_path, f"{run_utils.EVALUATOR_EXCEPTION_PREFIX} 401 Unauthorized")

    assert should_run_evaluator(tmp_path, "dp", "judge", "fill", 1)
    assert should_run_evaluator(tmp_path, "dp", "judge", "fill_skip_run_errors", 1)
    assert not should_run_evaluator(tmp_path, "dp", "judge", "fill_skip_errors", 1)


def test_fill_keeps_a_scored_result(tmp_path: Path) -> None:
    _write_result(tmp_path, "The answer does not match the expected output.")

    assert not should_run_evaluator(tmp_path, "dp", "judge", "fill", 1)
    assert not should_run_evaluator(tmp_path, "dp", "judge", "fill_skip_run_errors", 1)


def test_fill_skip_run_errors_keeps_a_failed_agent_run(tmp_path: Path) -> None:
    # Even when the judge also raised on the error output, the agent's failure stays scored.
    agent_error = {"type": "error", "error": "Error code: 500"}
    _write_result(tmp_path, f"{run_utils.EVALUATOR_EXCEPTION_PREFIX} timeout", agent_error)

    assert not should_run_evaluator(tmp_path, "dp", "judge", "fill_skip_run_errors", 1)


def test_fill_skip_run_errors_scores_a_datapoint_that_runs_again(tmp_path: Path) -> None:
    # The saved output is gone, so the agent runs again; its earlier failure must not be reused.
    _write_result(tmp_path, "agent failed", {"type": "error", "error": "Error code: 500"})
    (get_datapoint_dir(tmp_path, "dp", 1) / "output.json").unlink()

    assert run_utils.should_run_datapoint(tmp_path, "dp", "fill_skip_run_errors", 1)
    assert should_run_evaluator(tmp_path, "dp", "judge", "fill_skip_run_errors", 1)


def test_fill_skip_run_errors_reruns_an_unreadable_result(tmp_path: Path) -> None:
    _write_result(tmp_path, "scored")
    (get_datapoint_dir(tmp_path, "dp", 1) / "results" / "judge.json").write_text("{truncated")

    assert should_run_evaluator(tmp_path, "dp", "judge", "fill_skip_run_errors", 1)


def test_an_evaluator_exception_does_not_rerun_the_agent(tmp_path: Path) -> None:
    _write_result(tmp_path, f"{run_utils.EVALUATOR_EXCEPTION_PREFIX} timeout")

    assert not run_utils.is_datapoint_errored(tmp_path, "dp", 1)


async def test_an_unreadable_cached_result_is_evaluated_again(tmp_path: Path) -> None:
    manager = EvaluationManager(
        registry=cast(CoreAgentRegistry, MagicMock()), config=RuntimeConfig(mode="fill")
    )
    evaluator = SimpleNamespace(id="judge")
    execution = SimpleNamespace(
        loop_details=SimpleNamespace(loop_key="basic", model_dump=lambda: {}),
        run_number=1,
        datapoint=SimpleNamespace(name="dp", evaluation_criterias={"judge": {}}),
    )
    agent = cast(Agent, SimpleNamespace(name="sql", version="test", datapoint_splits={}))
    run_task = AsyncMock(return_value=("sql", "test", None, "judge", "dp", 1, {}, {}, None, "", ""))

    with (
        patch.object(manager, "_get_evaluators", return_value={"judge": evaluator}),
        patch(
            "agents_gym_core.runtime.runners.evaluation_manager.get_experiment_path",
            return_value=tmp_path,
        ),
        patch(
            "agents_gym_core.runtime.runners.evaluation_manager.should_run_evaluator",
            return_value=False,
        ),
        patch.object(manager, "_try_load_cached_result", return_value=None),
        patch.object(manager, "_run_evaluator_task", run_task),
        patch.object(manager, "_build_result_data", return_value={"evaluator_id": "judge"}),
    ):
        results = await manager.evaluate_one(cast(Any, execution), agent)

    run_task.assert_awaited_once()
    assert results == [{"evaluator_id": "judge"}]


async def test_a_datapoint_that_ran_is_evaluated_again_even_if_a_result_is_saved(
    tmp_path: Path,
) -> None:
    # The run saves the new output.json before evaluation, so the saved-result check alone would
    # keep the earlier failure: the execution's ``ran`` flag must override it.
    manager = EvaluationManager(
        registry=cast(CoreAgentRegistry, MagicMock()),
        config=RuntimeConfig(mode="fill_skip_run_errors"),
    )
    agent = cast(Agent, SimpleNamespace(name="sql", version="test", datapoint_splits={}))
    run_task = AsyncMock(return_value=("sql", "test", None, "judge", "dp", 1, {}, {}, None, "", ""))
    cached = MagicMock(return_value={"evaluator_id": "judge", "cached": True})

    def execution(ran: bool) -> Any:
        return SimpleNamespace(
            loop_details=SimpleNamespace(loop_key="basic", model_dump=lambda: {}),
            run_number=1,
            datapoint=SimpleNamespace(name="dp", evaluation_criterias={"judge": {}}),
            ran=ran,
        )

    with (
        patch.object(
            manager, "_get_evaluators", return_value={"judge": SimpleNamespace(id="judge")}
        ),
        patch(
            "agents_gym_core.runtime.runners.evaluation_manager.get_experiment_path",
            return_value=tmp_path,
        ),
        patch(
            "agents_gym_core.runtime.runners.evaluation_manager.should_run_evaluator",
            return_value=False,  # what the saved agent-error result says
        ),
        patch.object(manager, "_try_load_cached_result", cached),
        patch.object(manager, "_run_evaluator_task", run_task),
        patch.object(manager, "_build_result_data", return_value={"evaluator_id": "judge"}),
    ):
        loaded = await manager.evaluate_one(execution(ran=False), agent)
        fresh = await manager.evaluate_one(execution(ran=True), agent)

    cached.assert_called_once()  # loaded from disk: the saved result is kept
    run_task.assert_awaited_once()  # ran in this invocation: evaluated again
    assert len(loaded) == 1
    assert fresh == [{"evaluator_id": "judge"}]
