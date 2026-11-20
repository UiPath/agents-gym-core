"""Utilities for agent execution and evaluation runs."""

import hashlib
import json
from collections.abc import Collection
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.models import ErrorEvaluationResult, NumericEvaluationResult, WorkloadExecution
from uipath.eval.models.evaluation_set import EvaluationItem

from agents_gym_core.loops import LoopDetails
from agents_gym_core.runtime.cost_tracking import CostRecord, load_cost_record
from agents_gym_core.utils import DirectoryProvider
from agents_gym_core.utils.gym_types import Agent, EvaluatorDatapointResult
from agents_gym_core.utils.span_normalize import normalize_tool_spans
from agents_gym_core.utils.trace_utils import (
    OUTPUT_FILENAME,
    get_agent_output_from_trace,
    info,
    load_agent_output,
    load_execution_data,
    warning,
)

RunMode = Literal[
    "overwrite", "fill", "fill_rerun_evals", "fill_skip_errors", "fill_skip_run_errors"
]

# (graph, datapoint, agent, run_number)
GraphDatapoint = tuple[CompiledStateGraph, EvaluationItem, Agent, int]


class AggregationMode(StrEnum):
    """Aggregation mode for score calculation."""

    EVALUATOR_RUN = (
        "evaluator_run"  # Aggregates the results per evaluators for each run, then across runs.
    )
    DATAPOINT_RUN = (
        "datapoint_run"  # Aggregates the results per datapoints for each run, then across runs.
    )
    RUN_DATAPOINT = "run_datapoint"  # Aggregates the results across runs for each datapoint.


class ExecutionMetadata(BaseModel):
    """Metadata associated with an agent execution."""

    model_config = {"arbitrary_types_allowed": True}

    loop_details: LoopDetails
    agent: Agent
    run_number: int | None = None


class AgentDatapointExecution(BaseModel):
    datapoint: EvaluationItem
    workload_execution: WorkloadExecution
    run_number: int = 1
    agent_name: str = ""
    agent_version: str = ""
    loop_details: LoopDetails | None = None
    cost: CostRecord | None = None
    # True when the agent ran in this invocation, False when the execution was loaded from
    # disk. A datapoint that ran is evaluated again in every mode, never served old results.
    ran: bool = Field(default=False, exclude=True)

    @property
    def loop_name(self) -> str:
        """Get the loop name string."""
        return self.loop_details.loop_name if self.loop_details else ""

    @property
    def loop_key(self) -> str:
        """Get the loop key string."""
        return self.loop_details.loop_key if self.loop_details else ""


def get_agent_config_hash(agent: Agent) -> str:
    """8-character hash of the agent's name and version."""
    agent_config = {
        "agent_name": agent.name,
        "agent_version": agent.version,
    }
    return hashlib.sha256(json.dumps(agent_config, sort_keys=True).encode()).hexdigest()[:8]


def get_loop_config_hash(loop_key: str, **kwargs: Any) -> str:
    """8-character hash of the loop key and any extra loop parameters."""
    loop_config = {
        "loop_key": loop_key,
        **kwargs,
    }
    return hashlib.sha256(json.dumps(loop_config, sort_keys=True).encode()).hexdigest()[:8]


def get_experiment_path(
    base_output_dir: Path,
    experiment_name: str,
    agent: Agent,
    loop_details: LoopDetails,
) -> Path:
    """``base_output_dir / experiment_name / agent / version / loop_name / loop_key``."""
    path = (
        base_output_dir
        / experiment_name
        / agent.name
        / agent.version
        / loop_details.loop_name
        / loop_details.loop_key
    )
    return path


def get_datapoint_dir(
    experiment_dir: Path, datapoint_name: str, run_number: int | None = None
) -> Path:
    """``experiment_dir / <datapoint_name>``, plus ``run_<N>`` when ``run_number`` is given."""
    # A "/" in the name would create nested directories.
    safe_name = datapoint_name.replace("/", "_")
    datapoint_dir = experiment_dir / safe_name
    if run_number is not None:
        datapoint_dir = datapoint_dir / f"run_{run_number}"
    return datapoint_dir


def get_datapoint_output_file(
    experiment_dir: Path, datapoint_name: str, run_number: int | None = None
) -> Path:
    """Path of a datapoint's output.json."""
    return get_datapoint_dir(experiment_dir, datapoint_name, run_number) / OUTPUT_FILENAME


def is_datapoint_executed(
    experiment_dir: Path, datapoint_name: str, run_number: int | None = None
) -> bool:
    """True if the datapoint's output.json exists, is non-empty and parses as JSON.

    A corrupt or truncated file counts as not executed, so the datapoint runs again
    instead of being scored as an empty output.
    """
    output_file = get_datapoint_output_file(experiment_dir, datapoint_name, run_number)
    if not output_file.exists() or output_file.stat().st_size == 0:
        return False
    try:
        content = output_file.read_text(encoding="utf-8").strip()
        if not content:
            return False
        json.loads(content)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        warning(
            "output.json is corrupt and will be re-run",
            datapoint_name=datapoint_name,
            output_file=str(output_file),
        )
        return False


def is_evaluator_errored(result_file: Path) -> bool:
    """True if the result file is unreadable or records an agent output of type ``error``."""
    try:
        with open(result_file) as f:
            result_data = json.load(f)
            agent_output = result_data.get("agent_output", {})
            if isinstance(agent_output, dict) and agent_output.get("type") == "error":
                return True
    except Exception:
        return True

    return False


# Prefix the UiPath SDK puts in ``details`` when an evaluator raised (score 0.0).
EVALUATOR_EXCEPTION_PREFIX = "Exception thrown by evaluator:"


def evaluator_exception_lines(evaluation_results: dict[str, Any]) -> list[str]:
    """Summarize SDK evaluator exceptions recorded by this invocation."""
    lines: list[str] = []
    for loop_results in evaluation_results.values():
        for evaluator_results in loop_results.values():
            for evaluator_id, datapoint_results in evaluator_results.items():
                for datapoint_key, result_data in datapoint_results.items():
                    result = result_data.get("result")
                    details = getattr(result, "details", None)
                    if not (
                        isinstance(result, ErrorEvaluationResult)
                        and isinstance(details, str)
                        and details.startswith(EVALUATOR_EXCEPTION_PREFIX)
                    ):
                        continue
                    agent_output = result_data.get("agent_output")
                    if isinstance(agent_output, dict) and agent_output.get("type") == "error":
                        continue  # the agent run failed; that is not an evaluator exception
                    datapoint_name = datapoint_key.rsplit(":run_", 1)[0]
                    lines.append(
                        f"{result_data.get('agent_name')}:{result_data.get('agent_version')} "
                        f"{datapoint_name} run {result_data.get('run_number')} {evaluator_id}: "
                        f"{details.removeprefix(EVALUATOR_EXCEPTION_PREFIX).strip()[:200]}"
                    )
    return lines


def is_evaluator_exception(result_file: Path) -> bool:
    """True if the evaluator itself raised (e.g. a judge with bad credentials).

    Unlike :func:`is_evaluator_errored`, this concerns the evaluator, not the agent run,
    so only the evaluator runs again.
    """
    try:
        with open(result_file) as f:
            details = json.load(f).get("details")
    except Exception:
        return True
    return isinstance(details, str) and details.startswith(EVALUATOR_EXCEPTION_PREFIX)


# Exception types of a model endpoint outage (connection,
# timeout, rate limit, credentials, server error), from the openai, anthropic, litellm, httpx,
# langchain-anthropic and UiPath LLM clients. A run that ended in one of them is an outage, not
# a failure of the agent.
ENDPOINT_ERROR_TITLES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "Timeout",
        "RateLimitError",
        "AuthenticationError",
        "PermissionDeniedError",
        "InternalServerError",
        "ServiceUnavailableError",
        "BadGatewayError",
        "OverloadedError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
        "AnthropicConnectionError",
        "AnthropicTimeoutError",
        "AnthropicRateLimitError",
        "AnthropicAuthenticationError",
        "AnthropicPermissionDeniedError",
        "AnthropicAPIError",
        "AnthropicOverloadedError",
        "UiPathRequestTimeoutError",
        "UiPathRateLimitError",
        "UiPathTooManyRequestsError",
        "UiPathAuthenticationError",
        "UiPathPermissionDeniedError",
        "UiPathInternalServerError",
        "UiPathBadGatewayError",
        "UiPathServiceUnavailableError",
        "UiPathGatewayTimeoutError",
        "UiPathOriginTimeoutError",
    }
)


def is_endpoint_error(agent_output: Any) -> bool:
    """True if the agent output records a run that failed because the model endpoint had an
    outage (see :data:`ENDPOINT_ERROR_TITLES`)."""
    return (
        isinstance(agent_output, dict)
        and agent_output.get("type") == "error"
        and agent_output.get("code") == "GENERAL_EXECUTION_ERROR"
        and agent_output.get("title") in ENDPOINT_ERROR_TITLES
    )


def _result_files(
    experiment_dir: Path,
    datapoint_name: str,
    run_number: int | None,
    evaluator_ids: Collection[str] | None,
) -> list[Path]:
    """The datapoint's saved result files: those of ``evaluator_ids`` when given (a result left by
    an evaluator the datapoint no longer has is ignored), else every one."""
    results_dir = get_datapoint_dir(experiment_dir, datapoint_name, run_number) / "results"
    if evaluator_ids is None:
        return sorted(results_dir.glob("*.json"))
    return [f for i in sorted(evaluator_ids) if (f := results_dir / f"{i}.json").exists()]


def endpoint_error_title(
    experiment_dir: Path,
    datapoint_name: str,
    run_number: int | None = None,
    evaluator_ids: Collection[str] | None = None,
) -> str | None:
    """The exception type if the datapoint's saved results record an endpoint failure."""
    for result_file in _result_files(experiment_dir, datapoint_name, run_number, evaluator_ids):
        try:
            agent_output = json.loads(result_file.read_text()).get("agent_output")
        except (OSError, ValueError, AttributeError):
            continue
        if is_endpoint_error(agent_output):
            return agent_output["title"]
    return None


def is_datapoint_errored(
    experiment_dir: Path,
    datapoint_name: str,
    run_number: int | None = None,
    evaluator_ids: Collection[str] | None = None,
) -> bool:
    """True if any evaluator result of the datapoint (of ``evaluator_ids``, when given) records
    an errored agent run.

    Only ``results/`` is inspected: a run with no evaluator results counts as not errored.
    """
    return any(
        is_evaluator_errored(result_file)
        for result_file in _result_files(experiment_dir, datapoint_name, run_number, evaluator_ids)
    )


def is_evaluator_executed(
    experiment_dir: Path,
    datapoint_name: str,
    evaluator_id: str,
    run_number: int | None = None,
    include_errored: bool = False,
) -> bool:
    """True if the evaluator's result file exists and, unless ``include_errored``,
    neither the agent run nor the evaluator errored."""
    datapoint_dir = get_datapoint_dir(experiment_dir, datapoint_name, run_number)
    result_file = datapoint_dir / "results" / f"{evaluator_id}.json"

    if not result_file.exists():
        return False

    if include_errored:
        return True
    return not (is_evaluator_errored(result_file) or is_evaluator_exception(result_file))


def load_execution(
    experiment_dir: Path,
    datapoint: EvaluationItem,
    directory_provider: DirectoryProvider,
    agent: Agent,
    run_number: int | None = None,
    loop_details: LoopDetails | None = None,
) -> AgentDatapointExecution | None:
    """Rebuild a datapoint execution from output.json and cost.json; None if no output.json.

    The spans come from the OTLP spans in output.json, the agent output from agent_output.json
    (what the agent returned), or from the spans for a run saved without one.
    ``directory_provider`` is unused.
    """
    datapoint_dir = get_datapoint_dir(experiment_dir, datapoint.name, run_number)
    output_file = get_datapoint_output_file(experiment_dir, datapoint.name, run_number)

    if not output_file.exists():
        return None

    agent_trace = load_execution_data(datapoint_dir)
    saved = load_agent_output(datapoint_dir)
    agent_output = (
        saved["agent_output"] if saved is not None else get_agent_output_from_trace(agent_trace)
    )

    workload_execution = WorkloadExecution(
        agent_input=datapoint.inputs,
        workload_output=agent_output,
        workload_trace=normalize_tool_spans(agent_trace),
        simulation_instructions=datapoint.input_mocking_strategy.prompt
        if datapoint.input_mocking_strategy
        else "",
    )

    return AgentDatapointExecution(
        datapoint=datapoint,
        workload_execution=workload_execution,
        run_number=run_number if run_number is not None else 1,
        agent_name=agent.name,
        agent_version=agent.version,
        loop_details=loop_details,
        cost=load_cost_record(datapoint_dir),
    )


def should_run_datapoint(
    experiment_dir: Path,
    datapoint_name: str,
    mode: RunMode,
    run_number: int | None = None,
    evaluator_ids: Collection[str] | None = None,
) -> bool:
    """Whether the datapoint runs under ``mode``, given what is on disk.

    ``evaluator_ids`` are the datapoint's evaluators; results of any other are ignored.
    """
    if mode == "overwrite":
        return True

    # The fill_skip_* modes keep errored datapoints; fill_skip_run_errors runs again one that
    # failed only because the model endpoint had an outage.
    if mode == "fill_skip_errors":
        return not is_datapoint_executed(experiment_dir, datapoint_name, run_number)
    if mode == "fill_skip_run_errors":
        return (
            not is_datapoint_executed(experiment_dir, datapoint_name, run_number)
            or endpoint_error_title(experiment_dir, datapoint_name, run_number, evaluator_ids)
            is not None
        )

    # fill and fill_rerun_evals also run errored datapoints again.
    executed = is_datapoint_executed(experiment_dir, datapoint_name, run_number)
    if not executed:
        return True

    return is_datapoint_errored(experiment_dir, datapoint_name, run_number, evaluator_ids)


def should_run_evaluator(
    experiment_dir: Path,
    datapoint_name: str,
    evaluator_id: str,
    mode: RunMode,
    run_number: int | None = None,
    evaluator_ids: Collection[str] | None = None,
) -> bool:
    """Whether the evaluator runs for the datapoint under ``mode``, given what is on disk.

    ``evaluator_ids`` are the datapoint's evaluators, as for :func:`should_run_datapoint`.
    """
    if mode == "overwrite" or mode == "fill_rerun_evals":
        return True

    # fill_skip_errors keeps every existing result, errored or not.
    if mode == "fill_skip_errors":
        return not is_evaluator_executed(
            experiment_dir, datapoint_name, evaluator_id, run_number, include_errored=True
        )

    # fill_skip_run_errors keeps a failed agent run's results and runs again only an evaluator
    # that raised, or a result that is missing or unreadable.
    if mode == "fill_skip_run_errors":
        # A datapoint that runs again (its output is missing or unreadable, or the model endpoint
        # failed) is scored again too. This is the plan (the run inventory); in the run itself the
        # execution's ``ran`` flag decides, because the new output.json is saved before evaluation.
        if should_run_datapoint(experiment_dir, datapoint_name, mode, run_number, evaluator_ids):
            return True
        datapoint_dir = get_datapoint_dir(experiment_dir, datapoint_name, run_number)
        try:
            result = json.loads((datapoint_dir / "results" / f"{evaluator_id}.json").read_text())
        except (OSError, ValueError):
            return True
        if not isinstance(result, dict):
            return True
        agent_output = result.get("agent_output")
        if isinstance(agent_output, dict) and agent_output.get("type") == "error":
            return False
        details = result.get("details")
        return isinstance(details, str) and details.startswith(EVALUATOR_EXCEPTION_PREFIX)

    # fill also runs errored evaluations again.
    return not is_evaluator_executed(experiment_dir, datapoint_name, evaluator_id, run_number)


def reconstruct_details(
    details_raw: Any,
    evaluator: BaseEvaluator | None = None,
) -> str | BaseModel | None:
    """Rebuild cached ``details`` as the evaluator's justification model, when it has one.

    Otherwise None and strings are returned as is, a dict becomes a JSON string and any
    other value its ``str()``.
    """
    if details_raw is None or isinstance(details_raw, str):
        return details_raw

    if isinstance(evaluator, BaseEvaluator):
        try:
            justification_type = evaluator._extract_justification_type()
            if isinstance(justification_type, type) and issubclass(justification_type, BaseModel):
                return justification_type.model_validate(details_raw)
        except Exception:
            pass

    if isinstance(details_raw, dict):
        return json.dumps(details_raw)

    return str(details_raw)


def load_evaluator_result(
    result_file: Path,
    evaluator: BaseEvaluator | None = None,
) -> dict[str, Any] | None:
    """Load one evaluator result file; None if it does not exist.

    Returns ``result``, ``evaluation_criteria``, ``agent_output`` and the recorded split and
    judge metadata; ``evaluator`` is used to rebuild ``details``.
    """
    if not result_file.exists():
        return None

    with open(result_file) as f:
        result_data = json.load(f)

    details = reconstruct_details(result_data.get("details"), evaluator)
    loaded: dict[str, Any] = {
        "result": NumericEvaluationResult(
            score=result_data.get("score", 0.0),
            details=details,
        ),
        "evaluation_criteria": result_data.get("evaluation_criteria"),
        "agent_output": result_data.get("agent_output"),
    }
    # Preserve recorded attribution; the caller supplies the split when it is absent.
    for field in ("split", "judge_model", "judge_prompt"):
        if field in result_data:
            loaded[field] = result_data[field]
    return loaded


def load_cached_evaluator_result(
    experiment_dir: Path,
    datapoint_name: str,
    evaluator_id: str,
    evaluator: BaseEvaluator | None = None,
    run_number: int | None = None,
) -> dict[str, Any] | None:
    """Load a datapoint run's ``results/<evaluator_id>.json`` with :func:`load_evaluator_result`."""
    datapoint_dir = get_datapoint_dir(experiment_dir, datapoint_name, run_number)
    result_file = datapoint_dir / "results" / f"{evaluator_id}.json"
    return load_evaluator_result(result_file, evaluator)


def save_evaluation_results(
    results: dict[str, dict[str, dict[str, Any]]],
    agent: Agent,
    loop_details: "LoopDetails",
    output_dir: Path | None = None,
) -> Path:
    """Write ``results`` (evaluator_id -> datapoint_key -> result) as a JSON list to
    ``output_dir/evaluation_results.json`` (default ``./results``) and return its path."""
    if output_dir is None:
        output_dir = Path("./results")

    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_entries = []
    loop_key = loop_details.loop_key
    for evaluator_id, datapoint_results in results.items():
        for datapoint_name, result in datapoint_results.items():
            experiment_id = f"{agent.name}_{agent.version}_{loop_key}_{evaluator_id}"

            # Keys have the form "<datapoint_name>:run_<N>".
            if ":run_" in datapoint_name:
                run_number = int(datapoint_name.split(":run_")[1])
            else:
                run_number = 1

            result_obj = result.get("result")
            details = None
            if result_obj and hasattr(result_obj, "details"):
                details_raw = result_obj.details
                if details_raw and hasattr(details_raw, "model_dump"):
                    details = details_raw.model_dump()
                else:
                    details = details_raw

            entry = EvaluatorDatapointResult(
                experiment_unique_identifier=experiment_id,
                agent=agent.name,
                agent_version=agent.version,
                loop_name=loop_details.loop_name,
                loop_version=loop_details.loop_version,
                loop_key=loop_key,
                loop_models=loop_details.models,
                datapoint_name=datapoint_name,
                score=result["result"].score,
                evaluator_id=evaluator_id,
                agent_output=result["agent_output"],
                evaluation_criteria=result["evaluation_criteria"],
                details=details,
                evaluator_weights=agent.evaluator_weights,
                judge_model=result.get("judge_model"),
                judge_prompt=result.get("judge_prompt"),
                run_number=run_number,
                # split keeps its schema default, "all_data".
            )
            evaluation_entries.append(entry)

    output_file = output_dir / "evaluation_results.json"

    results_list = [entry.model_dump(exclude_none=True) for entry in evaluation_entries]

    with open(output_file, "w") as f:
        json.dump(results_list, f, indent=2)

    info("Evaluation results saved", output_file=str(output_file), num_results=len(results_list))
    return output_file


def print_experiment_inventory(
    experiment_dirs: Path | list[Path],
    agents: Agent | list[Agent],
    max_datapoints: int = 0,
    mode: RunMode = "fill",
    num_runs: int = 1,
    is_final: bool = False,
    datapoint_names: list[list[str]] | None = None,
) -> dict[str, int]:
    """Print and return datapoint and evaluation status counts read from the experiment dirs.

    ``agents[i]`` belongs to ``experiment_dirs[i]``; ``datapoint_names[i]``, when given, are
    the datapoints the run selected there (``--datapoints``, ``--splits``), otherwise the
    first ``max_datapoints`` of the eval set. With ``is_final`` the to-be-run counts print as 0.
    """
    if isinstance(experiment_dirs, Path):
        dirs_list = [experiment_dirs]
    else:
        dirs_list = experiment_dirs

    if isinstance(agents, Agent):
        agents_list = [agents]
    else:
        agents_list = agents

    if len(dirs_list) != len(agents_list):
        raise ValueError(
            f"Number of experiment dirs ({len(dirs_list)}) must match number of agents ({len(agents_list)})"
        )

    total_completed = 0
    total_errored = 0
    total_not_run = 0
    total_evals_completed = 0
    total_evals_errored = 0
    total_evals_not_run = 0
    total_evals_total = 0
    total_to_run_datapoints = 0
    total_to_run_evals = 0
    total_datapoints = 0

    for i_dir, (experiment_dir, agent) in enumerate(zip(dirs_list, agents_list)):
        expected_datapoints = (
            datapoint_names[i_dir]
            if datapoint_names is not None
            else [
                dp.name
                for i, dp in enumerate(agent.evaluation_set.evaluations)
                if max_datapoints == 0 or i < max_datapoints
            ]
        )
        inventory = _get_experiment_inventory_counts_from_disk(
            experiment_dir, agent, expected_datapoints, mode, num_runs
        )
        total_datapoints += inventory["total"]
        total_completed += inventory["completed"]
        total_errored += inventory["errored"]
        total_not_run += inventory["not_run"]
        total_evals_completed += inventory["evals_completed"]
        total_evals_errored += inventory["evals_errored"]
        total_evals_not_run += inventory["evals_not_run"]
        total_evals_total += inventory["evals_total"]
        total_to_run_datapoints += inventory["to_run_datapoints"]
        total_to_run_evals += inventory["to_run_evals"]

    print("\n" + "=" * 60)
    print("EXPERIMENT INVENTORY")
    print("=" * 60)
    print(f"Agent Versions:          {len(agents_list)}")
    print(f"Runs per Datapoint:      {num_runs}")
    print(f"Total Datapoints:        {total_datapoints}")
    print(f"  ✓ Completed:           {total_completed}")
    print(f"  ✗ Errored:             {total_errored}")
    print(f"  ○ Not Run:             {total_not_run}")
    print(f"  ▶ To Be Run:           {0 if is_final else total_to_run_datapoints}")
    print("-" * 60)
    print(f"Total Evaluations:       {total_evals_total}")
    print(f"  ✓ Completed:           {total_evals_completed}")
    print(f"  ✗ Errored:             {total_evals_errored}")
    print(f"  ○ Not Run:             {total_evals_not_run}")
    print(f"  ▶ To Be Run:           {0 if is_final else total_to_run_evals}")
    print("=" * 60 + "\n")

    return {
        "total": total_datapoints,
        "completed": total_completed,
        "errored": total_errored,
        "not_run": total_not_run,
        "to_run_datapoints": total_to_run_datapoints,
        "evals_total": total_evals_total,
        "evals_completed": total_evals_completed,
        "evals_errored": total_evals_errored,
        "evals_not_run": total_evals_not_run,
        "to_run_evals": total_to_run_evals,
    }


def _get_experiment_inventory_counts_from_disk(
    experiment_dir: Path,
    agent: Agent,
    expected_datapoint_names: list[str],
    mode: RunMode = "fill",
    num_runs: int = 1,
) -> dict[str, int]:
    """Count datapoint and evaluation statuses on disk, and how many will run under ``mode``."""
    total = len(expected_datapoint_names) * max(num_runs, 1)
    completed = 0
    errored = 0
    not_run = 0
    evals_completed = 0
    evals_errored = 0
    evals_not_run = 0
    evals_total = 0
    to_run_datapoints = 0
    to_run_evals = 0

    run_numbers = list(range(1, num_runs + 1)) if num_runs >= 1 else [None]

    # Expected evaluator IDs per datapoint name, computed once; the first occurrence wins.
    evaluator_ids_by_name: dict[str, set[str]] = {}
    for eval_item in agent.evaluation_set.evaluations:
        if eval_item.name not in evaluator_ids_by_name:
            evaluator_ids_by_name[eval_item.name] = set(eval_item.evaluation_criterias.keys())

    for datapoint_name in expected_datapoint_names:
        expected_evaluator_ids = evaluator_ids_by_name.get(datapoint_name, set())

        for run_number in run_numbers:
            datapoint_dir = get_datapoint_dir(experiment_dir, datapoint_name, run_number)
            msg_file = get_datapoint_output_file(experiment_dir, datapoint_name, run_number)
            results_dir = datapoint_dir / "results"

            datapoint_executed = msg_file.exists() and msg_file.stat().st_size > 0
            has_error = False

            if not datapoint_executed:
                not_run += 1
            else:
                for result_file in _result_files(
                    experiment_dir, datapoint_name, run_number, expected_evaluator_ids or None
                ):
                    try:
                        with open(result_file) as f:
                            agent_output = json.load(f).get("agent_output", {})
                        if isinstance(agent_output, dict) and agent_output.get("type") == "error":
                            has_error = True
                            break
                    except Exception:
                        continue

                if has_error:
                    errored += 1
                else:
                    completed += 1

            if expected_evaluator_ids:
                evals_total += len(expected_evaluator_ids)

                datapoint_will_run = should_run_datapoint(
                    experiment_dir, datapoint_name, mode, run_number, expected_evaluator_ids
                )

                if datapoint_will_run:
                    to_run_datapoints += 1

                if datapoint_executed:
                    if results_dir.exists():
                        for evaluator_id in expected_evaluator_ids:
                            result_file = results_dir / f"{evaluator_id}.json"
                            # An errored agent run, an evaluator that raised and an unreadable
                            # file all count as errored.
                            if not result_file.exists():
                                evals_not_run += 1
                            elif is_evaluator_errored(result_file) or is_evaluator_exception(
                                result_file
                            ):
                                evals_errored += 1
                            else:
                                evals_completed += 1

                            if should_run_evaluator(
                                experiment_dir,
                                datapoint_name,
                                evaluator_id,
                                mode,
                                run_number,
                                expected_evaluator_ids,
                            ):
                                to_run_evals += 1
                    else:
                        evals_not_run += len(expected_evaluator_ids)
                        if mode == "overwrite":
                            to_run_evals += len(expected_evaluator_ids)
                        elif mode in (
                            "fill",
                            "fill_rerun_evals",
                            "fill_skip_errors",
                            "fill_skip_run_errors",
                        ):
                            to_run_evals += len(expected_evaluator_ids)
                else:
                    evals_not_run += len(expected_evaluator_ids)
                    if datapoint_will_run:
                        to_run_evals += len(expected_evaluator_ids)

    return {
        "total": total,
        "completed": completed,
        "errored": errored,
        "not_run": not_run,
        "to_run_datapoints": to_run_datapoints,
        "evals_total": evals_total,
        "evals_completed": evals_completed,
        "evals_errored": evals_errored,
        "evals_not_run": evals_not_run,
        "to_run_evals": to_run_evals,
    }
