"""Runtime helper class with config as instance variable."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.base_evaluator import EvaluationResult
from uipath.eval.evaluators.llm_as_judge_evaluator import LLMJudgeMixin
from uipath.eval.models import NumericEvaluationResult
from uipath.eval.models.evaluation_set import EvaluationItem

from agents_gym_core.loops import LoopDetails
from agents_gym_core.runtime.evaluation_criteria import (
    is_infrastructure_failure,
    prepare_evaluation_criteria,
)
from agents_gym_core.runtime.run_utils import (
    AgentDatapointExecution,
    get_datapoint_dir,
    get_experiment_path,
)
from agents_gym_core.runtime.runtime_config import RuntimeConfig
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.trace_utils import info


class RuntimeHelpers:
    """Helper methods that use RuntimeConfig."""

    def __init__(self, config: RuntimeConfig):
        self.config = config

    def compute_inventory_targets(
        self,
        graphs_list: Sequence[tuple[CompiledStateGraph, EvaluationItem, Agent]],
    ) -> tuple[list[Path], list[Agent], list[list[str]]]:
        """Experiment dirs, agents and selected datapoint names for the inventory, one per
        (agent, version, loop_key)."""
        experiment_dirs: list[Path] = []
        agents_to_inventory: list[Agent] = []
        datapoint_names: list[list[str]] = []

        index: dict[tuple[str, str, str], int] = {}
        for graph, datapoint, agent in graphs_list:
            details = graph.loop_details  # pyright: ignore[reportAttributeAccessIssue]
            key = (agent.name, agent.version, details.loop_key)
            if key not in index:
                index[key] = len(experiment_dirs)
                experiment_dirs.append(self.get_experiment_dir(agent, details))
                agents_to_inventory.append(agent)
                datapoint_names.append([])
            names = datapoint_names[index[key]]
            if datapoint.name not in names:  # one entry per run of a datapoint
                names.append(datapoint.name)

        return experiment_dirs, agents_to_inventory, datapoint_names

    async def run_and_save_evaluator(
        self,
        agent: Agent,
        evaluator: BaseEvaluator,
        execution: AgentDatapointExecution,
        experiment_dir: Path | None,
        run_number: int | None = None,
    ) -> tuple[
        str, dict[str, Any] | None, dict[str, Any] | str, EvaluationResult, str | None, str | None
    ]:
        """Run one evaluator; with ``experiment_dir``, save to ``results/<evaluator_id>.json``.

        Returns (evaluator_id, evaluation_criteria, agent_output, result, judge_model,
        judge_prompt).
        """
        datapoint = execution.datapoint
        workload_execution = execution.workload_execution

        evaluation_criteria = prepare_evaluation_criteria(
            evaluator, workload_execution, datapoint.evaluation_criterias[evaluator.id]
        )
        if self.config.verbose:
            info(
                "Evaluating criterion",
                evaluator_id=evaluator.id,
                datapoint_name=datapoint.name,
                agent_name=agent.name,
                evaluation_criteria=str(evaluation_criteria),
            )

        if is_infrastructure_failure(workload_execution.workload_output):
            result = NumericEvaluationResult(
                score=0.0, details=workload_execution.workload_output.get("details")
            )
        else:
            result = await evaluator.validate_and_evaluate_criteria(
                workload_execution=workload_execution, evaluation_criteria=evaluation_criteria
            )

        if experiment_dir:
            datapoint_dir = get_datapoint_dir(
                experiment_dir,
                datapoint.name,
                run_number=run_number,
            )
            results_dir = datapoint_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            result_file = results_dir / f"{evaluator.id}.json"
            result_to_save = {
                "evaluator_id": evaluator.id,
                "datapoint_name": datapoint.name,
                "score": result.score,
                "details": result.details.model_dump()
                if isinstance(result.details, BaseModel)
                else result.details,
                "evaluation_criteria": evaluation_criteria,
                "agent_output": workload_execution.workload_output,
                "split": agent.datapoint_splits.get(datapoint.name, "all_data"),
            }
            if isinstance(evaluator, LLMJudgeMixin):
                result_to_save["judge_model"] = evaluator.evaluator_config.model
                result_to_save["judge_prompt"] = evaluator.evaluator_config.prompt

            with open(result_file, "w") as f:
                json.dump(result_to_save, f, indent=2)

        judge_model = None
        judge_prompt = None
        if isinstance(evaluator, LLMJudgeMixin):
            judge_model = evaluator.evaluator_config.model
            judge_prompt = evaluator.evaluator_config.prompt

        return (
            evaluator.id,
            evaluation_criteria,
            workload_execution.workload_output,
            result,
            judge_model,
            judge_prompt,
        )

    def get_experiment_dir(self, agent: Agent, loop_details: LoopDetails) -> Path:
        """Experiment directory of an agent and loop under the configured output dir."""
        return get_experiment_path(
            base_output_dir=self.config.base_output_dir,
            experiment_name=self.config.experiment_name,
            agent=agent,
            loop_details=loop_details,
        )
