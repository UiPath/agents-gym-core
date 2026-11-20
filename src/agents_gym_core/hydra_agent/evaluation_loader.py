"""Loader for evaluation sets and evaluators."""

import json
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from uipath.eval.evaluators import BaseEvaluator
from uipath.eval.evaluators.evaluator_factory import EvaluatorFactory
from uipath.eval.helpers import EvalHelpers
from uipath.eval.models.evaluation_set import EvaluationSet

from agents_gym_core.utils.directory_provider import DirectoryProvider

_WELL_KNOWN_SPLITS = {"train", "val", "test", "dev"}


def _infer_split_from_path(path: Path) -> str:
    """Split label of an eval-set file: ``x__<any>``, ``x_<train|val|test|dev>`` (or
    ``x-<train|val|test|dev>``), else ``all_data``.

    A custom label needs the double underscore so agent names such as ``my_agent`` are not
    read as splits.
    """
    stem = path.stem

    last_dunder = stem.rfind("__")
    if last_dunder != -1:
        label = stem[last_dunder + 2 :]
        if label:
            return label

    last_separator = max(stem.rfind("_"), stem.rfind("-"))
    if last_separator != -1:
        label = stem[last_separator + 1 :]
        if label in _WELL_KNOWN_SPLITS:
            return label

    return "all_data"


def check_datapoint_names(names: list[str], where: str = "") -> None:
    """Raise unless each datapoint gets its own run directory.

    ``get_datapoint_dir`` turns "/" into "_", and macOS and Windows ignore letter case, so names
    that differ only there share one; an empty name, "." or ".." names no directory of its own.
    """
    if invalid := sorted({name for name in names if name in {"", ".", ".."}}):
        raise ValueError(f"Datapoint name(s) {invalid} cannot name a run directory.")
    counts = Counter(name.replace("/", "_").casefold() for name in names)
    if duplicates := sorted(
        {name for name in names if counts[name.replace("/", "_").casefold()] > 1}
    ):
        raise ValueError(
            f"Datapoint name(s) {duplicates} share a run directory{where}; "
            'datapoint names must be unique, also with "/" read as "_" and ignoring case.'
        )


@dataclass
class LabelledEvaluationSet:
    """An ``EvaluationSet`` with the split label of each datapoint, by datapoint name."""

    evaluation_set: EvaluationSet
    datapoint_splits: dict[str, str] = field(default_factory=dict)


class EvaluationLoader:
    """Loader for evaluation sets and evaluators."""

    @staticmethod
    def _preprocess_conversational_inputs(data: dict[str, Any]) -> dict[str, Any]:
        """Copy ``conversationalInputs.currentUserPrompt.text`` to ``inputs.input``.

        The SDK's ``EvaluationItem`` model drops ``conversationalInputs``.
        """
        if "evaluations" not in data:
            return data

        for evaluation in data["evaluations"]:
            inputs = evaluation.get("inputs", {})
            if inputs:
                continue

            conv_inputs = evaluation.get("conversationalInputs")
            if conv_inputs and isinstance(conv_inputs, dict):
                current_prompt = conv_inputs.get("currentUserPrompt", {})
                user_text = current_prompt.get("text", "")
                if user_text:
                    evaluation["inputs"] = {"input": user_text}

        return data

    @staticmethod
    def _load_eval_set_with_preprocessing(eval_set_path: str) -> EvaluationSet:
        """Load an eval set through ``EvalHelpers`` after the conversational-input fix-up."""
        with open(eval_set_path, encoding="utf-8") as f:
            data = json.load(f)

        data = EvaluationLoader._preprocess_conversational_inputs(data)

        # EvalHelpers.load_eval_set reads a path (and migrates older eval-set formats).
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp)
            tmp_path = tmp.name

        try:
            eval_set, _ = EvalHelpers.load_eval_set(tmp_path)
        finally:
            Path(tmp_path).unlink()

        return eval_set

    @staticmethod
    def merge_evaluation_sets(
        labelled_sets: list[LabelledEvaluationSet],
    ) -> LabelledEvaluationSet:
        """Merge the datapoints, evaluator refs and split labels of several sets.

        Raises:
            ValueError: If two datapoints share a run directory: the same name, or names that
                differ only by "/" and "_" (``get_datapoint_dir`` turns "/" into "_") or by letter case. Also for
                an empty name, "." or "..", which name no directory of their own.
        """
        if not labelled_sets:
            raise ValueError("Cannot merge empty list of evaluation sets")

        check_datapoint_names(
            [
                item.name
                for labelled in labelled_sets
                for item in (labelled.evaluation_set.evaluations or [])
            ],
            f" in the evaluation sets {[labelled.evaluation_set.name for labelled in labelled_sets]}",
        )

        if len(labelled_sets) == 1:
            return labelled_sets[0]

        base_set = labelled_sets[0].evaluation_set
        merged_evaluations = list(base_set.evaluations) if base_set.evaluations else []
        merged_evaluator_refs = set(base_set.evaluator_refs) if base_set.evaluator_refs else set()

        for labelled in labelled_sets[1:]:
            eval_set = labelled.evaluation_set
            if eval_set.evaluations:
                merged_evaluations.extend(eval_set.evaluations)
            if eval_set.evaluator_refs:
                merged_evaluator_refs.update(eval_set.evaluator_refs)

        merged_dict: dict[str, Any] = {
            "id": base_set.id + "_merged",
            "name": f"Merged evaluation set ({len(labelled_sets)} sets)",
            "evaluations": merged_evaluations,
            "evaluator_refs": list(merged_evaluator_refs),
        }

        batch_size = getattr(base_set, "batch_size", None)
        if batch_size:
            merged_dict["batch_size"] = batch_size
        file_name = getattr(base_set, "fileName", None) or getattr(base_set, "file_name", None)
        if file_name:
            merged_dict["fileName"] = file_name

        merged_splits: dict[str, str] = {}
        for labelled in labelled_sets:
            merged_splits.update(labelled.datapoint_splits)

        return LabelledEvaluationSet(
            evaluation_set=EvaluationSet(**merged_dict),
            datapoint_splits=merged_splits,
        )

    @staticmethod
    def load_evaluation_sets_from_paths(eval_paths: list[Path]) -> list[LabelledEvaluationSet]:
        """Load one :class:`LabelledEvaluationSet` per file, labelled by its filename."""
        labelled_sets: list[LabelledEvaluationSet] = []
        for full_path in eval_paths:
            if not full_path.exists():
                raise FileNotFoundError(f"Evaluation set not found: {full_path}")
            eval_set = EvaluationLoader._load_eval_set_with_preprocessing(str(full_path))
            split_label = _infer_split_from_path(full_path)
            datapoint_splits = {item.name: split_label for item in (eval_set.evaluations or [])}
            labelled_sets.append(
                LabelledEvaluationSet(
                    evaluation_set=eval_set,
                    datapoint_splits=datapoint_splits,
                )
            )
        return labelled_sets

    @staticmethod
    def load_evaluation_set(
        cfg: DictConfig, agent_name: str, directory_provider: DirectoryProvider
    ) -> LabelledEvaluationSet:
        """Load and merge the eval sets in ``cfg.agent.evaluation_sets``.

        Raises:
            ValueError: If ``evaluation_sets`` is missing or empty.
            FileNotFoundError: If an eval-set file does not exist.
        """
        if hasattr(cfg.agent, "evaluation_sets"):
            eval_paths_raw = OmegaConf.to_container(cfg.agent.evaluation_sets, resolve=True)
            if not isinstance(eval_paths_raw, list) or len(eval_paths_raw) == 0:
                raise ValueError("evaluation_sets must be a non-empty list")

            eval_paths_raw = [
                directory_provider.solve_path(agent_name, file_path) for file_path in eval_paths_raw
            ]

            # An agent whose init has not run gets an actionable error. Checked only with an
            # init: block, since get_agent_dir raises for a provider without the agent's directory.
            from agents_gym_core.hydra_agent.agent_init import AgentInit

            if AgentInit.has_init(cfg):
                agent_dir = directory_provider.get_agent_dir(agent_name)
                AgentInit.check_or_raise(cfg, agent_name, agent_dir, directory_provider)

            labelled_sets = EvaluationLoader.load_evaluation_sets_from_paths(list(eval_paths_raw))
            return EvaluationLoader.merge_evaluation_sets(labelled_sets)

        raise ValueError("Property 'evaluation_sets' must be specified in agent config")

    @staticmethod
    def get_evaluators(
        agent_name: str, directory_provider: DirectoryProvider
    ) -> dict[str, BaseEvaluator]:
        """Load the agent's ``evaluator-*.json`` files, keyed by evaluator ID."""
        evaluators_dir_paths = directory_provider.solve_evaluators_paths(agent_name)

        evaluators: dict[str, BaseEvaluator] = {}

        for evaluators_dir in evaluators_dir_paths:
            if not evaluators_dir.exists():
                continue

            for file in evaluators_dir.glob("evaluator-*.json"):
                with open(file) as f:
                    spec = json.load(f)

                evaluator = EvaluatorFactory.create_evaluator(
                    spec,
                    evaluators_dir=evaluators_dir,
                )

                assert isinstance(evaluator, BaseEvaluator), (
                    f"Evaluator {evaluator.id} must be a BaseEvaluator instance."
                )
                assert evaluator.id not in evaluators, f"Evaluator: {evaluator.id} is duplicated."

                evaluators[evaluator.id] = evaluator

        return evaluators
