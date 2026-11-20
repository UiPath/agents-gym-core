"""Result aggregation and final inventory for evaluation runs."""

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from agents_gym_core.runtime.cost_tracking import (
    CostRecord,
    load_cost_record,
)
from agents_gym_core.runtime.run_utils import (
    AggregationMode,
    GraphDatapoint,
    get_datapoint_dir,
    get_experiment_path,
    print_experiment_inventory,
)
from agents_gym_core.runtime.runtime_config import RuntimeConfig
from agents_gym_core.runtime.runtime_helpers import RuntimeHelpers
from agents_gym_core.utils.bootstrap import bootstrap_confidence_interval
from agents_gym_core.utils.gym_types import Agent

# Groupby columns
_AGENT_COLS = ["loop_name", "loop_key", "agent", "agent_version"]
_AGENT_RUN_COLS = [*_AGENT_COLS, "run"]
_AGENT_RUN_METRIC_COLS = [*_AGENT_RUN_COLS, "evaluator_id"]
_AGENT_RUN_TESTCASE_COLS = [*_AGENT_RUN_COLS, "datapoint_name"]
_AGENT_RUN_TESTCASE_METRIC_COLS = [*_AGENT_RUN_TESTCASE_COLS, "evaluator_id"]
# Columns that are not counted as value columns
_METADATA_COLS = {"loop_name", "loop_key", "agent", "agent_version", "split", "run"}


def _run_order(run: Any) -> tuple[int, int | str]:
    """Sort key for run labels: ``run_2`` before ``run_10``."""
    number = str(run).rpartition("_")[2]
    return (0, int(number)) if number.isdigit() else (1, str(run))


def _parse_agent_key(agent_key: str) -> tuple[str, str]:
    """Split an ``<agent>__<version>`` key at the first ``__``; a version may contain ``__``."""
    name, separator, version = agent_key.partition("__")
    return name, version if separator else "base"


def _merge_cost_record_into(target: CostRecord, other: CostRecord) -> None:
    """Add *other* into *target* in place (summing datapoints within a run).

    ``driver`` becomes ``None`` when any merged record lacks one: a partial driver
    total would read as a complete one.
    """
    target.merge(other)
    if target.driver is None or other.driver is None:
        target.driver = None
    else:
        target.driver.merge(other.driver)


def _add_split_and_count_columns(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Insert a ``split`` column after the agent/run columns and a ``total_tags`` column
    counting the non-null value columns of each row."""
    split_idx = len(_AGENT_RUN_COLS)
    df.insert(split_idx, "split", split)
    value_cols = [col for col in df.columns if col not in _METADATA_COLS]
    df.insert(split_idx + 2, "total_tags", df[value_cols].notna().sum(axis=1))
    return df


class AgentCostSummary(BaseModel):
    """LLM cost of one agent and loop.

    ``total_cost_usd`` is the mean over runs of a run's cost summed over its datapoints.
    Token counts and ``num_llm_calls`` are per-run means in the same way.
    """

    total_cost_usd: float = Field(description="Mean USD cost per run (sum across datapoints)")
    cost_per_datapoint: float = Field(
        default=0.0,
        description=(
            "Average USD cost per datapoint: first mean across runs for each "
            "datapoint, then mean across datapoints."
        ),
    )
    input_tokens: float = Field(default=0.0, description="Mean input tokens per run")
    output_tokens: float = Field(default=0.0, description="Mean output tokens per run")
    cached_tokens: float = Field(default=0.0, description="Mean cached-read tokens per run")
    num_llm_calls: float = Field(default=0.0, description="Mean LLM calls per run")
    by_model: dict[str, float] = Field(
        default_factory=dict,
        description="Mean USD cost per run, broken down by model name",
    )
    num_datapoints: int = Field(
        default=0,
        description="Number of distinct datapoints that contributed at least one cost record",
    )
    num_runs: int = Field(
        default=0,
        description="Number of runs whose cost.json was found on disk",
    )
    total_experiment_cost_usd: float = Field(
        default=0.0,
        description="Total USD spent across all (datapoint, run) pairs",
    )
    datapoints_missing_cost: int = Field(
        default=0,
        description="Datapoint-run combinations lacking cost.json (e.g. runs cached without cost tracking)",
    )
    unpriced_calls: int = Field(
        default=0,
        description="LLM calls whose model is not in the price table (summed across runs)",
    )
    gateway_priced_calls: int = Field(
        default=0,
        description="LLM calls costed from the gateway's reported dollar cost (summed across runs)",
    )


class AgentScoreData(BaseModel):
    """Scores of one agent in one loop and split."""

    loop_name: str = Field(default="", description="Loop name")
    runs_mean: float = Field(description="Mean score across runs")
    runs_ci: list[float] = Field(description="Confidence interval for runs [lower, upper]")
    runs_whiskers: list[float] = Field(description="Whiskers for runs [min, max]")
    runs_error_margin: float = Field(
        description="Conservative error margin (max of upper/lower deviations)"
    )
    runs: dict[str, float] = Field(description="Individual run scores by run ID")

    # Set only in RUN_DATAPOINT aggregation mode
    datapoints_mean: float | None = Field(default=None, description="Mean score across datapoints")
    datapoints_ci: list[float] | None = Field(
        default=None, description="Confidence interval for datapoints [lower, upper]"
    )
    datapoints_whiskers: list[float] | None = Field(
        default=None, description="Whiskers for datapoints [min, max]"
    )
    datapoints: dict[str, float | list[float]] | None = Field(
        default=None,
        description="Individual datapoint scores and CIs",
    )

    # Set when some run of the agent has a cost.json
    cost: AgentCostSummary | None = Field(
        default=None,
        description="Per-agent LLM cost averaged across runs",
    )


def _prepare_evaluator_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Weight of each agent/run/evaluator from ``evaluator_weights`` (default 1)."""
    weights_df = df.groupby(_AGENT_RUN_METRIC_COLS)["evaluator_weights"].first().reset_index()
    weights_df["weight"] = weights_df.apply(
        lambda row: (
            row["evaluator_weights"].get(row["evaluator_id"], 1)
            if isinstance(row["evaluator_weights"], dict)
            else 1
        ),
        axis=1,
    )
    return weights_df


def _apply_weighted_aggregation(
    df: pd.DataFrame,
    groupby_cols: list[str],
    weights_df: pd.DataFrame | None,
    merge_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Mean ``score`` per ``groupby_cols``, weighted by evaluator weight.

    Weights are joined on ``merge_cols`` (default ``_AGENT_RUN_METRIC_COLS``); a missing
    weight counts as 1 and a group with zero total weight scores 0. With ``weights_df``
    None the mean is unweighted.
    """
    if weights_df is None:
        return df.groupby(groupby_cols)["score"].mean().reset_index()

    if merge_cols is None:
        merge_cols = _AGENT_RUN_METRIC_COLS

    weight_cols = [*merge_cols, "weight"]
    merged = df.merge(
        weights_df[weight_cols],
        on=merge_cols,
        how="left",
    )
    merged["weight"] = merged["weight"].fillna(1)
    merged["weighted_score"] = merged["score"] * merged["weight"]

    def _weighted_mean(x: pd.DataFrame) -> float:
        weight_sum = x["weight"].sum()
        return float(x["weighted_score"].sum() / weight_sum) if weight_sum > 0 else 0.0

    result = (
        merged.groupby(groupby_cols)
        .apply(
            _weighted_mean,
            include_groups=False,  # type: ignore[call-overload]
        )
        .reset_index()
    )
    result.columns = [*result.columns[:-1], "score"]  # type: ignore[assignment]
    return result


def _compute_testcase_weighted_scores(
    df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> pd.DataFrame:
    """Weighted score per agent/run/datapoint over the datapoint's per-evaluator means."""
    datapoint_metrics = df.groupby(_AGENT_RUN_TESTCASE_METRIC_COLS)["score"].mean().reset_index()

    return _apply_weighted_aggregation(
        datapoint_metrics,
        groupby_cols=_AGENT_RUN_TESTCASE_COLS,
        weights_df=weights_df,
    )


def _aggregate_evaluator_run(
    df: pd.DataFrame,
    use_metric_weights: bool,
    weights_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """EVALUATOR_RUN score per run: mean per evaluator, then a (weighted) mean over evaluators."""
    metric_per_agent_run = df.groupby(_AGENT_RUN_METRIC_COLS)["score"].mean().reset_index()

    return _apply_weighted_aggregation(
        metric_per_agent_run,
        groupby_cols=_AGENT_RUN_COLS,
        weights_df=weights_df if use_metric_weights else None,
    )


def _aggregate_datapoint_run(
    df: pd.DataFrame,
    use_metric_weights: bool,
    weights_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """DATAPOINT_RUN score per run: (weighted) score per datapoint, then the mean over them."""
    if use_metric_weights and weights_df is not None:
        datapoint_scores = _compute_testcase_weighted_scores(df, weights_df)
    else:
        datapoint_scores = df.groupby(_AGENT_RUN_TESTCASE_COLS)["score"].mean().reset_index()

    return datapoint_scores.groupby(_AGENT_RUN_COLS)["score"].mean().reset_index()


def _build_final_scores_with_ci(
    agent_score_per_run: pd.DataFrame,
    confidence_level: float,
) -> dict[str, dict[str, AgentScoreData]]:
    """``{loop_key: {agent_key: AgentScoreData}}`` with run scores and a bootstrap CI over runs."""
    final_scores: dict[str, dict[str, AgentScoreData]] = {}

    for (loop_name_, loop_key_, agent_, agent_version_), group in agent_score_per_run.groupby(
        _AGENT_COLS, sort=False
    ):
        loop_name = str(loop_name_)
        loop_key = str(loop_key_)
        agent = str(agent_)
        agent_version = str(agent_version_)
        runs_dict = dict(zip(group["run"].astype(str), group["score"]))
        run_scores = group["score"].tolist()
        stats = bootstrap_confidence_interval(run_scores, confidence_level=confidence_level)
        error_margin = max(stats.cihi - stats.mean, stats.mean - stats.cilo)

        agent_key = f"{agent}__{agent_version}"

        if loop_key not in final_scores:
            final_scores[loop_key] = {}

        final_scores[loop_key][agent_key] = AgentScoreData(
            loop_name=loop_name,
            runs_mean=stats.mean,
            runs_ci=[stats.cilo, stats.cihi],
            runs_whiskers=[stats.whislo, stats.whishi],
            runs_error_margin=error_margin,
            runs=runs_dict,
        )

    return final_scores


def _add_datapoint_scores_to_final(
    final_scores: dict[str, dict[str, AgentScoreData]],
    df: pd.DataFrame,
    use_metric_weights: bool,
    weights_df: pd.DataFrame | None,
    confidence_level: float,
) -> None:
    """RUN_DATAPOINT: add each datapoint's mean and CI over runs, then the CI over datapoints.

    Modifies ``final_scores`` in place.
    """
    if use_metric_weights and weights_df is not None:
        datapoint_scores_by_run = _compute_testcase_weighted_scores(df, weights_df)
    else:
        datapoint_scores_by_run = df.groupby(_AGENT_RUN_TESTCASE_COLS)["score"].mean().reset_index()

    agent_datapoint_cols = [*_AGENT_COLS, "datapoint_name"]
    for (
        _loop_name_,
        loop_key_,
        agent_,
        agent_version_,
        datapoint_name_,
    ), group in datapoint_scores_by_run.groupby(agent_datapoint_cols):
        loop_key = str(loop_key_)
        agent_key = f"{agent_}__{agent_version_}"
        datapoint_key = str(datapoint_name_)
        run_scores = group["score"].tolist()

        if loop_key in final_scores and agent_key in final_scores[loop_key]:
            agent_score_data = final_scores[loop_key][agent_key]
            if agent_score_data.datapoints is None:
                agent_score_data.datapoints = {}

            stats = bootstrap_confidence_interval(run_scores, confidence_level=confidence_level)
            agent_score_data.datapoints[datapoint_key] = stats.mean
            agent_score_data.datapoints[f"{datapoint_key}_ci"] = [stats.cilo, stats.cihi]
            agent_score_data.datapoints[f"{datapoint_key}_whiskers"] = [stats.whislo, stats.whishi]

    for agents_dict in final_scores.values():
        for agent_data in agents_dict.values():
            if agent_data.datapoints is not None:
                testcase_scores: list[float] = [
                    v  # type: ignore[misc]
                    for k, v in agent_data.datapoints.items()
                    if not k.endswith("_ci")
                    and not k.endswith("_whiskers")
                    and isinstance(v, float)
                ]
                if testcase_scores:
                    stats = bootstrap_confidence_interval(
                        testcase_scores, confidence_level=confidence_level
                    )
                    agent_data.datapoints_mean = stats.mean
                    agent_data.datapoints_ci = [stats.cilo, stats.cihi]
                    agent_data.datapoints_whiskers = [stats.whislo, stats.whishi]


def _build_metrics_dataframe(
    df: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    """One row per agent/run with each evaluator's mean score as a column."""
    metrics_per_run = pd.pivot_table(
        df,
        values="score",
        index=_AGENT_RUN_COLS,
        columns=["evaluator_id"],
        aggfunc="mean",
    ).reset_index()

    return _add_split_and_count_columns(metrics_per_run, split)


def _build_datapoints_dataframe(
    df: pd.DataFrame,
    split: str,
    use_metric_weights: bool,
    weights_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """One row per agent/run with each datapoint's (weighted) score as a column."""
    datapoint_scores = _apply_weighted_aggregation(
        df,
        groupby_cols=_AGENT_RUN_TESTCASE_COLS,
        weights_df=weights_df if use_metric_weights else None,
    )

    datapoint_aggregated = pd.pivot_table(
        datapoint_scores,
        values="score",
        index=_AGENT_RUN_COLS,
        columns=["datapoint_name"],
        aggfunc="mean",
    ).reset_index()

    if isinstance(datapoint_aggregated.columns, pd.MultiIndex):
        datapoint_aggregated.columns = [
            col[1] if col[0] == "" else col[0] for col in datapoint_aggregated.columns
        ]

    return _add_split_and_count_columns(datapoint_aggregated, split)


def calculate_agent_scores(
    df: pd.DataFrame,
    use_metric_weights: bool = True,
    agg: AggregationMode = AggregationMode.DATAPOINT_RUN,
    confidence_level: float = 0.95,
    max_runs_to_plot: int = -1,
) -> tuple[dict[str, dict[str, dict[str, AgentScoreData]]], pd.DataFrame, pd.DataFrame]:
    """Aggregate one row per score into final scores with bootstrap CIs, for all data and
    for each named split.

    Args:
        df: One row per (agent, loop, evaluator, datapoint, run) score.
        max_runs_to_plot: If > 0, keep only the first that many runs of each agent and loop.

    Returns:
        ``({loop_key: {agent_key: {split: AgentScoreData}}}, metrics_df, datapoints_df)``.
    """

    splits = ["all_data"]
    # Any named split (e.g. "train") is scored on its own besides "all_data".
    if "split" in df.columns:
        unique_splits = [s for s in df["split"].unique().tolist() if s != "all_data"]
        if len(unique_splits) >= 1:
            splits = ["all_data"] + unique_splits

    final_score_per_agent_all: dict[str, dict[str, dict[str, AgentScoreData]]] = {}
    df_agg_results_on_metrics_all = pd.DataFrame()
    df_agg_results_on_datapoints_all = pd.DataFrame()

    for split in splits:
        if split == "all_data":
            split_df = df.copy()
        else:
            split_df = df[df["split"] == split].copy()

        if max_runs_to_plot > 0:
            runs_to_keep_df = (
                split_df.groupby(_AGENT_COLS)["run"]
                .apply(lambda x: pd.Series(sorted(x.unique(), key=_run_order)[:max_runs_to_plot]))
                .reset_index()
                .drop(columns=["level_4"])
                .rename(columns={0: "run"})
            )
            split_df_tuples = split_df[_AGENT_RUN_COLS].apply(tuple, axis=1)
            keep_tuples = runs_to_keep_df[_AGENT_RUN_COLS].apply(tuple, axis=1)
            split_df = split_df[split_df_tuples.isin(keep_tuples)]

        weights_df = _prepare_evaluator_weights(split_df) if use_metric_weights else None

        if agg == AggregationMode.EVALUATOR_RUN:
            agent_score_per_run = _aggregate_evaluator_run(split_df, use_metric_weights, weights_df)
        elif agg == AggregationMode.DATAPOINT_RUN:
            agent_score_per_run = _aggregate_datapoint_run(split_df, use_metric_weights, weights_df)
        else:  # AggregationMode.RUN_DATAPOINT
            agent_score_per_run = _aggregate_datapoint_run(split_df, use_metric_weights, weights_df)

        agent_scores = _build_metrics_dataframe(split_df, split)
        final_scores = _build_final_scores_with_ci(agent_score_per_run, confidence_level)

        if agg == AggregationMode.RUN_DATAPOINT:
            _add_datapoint_scores_to_final(
                final_scores, split_df, use_metric_weights, weights_df, confidence_level
            )

        datapoint_aggregated = _build_datapoints_dataframe(
            split_df, split, use_metric_weights, weights_df
        )

        df_agg_results_on_metrics_all = pd.concat([df_agg_results_on_metrics_all, agent_scores])
        df_agg_results_on_datapoints_all = pd.concat(
            [df_agg_results_on_datapoints_all, datapoint_aggregated]
        )

        for loop_key, agents_dict in final_scores.items():
            if loop_key not in final_score_per_agent_all:
                final_score_per_agent_all[loop_key] = {}
            for agent_key, score_data in agents_dict.items():
                if agent_key not in final_score_per_agent_all[loop_key]:
                    final_score_per_agent_all[loop_key][agent_key] = {}
                final_score_per_agent_all[loop_key][agent_key][split] = score_data

    final_score_per_agent_all = dict(sorted(final_score_per_agent_all.items()))

    df_agg_results_on_metrics_all = df_agg_results_on_metrics_all.sort_values(
        by=["agent", "agent_version", "split"]
    ).reset_index(drop=True)

    df_agg_results_on_datapoints_all = df_agg_results_on_datapoints_all.sort_values(
        by=["agent", "agent_version", "split", "run"]
    ).reset_index(drop=True)

    return (
        final_score_per_agent_all,
        df_agg_results_on_metrics_all,
        df_agg_results_on_datapoints_all,
    )


def final_scores_json(final_scores: dict[str, Any]) -> str:
    """final_score_per_agent.json's text: indented JSON with lists of numbers on one line."""
    json_str = json.dumps(final_scores, indent=2)
    json_str = re.sub(r"\[\s+", "[", json_str)
    json_str = re.sub(r"\s+\]", "]", json_str)
    return re.sub(r"(\d),\s+(\d)", r"\1, \2", json_str)


class ResultAggregator:
    """Aggregates scores and costs, prints the final inventory and saves the results."""

    def __init__(
        self,
        graphs_list: list[GraphDatapoint],
        all_datapoint_names: list[str],
        loaded_agents: list[Agent],
        config: RuntimeConfig,
    ):
        self.graphs_list = graphs_list
        self.all_datapoint_names = all_datapoint_names
        self.loaded_agents = loaded_agents
        self.config = config
        self.helpers = RuntimeHelpers(config)

    def print_final_inventory(self) -> None:
        """Print the experiment inventory after all runs."""
        if not self.all_datapoint_names or not self.loaded_agents:
            return

        graphs_without_run = [(g, d, a) for g, d, a, _ in self.graphs_list]
        experiment_dirs, agents_to_inventory, datapoint_names = (
            self.helpers.compute_inventory_targets(graphs_without_run)
        )
        if experiment_dirs:
            print_experiment_inventory(
                experiment_dirs,
                agents_to_inventory,
                self.config.max_datapoints,
                datapoint_names=datapoint_names,
                mode=self.config.mode,
                num_runs=self.config.num_runs,
                is_final=True,
            )

    def _convert_evaluation_results_to_dataframe(
        self,
        evaluation_results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]],
        agent_evaluator_weights: dict[str, dict[str, float]] | None = None,
    ) -> pd.DataFrame:
        """One row per score from agent_key -> loop_key -> evaluator_id -> datapoint_key ->
        result_data; ``agent_evaluator_weights`` maps agent_key -> evaluator_id -> weight."""
        weights_map = agent_evaluator_weights or {}
        rows = []
        for agent_key, loop_results in evaluation_results.items():
            agent_name, agent_version = _parse_agent_key(agent_key)
            agent_weights = weights_map.get(agent_key, {})
            for loop_key, evaluator_results in loop_results.items():
                for evaluator_id, datapoint_results in evaluator_results.items():
                    for datapoint_name, result_data in datapoint_results.items():
                        result = result_data.get("result")

                        if hasattr(result, "score"):
                            score = result.score  # type: ignore
                        else:
                            continue

                        if ":run_" in datapoint_name:
                            clean_datapoint_name = datapoint_name.rsplit(":run_", 1)[0]
                            assert datapoint_name.rsplit(":run_", 1)[1] == str(
                                result_data.get("run_number")
                            )
                        else:
                            clean_datapoint_name = datapoint_name

                        loop_details = result_data.get("loop_details", {})
                        loop_name = loop_details.get("loop_name", "")
                        loop_version = loop_details.get("loop_version", "")
                        loop_models = loop_details.get("models", {})

                        rows.append(
                            {
                                "agent": agent_name,
                                "agent_version": agent_version,
                                "loop_name": loop_name,
                                "loop_version": loop_version,
                                "loop_key": loop_key,
                                "loop_models": loop_models,
                                "evaluator_id": evaluator_id,
                                "datapoint_name": clean_datapoint_name,
                                "score": float(score),
                                "run": f"run_{result_data.get('run_number', '1')}",
                                "evaluator_weights": agent_weights,
                                "split": result_data.get("split", "all_data"),
                            }
                        )

        return pd.DataFrame(rows)

    def aggregate_costs_per_agent(self) -> dict[str, dict[str, AgentCostSummary]]:
        """Read cost.json of every expected datapoint run and average each agent's cost over runs.

        Datapoint runs without a cost.json count in ``datapoints_missing_cost``, which tells
        partial coverage apart from runs that cost nothing.

        Returns:
            ``{loop_key: {agent_key: AgentCostSummary}}``, without agents that have no cost
            records.
        """
        # (loop_key, agent_key, run_number) -> rolled-up CostRecord for that run
        run_records: dict[tuple[str, str, int], CostRecord] = {}
        # (loop_key, agent_key, datapoint_name) -> list of per-run total_cost_usd
        datapoint_costs: dict[tuple[str, str, str], list[float]] = defaultdict(list)
        # (loop_key, agent_key) -> set of run numbers expected from graphs_list
        expected_runs: dict[tuple[str, str], set[int]] = defaultdict(set)
        # (loop_key, agent_key, run_number) -> count of datapoints without cost.json
        missing_counts: dict[tuple[str, str, int], int] = defaultdict(int)

        for graph, datapoint, agent, run_number in self.graphs_list:
            loop_details = graph.loop_details  # pyright: ignore[reportAttributeAccessIssue]
            loop_key = loop_details.loop_key
            agent_key = f"{agent.name}__{agent.version}"
            expected_runs[(loop_key, agent_key)].add(run_number)

            experiment_dir = get_experiment_path(
                base_output_dir=self.config.base_output_dir,
                experiment_name=self.config.experiment_name,
                agent=agent,
                loop_details=loop_details,
            )
            datapoint_dir = get_datapoint_dir(experiment_dir, datapoint.name, run_number)
            record = load_cost_record(datapoint_dir)
            if record is None:
                missing_counts[(loop_key, agent_key, run_number)] += 1
                continue

            datapoint_costs[(loop_key, agent_key, datapoint.name)].append(record.total_cost_usd)

            run_key = (loop_key, agent_key, run_number)
            rolled = run_records.get(run_key)
            if rolled is None:
                run_records[run_key] = record.model_copy(deep=True)
            else:
                _merge_cost_record_into(rolled, record)

        summaries: dict[str, dict[str, AgentCostSummary]] = {}
        for (loop_key, agent_key), runs in expected_runs.items():
            per_run_records = {
                r: run_records[(loop_key, agent_key, r)]
                for r in runs
                if (loop_key, agent_key, r) in run_records
            }
            if not per_run_records:
                continue

            num_runs = len(per_run_records)
            mean_cost = sum(rec.total_cost_usd for rec in per_run_records.values()) / num_runs
            mean_in = sum(rec.input_tokens for rec in per_run_records.values()) / num_runs
            mean_out = sum(rec.output_tokens for rec in per_run_records.values()) / num_runs
            mean_cached = sum(rec.cached_tokens for rec in per_run_records.values()) / num_runs
            mean_calls = sum(rec.num_llm_calls for rec in per_run_records.values()) / num_runs
            unpriced = sum(rec.unpriced_calls for rec in per_run_records.values())
            gateway_priced = sum(rec.gateway_priced_calls for rec in per_run_records.values())

            by_model_sum: dict[str, float] = defaultdict(float)
            for rec in per_run_records.values():
                for model_name, usage in rec.by_model.items():
                    by_model_sum[model_name] += usage.cost_usd
            by_model = {m: round(total / num_runs, 8) for m, total in by_model_sum.items()}

            per_dp_means = [
                sum(costs) / len(costs)
                for (lk, ak, _dp), costs in datapoint_costs.items()
                if lk == loop_key and ak == agent_key and costs
            ]
            cost_per_datapoint = (
                round(sum(per_dp_means) / len(per_dp_means), 8) if per_dp_means else 0.0
            )
            total_experiment_cost = round(
                sum(
                    c
                    for (lk, ak, _dp), costs in datapoint_costs.items()
                    if lk == loop_key and ak == agent_key
                    for c in costs
                ),
                8,
            )

            # Every datapoint run without cost.json counts, even in runs where others have one.
            datapoints_missing = sum(
                count
                for (lk, ak, _r), count in missing_counts.items()
                if lk == loop_key and ak == agent_key
            )

            summaries.setdefault(loop_key, {})[agent_key] = AgentCostSummary(
                total_cost_usd=round(mean_cost, 8),
                cost_per_datapoint=cost_per_datapoint,
                input_tokens=round(mean_in, 2),
                output_tokens=round(mean_out, 2),
                cached_tokens=round(mean_cached, 2),
                num_llm_calls=round(mean_calls, 2),
                by_model=by_model,
                num_datapoints=len(per_dp_means),
                num_runs=num_runs,
                total_experiment_cost_usd=total_experiment_cost,
                datapoints_missing_cost=datapoints_missing,
                unpriced_calls=unpriced,
                gateway_priced_calls=gateway_priced,
            )

        return summaries

    def _get_evaluator_weights_for_agent(
        self, agent_name: str, agent_version: str
    ) -> dict[str, float]:
        """evaluator_id -> weight of a loaded agent; empty if the agent is not loaded."""
        for agent in self.loaded_agents:
            if agent.name == agent_name and agent.version == agent_version:
                return agent.evaluator_weights
        return {}

    def aggregate_scores(
        self,
        evaluation_results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]],
        use_metric_weights: bool = True,
        agg: AggregationMode = AggregationMode.DATAPOINT_RUN,
        confidence_level: float = 0.95,
    ) -> tuple[dict[str, dict[str, dict[str, AgentScoreData]]], pd.DataFrame, pd.DataFrame]:
        """Aggregate evaluation results with :func:`calculate_agent_scores`, using each agent's
        evaluator weights."""
        agent_weights: dict[str, dict[str, float]] = {}
        for agent_key in evaluation_results.keys():
            agent_name, agent_version = _parse_agent_key(agent_key)
            agent_weights[agent_key] = self._get_evaluator_weights_for_agent(
                agent_name, agent_version
            )

        df = self._convert_evaluation_results_to_dataframe(evaluation_results, agent_weights)

        if df.empty:
            return {}, pd.DataFrame(), pd.DataFrame()

        return calculate_agent_scores(
            df,
            use_metric_weights=use_metric_weights,
            agg=agg,
            confidence_level=confidence_level,
        )

    def finalize(
        self,
        evaluation_results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]],
        use_metric_weights: bool = True,
        agg: AggregationMode = AggregationMode.DATAPOINT_RUN,
        confidence_level: float = 0.95,
        output_dir: Path | str | None = None,
    ) -> dict[str, Any]:
        """Print the final inventory, aggregate scores, attach per-agent cost and save.

        Files are written only when ``output_dir`` is given and some scores exist.

        Returns:
            A dict with ``evaluation_results``, ``aggregated_scores``, ``metrics_df``,
            ``datapoints_df``, ``costs_df`` and, when files were written, ``saved_files``.
        """
        self.print_final_inventory()

        result: dict[str, Any] = {
            "evaluation_results": evaluation_results,
        }

        aggregated_scores, metrics_df, datapoints_df = self.aggregate_scores(
            evaluation_results,
            use_metric_weights=use_metric_weights,
            agg=agg,
            confidence_level=confidence_level,
        )
        result["aggregated_scores"] = aggregated_scores
        result["metrics_df"] = metrics_df
        result["datapoints_df"] = datapoints_df

        # Every split of an agent gets the same cost summary. Without one, cost stays None
        # and is left out of the JSON.
        cost_summaries = self.aggregate_costs_per_agent()
        result["costs_df"] = pd.DataFrame(
            [
                {
                    "loop_key": loop_key,
                    "loop_name": loop_key.split("__", 1)[0],
                    "agent": agent_key,
                    **summary.model_dump(exclude={"by_model"}),
                }
                for loop_key, agents in cost_summaries.items()
                for agent_key, summary in agents.items()
            ]
        )
        for loop_key, agents_dict in aggregated_scores.items():
            summaries_for_loop = cost_summaries.get(loop_key, {})
            for agent_key, splits_dict in agents_dict.items():
                summary = summaries_for_loop.get(agent_key)
                if summary is None:
                    continue
                for score_data in splits_dict.values():
                    score_data.cost = summary

        if aggregated_scores:
            self._print_score_summary(aggregated_scores, confidence_level)

            if output_dir is not None:
                saved_files = self.save_results(
                    evaluation_results=evaluation_results,
                    aggregated_scores=aggregated_scores,
                    metrics_df=metrics_df,
                    datapoints_df=datapoints_df,
                    output_dir=Path(output_dir),
                )
                result["saved_files"] = saved_files

        return result

    def _print_score_summary(
        self,
        aggregated_scores: dict[str, dict[str, dict[str, AgentScoreData]]],
        confidence_level: float = 0.95,
    ) -> None:
        """Print a summary of aggregated scores."""
        print("\n" + "=" * 60)
        print("AGGREGATED SCORES SUMMARY")
        print("=" * 60)

        for loop_key, agents_dict in aggregated_scores.items():
            print(f"\nLoop: {loop_key}")
            print("-" * 40)

            for agent_key, splits_dict in agents_dict.items():
                agent_name, agent_version = _parse_agent_key(agent_key)
                print(f"  Agent: {agent_name} ({agent_version})")

                for split, score_data in splits_dict.items():
                    ci_lo, ci_hi = score_data.runs_ci
                    n_runs = len(score_data.runs)
                    print(
                        f"    [{split}] Score: {score_data.runs_mean:.4f} ± {score_data.runs_error_margin:.4f} "
                        f"({confidence_level:.0%} CI: [{ci_lo:.4f}, {ci_hi:.4f}], n={n_runs} runs)"
                    )
                    if score_data.cost is not None:
                        cost = score_data.cost
                        note = (
                            f" ({cost.datapoints_missing_cost} dp-runs lack cost data)"
                            if cost.datapoints_missing_cost
                            else ""
                        )
                        print(
                            f"      Cost/run (mean of {cost.num_runs}): "
                            f"${cost.total_cost_usd:.6f}{note}"
                        )

        print("=" * 60 + "\n")

    def save_results(
        self,
        evaluation_results: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]],
        aggregated_scores: dict[str, dict[str, dict[str, AgentScoreData]]],
        metrics_df: pd.DataFrame,
        datapoints_df: pd.DataFrame,
        output_dir: Path,
        decimals: int = 4,
    ) -> dict[str, Path]:
        """Write the results to ``output_dir``, rounding floats to ``decimals`` places.

        Returns the paths keyed ``combined_results`` (combined_results.json),
        ``final_scores`` (final_score_per_agent.json), ``metrics``
        (agg_results_on_metrics.csv) and ``datapoints`` (agg_results_on_datapoints.csv).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files: dict[str, Path] = {}

        # combined_results.json entries use the EvaluatorDatapointResult field names, so
        # they load directly into that schema.
        combined_results: list[dict[str, Any]] = []
        for agent_key, loop_results in evaluation_results.items():
            agent_name, agent_version = _parse_agent_key(agent_key)
            evaluator_weights = self._get_evaluator_weights_for_agent(agent_name, agent_version)
            for loop_key, evaluator_results in loop_results.items():
                for evaluator_id, datapoint_results in evaluator_results.items():
                    for datapoint_key, result_data in datapoint_results.items():
                        result = result_data.get("result")

                        if result is None:
                            continue
                        score = float(result.score) if hasattr(result, "score") else None
                        details_raw = result.details if hasattr(result, "details") else None
                        if details_raw and hasattr(details_raw, "model_dump"):
                            details = details_raw.model_dump()
                        else:
                            details = details_raw

                        clean_datapoint_name = (
                            datapoint_key.rsplit(":run_", 1)[0]
                            if ":run_" in datapoint_key
                            else datapoint_key
                        )

                        loop_details = result_data.get("loop_details", {})
                        loop_name = loop_details.get("loop_name")
                        loop_version = loop_details.get("loop_version", "")
                        loop_models = loop_details.get("models", {})
                        run_number = result_data.get("run_number", 1)

                        experiment_id = f"{agent_name}__{agent_version}_{loop_key}_{clean_datapoint_name}:run_{run_number}"

                        result_entry: dict[str, Any] = {
                            "experiment_unique_identifier": experiment_id,
                            "agent": agent_name,
                            "agent_version": agent_version,
                            "loop_name": loop_name,
                            "loop_version": loop_version,
                            "loop_key": loop_key,
                            "loop_models": loop_models,
                            "datapoint_name": clean_datapoint_name,
                            "score": round(score, decimals) if score is not None else None,
                            "evaluator_id": evaluator_id,
                            "agent_output": result_data.get("agent_output"),
                            "evaluation_criteria": result_data.get("evaluation_criteria"),
                            "details": details,
                            "evaluator_weights": evaluator_weights,
                            "run_number": run_number,
                            "split": result_data.get("split", "all_data"),
                        }

                        # Judge fields only when set, as in the per-datapoint result files.
                        if result_data.get("judge_model"):
                            result_entry["judge_model"] = result_data["judge_model"]
                        if result_data.get("judge_prompt"):
                            result_entry["judge_prompt"] = result_data["judge_prompt"]

                        combined_results.append(result_entry)

        combined_results_file = output_dir / "combined_results.json"
        with open(combined_results_file, "w") as f:
            json.dump(combined_results, f, indent=2)
        saved_files["combined_results"] = combined_results_file

        if not metrics_df.empty:
            numeric_cols = metrics_df.select_dtypes(include=["float64", "float32"]).columns
            if len(numeric_cols) > 0:
                metrics_df = metrics_df.copy()
                metrics_df[numeric_cols] = metrics_df[numeric_cols].round(decimals)

        if not datapoints_df.empty:
            numeric_cols = datapoints_df.select_dtypes(include=["float64", "float32"]).columns
            if len(numeric_cols) > 0:
                datapoints_df = datapoints_df.copy()
                datapoints_df[numeric_cols] = datapoints_df[numeric_cols].round(decimals)

        def round_nested_dict(d: dict, decimal_places: int) -> dict:
            result = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    result[k] = round_nested_dict(v, decimal_places)
                elif isinstance(v, list):
                    result[k] = [round(x, decimal_places) if isinstance(x, float) else x for x in v]
                elif isinstance(v, float):
                    result[k] = round(v, decimal_places)
                else:
                    result[k] = v
            return result

        final_scores_dict: dict[str, Any] = {}
        for loop_key, agents_dict in aggregated_scores.items():
            final_scores_dict[loop_key] = {}
            for agent_key, splits_dict in agents_dict.items():
                final_scores_dict[loop_key][agent_key] = {}
                for split_key, agent_score_data in splits_dict.items():
                    final_scores_dict[loop_key][agent_key][split_key] = agent_score_data.model_dump(
                        exclude_none=True
                    )

        final_scores_dict = round_nested_dict(final_scores_dict, decimals)

        scores_file = output_dir / "final_score_per_agent.json"
        scores_file.write_text(final_scores_json(final_scores_dict))
        saved_files["final_scores"] = scores_file

        metrics_file = output_dir / "agg_results_on_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        saved_files["metrics"] = metrics_file

        datapoints_file = output_dir / "agg_results_on_datapoints.csv"
        datapoints_df.to_csv(datapoints_file, index=False)
        saved_files["datapoints"] = datapoints_file

        print(f"\nSaved aggregated results to {output_dir}:")
        print(f"  - {combined_results_file.name}")
        print(f"  - {scores_file.name}")
        print(f"  - {metrics_file.name}")
        print(f"  - {datapoints_file.name}")

        return saved_files
