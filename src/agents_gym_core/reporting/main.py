import colorsys
import heapq
from typing import Literal

import numpy as np
import pandas as pd
from IPython.display import HTML, display
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from strictly_typed_pandas.dataset import DataSet

from agents_gym_core.reporting.loop_analysis import _apply_bootstrap_ci_to_boxplot
from agents_gym_core.reporting.schemas import (
    AgentScorePerRunSchema,
    AgentScoreSummarySchema,
    EvaluatorDatapointResultSchema,
)
from agents_gym_core.reporting.utils import (
    display_df,
    extract_optimization_algorithm,
    get_models_color_palette,
    highlight_best_per_agent,
    highlight_max_per_series,
)
from agents_gym_core.runtime.run_utils import AggregationMode
from agents_gym_core.runtime.runners.result_aggregator import (
    calculate_agent_scores,
)
from agents_gym_core.utils.bootstrap import BootstrapStatistics


def calculate_agent_scores_for_report(
    df: DataSet[EvaluatorDatapointResultSchema],
    use_metric_weights: bool = True,
    agg: AggregationMode = AggregationMode.DATAPOINT_RUN,
    ci_confidence_level: float = 0.95,
    max_runs_to_plot: int = -1,
) -> tuple[DataSet[AgentScoreSummarySchema], DataSet[AgentScorePerRunSchema]]:
    """Score results with ``calculate_agent_scores`` (the function behind
    ``final_score_per_agent.json``) and return ``(summary_df, per_run_df)``.

    The scoring options are passed through to ``calculate_agent_scores``. Only the
    ``all_data`` split is reported.
    """
    df_processed = pd.DataFrame(df)

    if "loop_key" not in df_processed.columns and "loop_version" in df_processed.columns:
        df_processed = df_processed.rename(columns={"loop_version": "loop_key"})

    if "run" not in df_processed.columns:
        # experiment_unique_identifier ends with "run_<n>".
        if "experiment_unique_identifier" in df_processed.columns:
            df_processed["run"] = (
                df_processed["experiment_unique_identifier"]
                .str.extract(r":?(run_\d+)$")[0]
                .fillna("run_1")
            )
        else:
            df_processed["run"] = "run_1"

    # Aggregation drops these columns, so look them up per loop_key first.
    loop_models_lookup: dict[str, dict] = {}
    if "loop_models" in df_processed.columns:
        for _, row in (
            df_processed[["loop_key", "loop_models"]].drop_duplicates("loop_key").iterrows()
        ):
            lm = row["loop_models"]
            if isinstance(lm, dict):
                loop_models_lookup[row["loop_key"]] = lm

    loop_version_lookup: dict[str, str] = {}
    if "loop_version" in df_processed.columns:
        for _, row in (
            df_processed[["loop_key", "loop_version"]].drop_duplicates("loop_key").iterrows()
        ):
            loop_version_lookup[row["loop_key"]] = str(row["loop_version"])

    final_scores_dict, _, _ = calculate_agent_scores(
        df_processed,
        use_metric_weights=use_metric_weights,
        agg=agg,
        confidence_level=ci_confidence_level,
        max_runs_to_plot=max_runs_to_plot,
    )

    summary_data = []
    per_run_data = []

    for loop_key, agents_dict in final_scores_dict.items():
        loop_models = loop_models_lookup.get(loop_key, {})
        loop_version = loop_version_lookup.get(loop_key, "")
        # "<main model>,<role>:<model>,..."
        primary = loop_models.get("model", "")
        others = sorted(f"{k}:{v}" for k, v in loop_models.items() if k != "model")
        model = ",".join([primary] + others) if primary else ",".join(others)

        for agent_key, splits_dict in agents_dict.items():
            if isinstance(splits_dict, dict) and "all_data" in splits_dict:
                all_data = splits_dict["all_data"]
                # agent_key is "{agent}__{agent_version}"; a version may contain "__".
                agent, _, agent_version = agent_key.partition("__")
                agent_version = agent_version or "base"

                mean_score = all_data.runs_mean
                error_margin = all_data.runs_error_margin
                cilo, cihi = all_data.runs_ci
                whislo, whishi = all_data.runs_whiskers
                summary_data.append(
                    {
                        "model": model,
                        "loop_name": all_data.loop_name,
                        "loop_version": loop_version,
                        "loop_key": loop_key,
                        "agent": agent,
                        "agent_version": agent_version,
                        "score": f"{mean_score:.3f} ± {error_margin:.3f}",
                        "mean": mean_score,
                        "cilo": cilo,
                        "cihi": cihi,
                        "whislo": whislo,
                        "whishi": whishi,
                    }
                )

                for run, score in all_data.runs.items():
                    per_run_data.append(
                        {
                            "model": model,
                            "loop_name": all_data.loop_name,
                            "loop_version": loop_version,
                            "loop_key": loop_key,
                            "agent": agent,
                            "agent_version": agent_version,
                            "run": run,
                            "score": score,
                        }
                    )

    summary_df = DataSet[AgentScoreSummarySchema](summary_data)
    per_run_df = DataSet[AgentScorePerRunSchema](per_run_data)
    return summary_df, per_run_df


def pivot_on_loops(df: pd.DataFrame, loop_on_column: bool = True) -> None:
    loop_cols = ["loop_name", "loop_key"]

    if loop_on_column:
        index = ["agent", "agent_version"]
        columns = loop_cols
        axis = 1
    else:
        index = loop_cols
        columns = ["agent", "agent_version"]
        axis = 0

    pivot = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values="score",
        aggfunc="first",  # pyright: ignore[reportArgumentType, reportCallIssue]
    )

    styled = pivot.style.format(na_rep="-", escape="html").apply(
        highlight_max_per_series, axis=axis
    )
    display_df(styled)


def pivot_on_versions(df: pd.DataFrame, pivot_type: Literal["loop", "agent"] = "loop") -> None:
    loop_cols = ["loop_name", "loop_key"]

    if pivot_type == "loop":
        index = ["agent", "agent_version"]
        columns = loop_cols
    elif pivot_type == "agent":
        index = loop_cols
        columns = ["agent", "agent_version"]

    pivot = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values="score",
        aggfunc="first",  # pyright: ignore[reportArgumentType, reportCallIssue]
    )
    # With versions on the rows ("loop"), find the best version per agent on the transpose.
    styled = pivot.style.format(na_rep="-", escape="html").apply(
        lambda d: (
            highlight_best_per_agent(d.T).T if pivot_type == "loop" else highlight_best_per_agent(d)
        ),
        axis=None,
    )
    display_df(styled)


def pivot_agent_versions_average(df: pd.DataFrame) -> None:
    """Display, per loop, the mean score across agents for each agent_version."""
    loop_cols = ["loop_name", "loop_key"]

    df_numeric = df.copy()
    df_numeric["score_numeric"] = df_numeric["score"].str.split(" ± ").str[0].astype(float)

    average_pivot = pd.pivot_table(
        df_numeric,
        index=loop_cols,
        columns=["agent_version"],
        values="score_numeric",
        aggfunc="mean",  # pyright: ignore[reportArgumentType, reportCallIssue]
    )
    styled_average = average_pivot.style.format("{:.3f}", na_rep="-", escape="html").apply(
        highlight_max_per_series,
        axis=1,  # best agent_version per loop
    )
    display_df(styled_average)


def _determine_best_clustering(
    stats: pd.DataFrame, agent_versions: list[str]
) -> Literal["none", "version", "model", "version_model"]:
    """Pick a clustering mode from the number of agent versions and models.

    When both vary, two-layer clustering is used only with at least 3 loops per version.
    """
    num_versions = len(agent_versions)
    num_models = stats["model"].nunique()
    num_loops = len(stats)

    if num_versions == 1 and num_models == 1:
        return "none"

    if num_versions > 1 and num_models == 1:
        return "version"

    if num_versions == 1 and num_models > 1:
        return "model"

    if num_versions > 1 and num_models > 1:
        loops_per_version = num_loops / num_versions
        if loops_per_version < 3:
            return "version"
        return "version_model"

    return "none"


def _cluster_by_version(
    stats: pd.DataFrame, agent_versions: list[str]
) -> tuple[list[str], list[dict]]:
    """Order versions by their best loop, and loops by score within each version.

    Returns the ordered keys and one improvement line per version with several loops.
    """
    version_best_scores = stats.groupby("agent_version")["mean"].max().reset_index()
    version_best_scores.columns = ["agent_version", "best_mean"]
    version_best_scores = version_best_scores.sort_values("best_mean", ascending=True)
    ordered_versions = version_best_scores["agent_version"].tolist()

    ordered_keys = []
    improvement_lines = []
    current_position = 0

    for version in ordered_versions:
        version_stats = stats[stats["agent_version"] == version].copy()
        version_stats = version_stats.sort_values("mean", ascending=True)

        version_keys = version_stats["agent_loop_key"].tolist()
        ordered_keys.extend(version_keys)

        if len(version_stats) > 1:
            min_score = version_stats["mean"].min()
            max_score = version_stats["mean"].max()
            improvement_pct = ((max_score - min_score) / min_score * 100) if min_score > 0 else 0

            improvement_lines.append(
                {
                    "start_pos": current_position,
                    "end_pos": current_position + len(version_keys) - 1,
                    "min_score": min_score,
                    "max_score": max_score,
                    "improvement_pct": improvement_pct,
                    "version": version,
                }
            )

        current_position += len(version_keys)

    return ordered_keys, improvement_lines


def _cluster_by_model(stats: pd.DataFrame) -> tuple[list[str], list[dict]]:
    """Order models by their best loop; within a model, sort by score, non-optimized first.

    Returns the ordered keys and one improvement line per model with several loops.
    """
    model_best_scores = stats.groupby("model")["mean"].max().reset_index()
    model_best_scores.columns = ["model", "best_mean"]
    model_best_scores = model_best_scores.sort_values("best_mean", ascending=True)
    ordered_models = model_best_scores["model"].tolist()

    ordered_keys = []
    improvement_lines = []
    current_position = 0

    for model in ordered_models:
        model_stats = stats[stats["model"] == model].copy()
        model_stats["has_optimization_algorithm"] = model_stats["loop_key"].apply(
            lambda x: extract_optimization_algorithm(x) != "Default"
        )
        model_stats = model_stats.sort_values(
            ["has_optimization_algorithm", "mean"], ascending=True
        )

        model_keys = model_stats["agent_loop_key"].tolist()

        ordered_keys.extend(model_keys)

        if len(model_stats) > 1:
            min_score = model_stats.iloc[0]["mean"]
            max_score = model_stats.iloc[-1]["mean"]
            improvement_pct = ((max_score - min_score) / min_score * 100) if min_score > 0 else 0

            improvement_lines.append(
                {
                    "start_pos": current_position,
                    "end_pos": current_position + len(model_keys) - 1,
                    "min_score": min_score,
                    "max_score": max_score,
                    "improvement_pct": improvement_pct,
                    "model": model,
                }
            )

        current_position += len(model_keys)

    return ordered_keys, improvement_lines


def _cluster_by_version_and_model(
    stats: pd.DataFrame, agent_versions: list[str]
) -> tuple[list[str], list[dict]]:
    """Two-layer clustering: by agent version, then by model within each version.

    Returns the ordered keys and one improvement line per (version, model) with several loops.
    """
    version_best_scores = stats.groupby("agent_version")["mean"].max().reset_index()
    version_best_scores.columns = ["agent_version", "best_mean"]
    version_best_scores = version_best_scores.sort_values("best_mean", ascending=True)
    ordered_versions = version_best_scores["agent_version"].tolist()

    ordered_keys = []
    improvement_lines = []
    current_position = 0

    for version in ordered_versions:
        version_stats = stats[stats["agent_version"] == version].copy()

        model_best_scores = version_stats.groupby("model")["mean"].max().reset_index()
        model_best_scores.columns = ["model", "best_mean"]
        model_best_scores = model_best_scores.sort_values("best_mean", ascending=True)
        ordered_models = model_best_scores["model"].tolist()

        for model in ordered_models:
            model_stats = version_stats[version_stats["model"] == model].copy()
            model_stats = model_stats.sort_values("mean", ascending=True)

            model_keys = model_stats["agent_loop_key"].tolist()
            ordered_keys.extend(model_keys)

            if len(model_stats) > 1:
                min_score = model_stats["mean"].min()
                max_score = model_stats["mean"].max()
                improvement_pct = (
                    ((max_score - min_score) / min_score * 100) if min_score > 0 else 0
                )

                improvement_lines.append(
                    {
                        "start_pos": current_position,
                        "end_pos": current_position + len(model_keys) - 1,
                        "min_score": min_score,
                        "max_score": max_score,
                        "improvement_pct": improvement_pct,
                        "version": version,
                        "model": model,
                    }
                )

            current_position += len(model_keys)

    return ordered_keys, improvement_lines


def _get_colors_for_scheme(
    stats: pd.DataFrame, color_scheme: str, agent_versions: list[str]
) -> tuple[list, list]:
    """Return ``(bar_colors, legend_items)`` for ``color_scheme``.

    Beyond 10 agent versions, version colors are evenly spaced HSV hues.
    """
    if color_scheme == "version":
        num_versions = len(agent_versions)
        version_colors = {}

        if num_versions <= 10:
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i) for i in range(num_versions)]
        else:
            hues = np.linspace(0, 0.9, num_versions)
            colors = [colorsys.hsv_to_rgb(h, 0.7, 0.85) + (0.8,) for h in hues]

        version_colors = {version: colors[i] for i, version in enumerate(agent_versions)}
        bar_colors = [version_colors[stats.iloc[i]["agent_version"]] for i in range(len(stats))]
        legend_items = [
            Rectangle((0, 0), 1, 1, facecolor=version_colors[version], alpha=0.7, label=version)
            for version in agent_versions
        ]
    elif color_scheme == "model":
        models = list(stats["model"].unique())
        model_colors = get_models_color_palette(models)
        bar_colors = [model_colors[stats.iloc[i]["model"]] for i in range(len(stats))]
        legend_items = [
            Rectangle((0, 0), 1, 1, facecolor=model_colors[model], alpha=0.7, label=model)
            for model in sorted(models)
        ]
    elif color_scheme == "version_model":
        # One base color per version; models within a version are shades of it.
        unique_versions = stats["agent_version"].unique().tolist()
        num_versions = len(unique_versions)

        if num_versions <= 10:
            base_colors = [
                (1.0, 0.5, 0.0),  # Strong Orange
                (0.0, 0.3, 1.0),  # Strong Clear Blue
                (0.5, 0.0, 0.8),  # Strong Purple
                (0.0, 0.7, 0.2),  # Green
                (0.95, 0.1, 0.1),  # Red
                (0.0, 0.7, 0.8),  # Cyan/Teal
                (0.95, 0.75, 0.0),  # Gold
                (0.9, 0.0, 0.5),  # Magenta
                (0.6, 0.35, 0.0),  # Brown
                (0.2, 0.6, 1.0),  # Sky Blue
            ]
        else:
            hues = np.linspace(0, 0.9, num_versions)
            base_colors = [colorsys.hsv_to_rgb(h, 0.8, 0.9) for h in hues]

        version_color_map = {}
        for i, version in enumerate(unique_versions):
            if i < len(base_colors):
                base_color = base_colors[i]
            else:
                base_color = base_colors[i % len(base_colors)]
            version_color_map[version] = base_color

        bar_colors = []
        for i in range(len(stats)):
            version = stats.iloc[i]["agent_version"]
            model = stats.iloc[i]["model"]
            base_color = version_color_map[version]

            version_models = stats[stats["agent_version"] == version]["model"].unique().tolist()
            model_idx = version_models.index(model) if model in version_models else 0
            num_models = len(version_models)

            h, s, v = colorsys.rgb_to_hsv(*base_color)

            if num_models > 1:
                # Bars are sorted by ascending score, so the last model gets the full
                # base color and earlier ones are lighter and less saturated.
                ratio = 1.0 - (model_idx / (num_models - 1))
                s = max(s * (1.0 - ratio * 0.4), 0.4)
                v = min(v + ratio * 0.25, 1.0)

            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            bar_colors.append((r, g, b, 0.85))

        legend_items = []
        for version in unique_versions:
            version_models = stats[stats["agent_version"] == version]["model"].unique().tolist()
            for model_idx, model in enumerate(version_models):
                base_color = version_color_map[version]
                h, s, v = colorsys.rgb_to_hsv(*base_color)
                num_models = len(version_models)

                if num_models > 1:
                    ratio = 1.0 - (model_idx / (num_models - 1))
                    s = max(s * (1.0 - ratio * 0.4), 0.4)
                    v = min(v + ratio * 0.25, 1.0)

                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                legend_items.append(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        facecolor=(r, g, b, 0.85),
                        label=f"{version}/{model}",
                    )
                )
    else:
        num_versions = len(agent_versions)

        if num_versions <= 10:
            cmap = plt.get_cmap("tab10")
            colors = [cmap(i) for i in range(num_versions)]
        else:
            hues = np.linspace(0, 0.9, num_versions)
            colors = [colorsys.hsv_to_rgb(h, 0.7, 0.85) + (0.8,) for h in hues]

        version_colors = {version: colors[i] for i, version in enumerate(agent_versions)}
        bar_colors = [version_colors[stats.iloc[i]["agent_version"]] for i in range(len(stats))]
        legend_items = [
            Rectangle((0, 0), 1, 1, facecolor=version_colors[version], alpha=0.7, label=version)
            for version in agent_versions
        ]

    return bar_colors, legend_items


def _get_legend_title(color_scheme: str) -> str:
    """Get the appropriate legend title for the color scheme."""
    if color_scheme == "version":
        return "Agent Version"
    elif color_scheme == "model":
        return "Model"
    elif color_scheme == "version_model":
        return "Version/Model"
    else:
        return "Agent Version"


def _format_loop_labels(
    stats: pd.DataFrame, clustering: str, is_vertical: bool = False
) -> tuple[list[str], int, dict[str, dict[str, str]]]:
    """Return ``(formatted_labels, fontsize, config_mapping)`` for the bar labels.

    Vertical plots use short ``loop_name / Config<i>`` labels and return a mapping from
    each ``Config<i>`` to its loop_key and agent_version. Horizontal plots use
    ``loop_name / model`` labels, plus the agent version under model clustering.
    """
    formatted_labels = []
    config_mapping = {}

    if is_vertical:
        for i in range(len(stats)):
            row = stats.iloc[i]
            config_name = f"Config{i + 1}"
            formatted_labels.append(f"{row['loop_name']}\n{config_name}")
            config_mapping[config_name] = {
                "loop_key": row["loop_key"],
                "agent_version": row["agent_version"],
            }

        fontsize = 16

    elif clustering == "model":
        for i in range(len(stats)):
            row = stats.iloc[i]
            formatted_labels.append(f"{row['loop_name']}\n{row['model']}\n({row['agent_version']})")
        fontsize = 14  # three-line labels
    else:
        for i in range(len(stats)):
            row = stats.iloc[i]
            formatted_labels.append(f"{row['loop_name']}\n{row['model']}")

        fontsize = 16

    return formatted_labels, fontsize, config_mapping


def compare_loops_and_agent_versions(
    summary_df: DataSet[AgentScoreSummarySchema],
    per_run_df: DataSet[AgentScorePerRunSchema],
    agents: list[str] | None = None,
    agent_versions: list[str] | None = None,
    clustering: Literal["none", "version", "model", "version_model", "best"] = "version",
    orientation: Literal["horizontal", "vertical", "best"] = "best",
    use_ci_boxplots: bool = True,
    show_means: bool = False,
    show_points: bool = False,
    show_whiskers: bool = True,
    ci_confidence_level: float = 0.95,
):
    """Plot, for each agent, every (agent_version, loop) combination in one figure.

    The left panel shows the mean score labeled with its error margin; the right panel
    shows the per-run scores as box plots.

    Args:
        agents: Agents to plot (default: all).
        agent_versions: Agent versions to plot (default: all).
        clustering: How bars are ordered and colored:
            - "none": by score only
            - "version": by agent version, with an improvement arrow per version
            - "model": by model
            - "version_model": by agent version, then model
            - "best": chosen from the number of versions and models
        orientation: "horizontal", "vertical", or "best" (vertical for at most 6 bars).
        use_ci_boxplots: Draw boxes as bootstrap CIs instead of IQRs.
        ci_confidence_level: Confidence level the summary CIs were computed at (axis labels).
    """
    if not agents:
        agents = list(summary_df["agent"].unique())
        if not agents:
            raise ValueError("No agents found")
    if not agent_versions:
        agent_versions = list(summary_df["agent_version"].unique())
        if not agent_versions:
            raise ValueError("No agent versions found")

    for agent in agents:
        _plot_loops_across_versions(
            summary_df,
            per_run_df,
            agent,
            agent_versions,
            clustering,
            orientation,
            use_ci_boxplots,
            show_means,
            show_points,
            show_whiskers,
            ci_confidence_level,
        )
        display(HTML("<br>" * 10))


def _plot_loops_across_versions(
    summary_df: DataSet[AgentScoreSummarySchema],
    per_run_df: DataSet[AgentScorePerRunSchema],
    agent: str,
    agent_versions: list[str],
    clustering: Literal["none", "version", "model", "version_model", "best"],
    orientation: Literal["horizontal", "vertical", "best"],
    use_ci_boxplots: bool = True,
    show_means: bool = False,
    show_points: bool = False,
    show_whiskers: bool = True,
    ci_confidence_level: float = 0.95,
) -> None:
    """Draw the figure for one agent (see ``compare_loops_and_agent_versions``)."""
    summary_filtered = summary_df[
        (summary_df["agent"] == agent) & (summary_df["agent_version"].isin(agent_versions))
    ].copy()

    if summary_filtered.empty:
        print(f"No data found for agent {agent} with versions {agent_versions}")
        return

    summary_filtered["mean"] = summary_filtered["score"].str.split(" ± ").str[0].astype(float)
    summary_filtered["error_margin"] = (
        summary_filtered["score"].str.split(" ± ").str[1].astype(float)
    )

    # Symmetric bounds from the error margin.
    summary_filtered["ci_lower"] = summary_filtered["mean"] - summary_filtered["error_margin"]
    summary_filtered["ci_upper"] = summary_filtered["mean"] + summary_filtered["error_margin"]

    summary_filtered["agent_loop_key"] = (
        summary_filtered["agent_version"] + "::" + summary_filtered["loop_key"]
    )

    per_run_filtered = per_run_df[
        (per_run_df["agent"] == agent) & (per_run_df["agent_version"].isin(agent_versions))
    ].copy()
    per_run_filtered["agent_loop_key"] = (
        per_run_filtered["agent_version"] + "::" + per_run_filtered["loop_key"]
    )

    minmax = per_run_filtered.groupby("agent_loop_key")["score"].agg(["min", "max"]).reset_index()

    stats = summary_filtered.merge(minmax, on="agent_loop_key", how="left")

    if clustering == "best":
        clustering = _determine_best_clustering(stats, agent_versions)
        print(f"Auto-selected clustering mode: {clustering}")

    if clustering == "none":
        stats = stats.sort_values("mean", ascending=False)
        ordered_keys = stats["agent_loop_key"].tolist()
        improvement_lines = []
        color_scheme = "version"
    elif clustering == "version":
        ordered_keys, improvement_lines = _cluster_by_version(stats, agent_versions)
        color_scheme = "version"
    elif clustering == "model":
        ordered_keys, improvement_lines = _cluster_by_model(stats)
        color_scheme = "model"
    elif clustering == "version_model":
        ordered_keys, improvement_lines = _cluster_by_version_and_model(stats, agent_versions)
        color_scheme = "version_model"
    else:
        raise ValueError(f"Unknown clustering mode: {clustering}")

    stats["order"] = stats["agent_loop_key"].apply(lambda x: ordered_keys.index(x))
    stats = stats.sort_values("order")

    sorted_keys = stats["agent_loop_key"].tolist()
    sorted_loops = stats["loop_key"].tolist()
    sorted_means = stats["mean"].tolist()
    sorted_error_margins = stats["error_margin"].tolist()

    if orientation == "best":
        is_vertical = len(sorted_loops) <= 6
    else:
        is_vertical = orientation == "vertical"

    formatted_labels, label_fontsize, config_mapping = _format_loop_labels(
        stats, clustering, is_vertical
    )

    if is_vertical:
        figsize = (max(32, len(sorted_loops) * 4.5), 16)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1, 1]},
            sharex=True,
        )
    else:
        figsize = (28, max(16, len(sorted_loops) * 0.7))
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1, 1]},
            sharey=True,
        )

    bar_colors, legend_items = _get_colors_for_scheme(stats, color_scheme, agent_versions)

    if is_vertical:
        bars = axes[0].bar(
            range(len(sorted_loops)),
            sorted_means,
            color=bar_colors,
            alpha=0.7,
            capsize=7,
            width=0.7,
        )
        for i, bar in enumerate(bars):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{sorted_means[i]:.3f}\n± {sorted_error_margins[i]:.3f}",
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="normal",
            )
    else:
        bars = axes[0].barh(
            range(len(sorted_loops)),
            sorted_means,
            color=bar_colors,
            alpha=0.7,
            capsize=5,
            height=0.6,
        )
        for i, bar in enumerate(bars):
            axes[0].text(
                bar.get_width() + 0.02,
                bar.get_y() + bar.get_height() / 2,
                f"{sorted_means[i]:.3f} ± {sorted_error_margins[i]:.3f}",
                va="center",
                fontsize=16,
                fontweight="normal",
            )

    if improvement_lines:
        for line_info in improvement_lines:
            pos_start = line_info["start_pos"]
            pos_end = line_info["end_pos"]
            score_min = line_info["min_score"]
            score_max = line_info["max_score"]
            improvement_pct = line_info["improvement_pct"]

            arrow_color = "green" if improvement_pct >= 0 else "red"

            if is_vertical:
                axes[0].annotate(
                    "",
                    xy=(pos_end, score_max),
                    xytext=(pos_start, score_min),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=arrow_color,
                        lw=3.5,
                        alpha=0.7,
                        linestyle="--",
                    ),
                )
                axes[0].plot(
                    pos_start, score_min, "o", color=arrow_color, markersize=12, alpha=0.8, zorder=5
                )
                axes[0].plot(
                    pos_end, score_max, "o", color=arrow_color, markersize=12, alpha=0.8, zorder=5
                )

                mid_pos = (pos_start + pos_end) / 2
                mid_score = (score_min + score_max) / 2

                # Move the label off the line when its midpoint is close to a bar.
                dist_to_start = abs(mid_pos - pos_start)
                dist_to_end = abs(mid_pos - pos_end)

                if min(dist_to_start, dist_to_end) < 0.4:
                    text_pos = mid_pos + 0.3
                    ha_align = "left"
                else:
                    text_pos = mid_pos
                    ha_align = "center"

                sign = "+" if improvement_pct > 0 else ""
                axes[0].text(
                    text_pos,
                    mid_score,
                    f"{sign}{improvement_pct:.1f}%",
                    ha=ha_align,
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color=arrow_color,
                    bbox=dict(
                        boxstyle="round,pad=0.5",
                        facecolor="white",
                        alpha=0.95,
                        edgecolor=arrow_color,
                    ),
                )
            else:
                axes[0].annotate(
                    "",
                    xy=(score_max, pos_end),
                    xytext=(score_min, pos_start),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=arrow_color,
                        lw=2.5,
                        alpha=0.7,
                        linestyle="--",
                    ),
                )
                axes[0].plot(
                    score_min, pos_start, "o", color=arrow_color, markersize=8, alpha=0.8, zorder=5
                )
                axes[0].plot(
                    score_max, pos_end, "o", color=arrow_color, markersize=8, alpha=0.8, zorder=5
                )

                mid_score = (score_min + score_max) / 2
                mid_pos = (pos_start + pos_end) / 2
                dist_to_start = abs(mid_pos - pos_start)
                dist_to_end = abs(mid_pos - pos_end)

                if min(dist_to_start, dist_to_end) < 0.3:
                    text_pos = mid_pos + 0.5
                    va_align = "bottom"
                else:
                    text_pos = mid_pos
                    va_align = "center"

                sign = "+" if improvement_pct > 0 else ""
                axes[0].text(
                    mid_score,
                    text_pos,
                    f"{sign}{improvement_pct:.1f}%",
                    ha="center",
                    va=va_align,
                    fontsize=14,
                    fontweight="bold",
                    color=arrow_color,
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor="white",
                        alpha=0.95,
                        edgecolor=arrow_color,
                    ),
                )

    if is_vertical:
        axes[0].set_ylabel(f"Score Mean + {ci_confidence_level * 100:g}% CI", fontsize=26)
        axes[0].set_title("Average Performance across runs", fontsize=28)
        axes[0].grid(axis="y", linestyle="--", alpha=0.7)
        axes[0].set_ylim(0, 1.09)  # margins for text
        axes[0].set_xticks(
            range(len(sorted_loops)),
            labels=formatted_labels,
            fontsize=label_fontsize + 4,
            rotation=45,
            ha="right",
        )
        axes[0].tick_params(axis="y", labelsize=24)
    else:
        axes[0].set_xlabel(f"Score Mean + {ci_confidence_level * 100:g}% CI", fontsize=20)
        axes[0].set_ylabel("Loops", fontsize=20)
        axes[0].set_title("Average Performance across runs", fontsize=22)
        axes[0].grid(axis="x", linestyle="--", alpha=0.7)
        axes[0].set_xlim(0, 1.09)  # margins for text
        axes[0].set_yticks(
            range(len(sorted_loops)), labels=formatted_labels, fontsize=label_fontsize
        )
        axes[0].tick_params(axis="x", labelsize=18)

    box_data = []
    for key in sorted_keys:
        key_data = per_run_filtered[per_run_filtered["agent_loop_key"] == key]["score"].values
        box_data.append(key_data)

    if is_vertical:
        box = axes[1].boxplot(
            box_data,
            vert=True,
            patch_artist=True,
            widths=0.7,
            positions=range(len(sorted_loops)),
            showmeans=show_means,
            showfliers=False,
            whis=(0 if not show_whiskers else 1.5),
        )
    else:
        box = axes[1].boxplot(
            box_data,
            vert=False,
            patch_artist=True,
            widths=0.6,
            positions=range(len(sorted_loops)),
            showmeans=show_means,
            showfliers=False,
            whis=(0 if not show_whiskers else 1.5),
        )

    for patch, color in zip(box["boxes"], bar_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    if use_ci_boxplots:
        ci_data: dict[str, BootstrapStatistics] = {}
        for _, row in stats.iterrows():
            key = row["agent_loop_key"]
            ci_data[key] = BootstrapStatistics(
                mean=row["mean"],
                cilo=row["cilo"],
                cihi=row["cihi"],
                whislo=row["whislo"],
                whishi=row["whishi"],
            )

        boxes = box["boxes"]
        all_lines = axes[1].get_lines()

        _apply_bootstrap_ci_to_boxplot(
            axes[1],
            boxes,
            all_lines,
            ci_data,
            sorted_keys,
            is_vertical=is_vertical,
            show_whiskers=show_whiskers,
        )

    if show_points:
        for i, key_data in enumerate(box_data):
            jitter_amount = 0.04
            num_points = len(key_data)
            jitter = np.random.uniform(-jitter_amount, jitter_amount, num_points)

            if is_vertical:
                x_positions = np.full(num_points, i) + jitter
                axes[1].scatter(
                    x_positions,
                    key_data,
                    color="black",
                    alpha=0.5,
                    s=30,
                    zorder=3,
                )
            else:
                y_positions = np.full(num_points, i) + jitter
                axes[1].scatter(
                    key_data,
                    y_positions,
                    color="black",
                    alpha=0.5,
                    s=30,
                    zorder=3,
                )

    if is_vertical:
        axes[1].set_ylabel(f"Score Mean + {ci_confidence_level * 100:g}% CI", fontsize=26)
        axes[1].set_title("Score Distribution", fontsize=28)
        axes[1].grid(axis="y", linestyle="--", alpha=0.7)
        axes[1].set_ylim(0, 1.09)  # margins for text
        axes[1].set_xticks(
            range(len(sorted_loops)),
            labels=formatted_labels,
            fontsize=label_fontsize + 4,
            rotation=45,
            ha="right",
        )
        axes[1].tick_params(axis="y", labelsize=24)
    else:
        axes[1].set_xlabel(f"Score Mean + {ci_confidence_level * 100:g}% CI", fontsize=20)
        axes[1].set_ylabel("Loops", fontsize=20)
        axes[1].set_title("Score Distribution", fontsize=22)
        axes[1].grid(axis="x", linestyle="--", alpha=0.7)
        axes[1].set_xlim(0, 1.09)  # margins for text
        axes[1].set_yticks(
            range(len(sorted_loops)), labels=formatted_labels, fontsize=label_fontsize
        )
        axes[1].tick_params(axis="x", labelsize=18)

    agent_escaped = agent.replace("_", r"\_")
    if is_vertical:
        fig.suptitle(
            f"Loop Performance Comparison for $\\bf{{{agent_escaped}}}$",
            fontsize=30,
            y=0.97,
        )
    else:
        fig.suptitle(
            f"Loop Performance Comparison for $\\bf{{{agent_escaped}}}$",
            fontsize=24,
            y=0.99,
        )

    if is_vertical and config_mapping:
        mapping_lines = []
        for config_name in sorted(
            config_mapping.keys(), key=lambda x: int(x.replace("Config", ""))
        ):
            details = config_mapping[config_name]
            mapping_lines.append(
                f"{config_name}: Loop: {details['loop_key']}, "
                f"AgentVersion: {details['agent_version']}"
            )

        mapping_text = "\n".join(mapping_lines)
        # About 0.012 figure height per line.
        num_lines = len(mapping_lines)
        config_y_position = -0.005 - (num_lines * 0.012)

        text_obj = fig.text(
            0.05,
            config_y_position,
            mapping_text,
            ha="left",
            va="bottom",
            fontsize=18,
            bbox=dict(
                boxstyle="round,pad=1.2",
                facecolor="lightyellow",
                alpha=0.9,
                edgecolor="gray",
                linewidth=2,
            ),
            transform=fig.transFigure,
        )

        # Render once to measure the text box, then put a padded white rectangle behind it.
        fig.canvas.draw()
        renderer = fig.canvas.renderer  # pyright: ignore[reportAttributeAccessIssue]
        bbox = text_obj.get_window_extent(renderer=renderer)
        bbox_fig = bbox.transformed(fig.transFigure.inverted())

        padding = 0.02
        border_rect = Rectangle(
            (bbox_fig.x0 - padding, bbox_fig.y0 - padding),
            width=bbox_fig.width + 2 * padding,
            height=bbox_fig.height + 2 * padding,
            transform=fig.transFigure,
            facecolor="white",
            edgecolor="none",
            alpha=1.0,
            zorder=text_obj.get_zorder() - 1,
        )
        fig.patches.append(border_rect)

    if legend_items:
        if is_vertical:
            num_items = len(legend_items)
            ncol = 1

            axes[0].legend(
                handles=legend_items,
                loc="lower right",
                fontsize=16,
                title=_get_legend_title(color_scheme),
                title_fontsize=20,
                ncol=ncol,
                frameon=True,
                fancybox=True,
                shadow=True,
            )
        else:
            num_items = len(legend_items)
            if num_items % 2 == 0:
                ncol = 4
            elif num_items % 3 == 0:
                ncol = 3
            else:
                ncol = 5

            fig.legend(
                handles=legend_items,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.05),
                fontsize=16,
                title=_get_legend_title(color_scheme),
                title_fontsize=20,
                ncol=ncol,
                frameon=True,
                fancybox=True,
                shadow=True,
            )

    fig.tight_layout()
    if is_vertical:
        top_space = 0.90
        if config_mapping:
            num_configs = len(config_mapping)
            bottom_space = 0.15 + (num_configs * 0.02)
        else:
            bottom_space = 0.05
        fig.subplots_adjust(wspace=0.2, top=top_space, left=0.05, bottom=bottom_space)
    else:
        bottom_space = 0.10 if legend_items else 0.05
        fig.subplots_adjust(wspace=0.2, top=0.93, bottom=bottom_space)

    plt.show()


def sort_testcases_by_loop_difference(
    df: pd.DataFrame,
    loop1_name: str,
    loop2_name: str,
    loop1_version: str | None = None,
    loop2_version: str | None = None,
    score_threshold: float = 0.5,
    show_details: bool = True,
) -> pd.DataFrame:
    """Rank datapoints by the score difference between two loops, per agent and run.

    Also prints how many differences exceed ``score_threshold`` either way, the top 10
    cases each way and a per-agent summary (as tables when ``show_details``).

    Args:
        df: Evaluator-level results (one row per datapoint, run and evaluator).
        loop1_name: Loop key of the first loop, or its loop name if ``df`` has no
            ``loop_key`` column.
        loop2_name: Same for the second loop.
        loop1_version: If set, keep only rows of the first loop with this ``loop_version``.
        loop2_version: Same for the second loop.

    Returns:
        Columns agent, agent_version, run, datapoint_name, ``<loop1_name>_score``,
        ``<loop2_name>_score`` and score_difference (loop1 - loop2), sorted by
        score_difference in descending order.
    """

    if "run" not in df.columns:
        df = df.copy()
        df["run"] = df["experiment_unique_identifier"].str.split("_").str[-1].str.split(":").str[-1]

    # Use 'loop_key' column if it exists (full identifier), otherwise fall back to 'loop_name'
    loop_col = "loop_key" if "loop_key" in df.columns else "loop_name"
    loop1_filter = df[loop_col] == loop1_name
    if loop1_version:
        loop1_filter = loop1_filter & (df["loop_version"] == loop1_version)

    loop2_filter = df[loop_col] == loop2_name
    if loop2_version:
        loop2_filter = loop2_filter & (df["loop_version"] == loop2_version)

    df_loop1 = df[loop1_filter].copy()
    df_loop2 = df[loop2_filter].copy()

    if df_loop1.empty or df_loop2.empty:
        print("Warning: No data found for one or both loops.")
        print(f"Loop1 ({loop1_name}): {len(df_loop1)} records")
        print(f"Loop2 ({loop2_name}): {len(df_loop2)} records")
        return pd.DataFrame()

    # Weighted like calculate_agent_scores, but kept at datapoint level; a datapoint whose
    # evaluators all weigh 0 falls back to their unweighted mean.

    def calculate_datapoint_scores(df_loop: pd.DataFrame) -> pd.DataFrame:
        """Calculate aggregate scores per datapoint for a given loop."""
        datapoint_scores = []

        for (agent, agent_version, run, datapoint_name), group in df_loop.groupby(
            ["agent", "agent_version", "run", "datapoint_name"]
        ):
            # Weights are assumed constant within the group.
            evaluator_weights = group.iloc[0]["evaluator_weights"]

            # Without weights, average all evaluator scores.
            if not evaluator_weights or not isinstance(evaluator_weights, dict):
                final_score = group["score"].mean()
            else:
                total_weighted_score = 0
                total_weight = 0

                for _, row in group.iterrows():
                    evaluator_id = row["evaluator_id"]
                    weight = evaluator_weights.get(evaluator_id, 1)
                    if weight > 0:
                        total_weighted_score += row["score"] * weight
                        total_weight += weight

                if total_weight > 0:
                    final_score = total_weighted_score / total_weight
                else:
                    final_score = group["score"].mean()

            datapoint_scores.append(
                {
                    "agent": agent,
                    "agent_version": agent_version,
                    "run": run,
                    "datapoint_name": datapoint_name,
                    "score": final_score,
                }
            )

        return pd.DataFrame(datapoint_scores)

    loop1_datapoint_scores = calculate_datapoint_scores(df_loop1)
    loop2_datapoint_scores = calculate_datapoint_scores(df_loop2)

    merged_scores = pd.merge(
        loop1_datapoint_scores,
        loop2_datapoint_scores,
        on=["agent", "agent_version", "run", "datapoint_name"],
        suffixes=("_loop1", "_loop2"),
        how="outer",
    )

    # A datapoint missing from one loop scores 0 there.
    merged_scores["score_loop1"] = merged_scores["score_loop1"].fillna(0)
    merged_scores["score_loop2"] = merged_scores["score_loop2"].fillna(0)

    merged_scores["score_difference"] = merged_scores["score_loop1"] - merged_scores["score_loop2"]

    merged_scores = merged_scores.rename(
        columns={"score_loop1": f"{loop1_name}_score", "score_loop2": f"{loop2_name}_score"}
    )

    result = merged_scores.sort_values("score_difference", ascending=False).reset_index(drop=True)

    print(f"\n=== Test Case Score Comparison: {loop1_name} vs {loop2_name} ===")
    print(f"Total test cases compared: {len(result)}")
    print(
        f"Cases where {loop1_name} performs better (diff > {score_threshold}): {len(result[result['score_difference'] > score_threshold])}"
    )
    print(
        f"Cases where {loop2_name} performs better (diff < -{score_threshold}): {len(result[result['score_difference'] < -score_threshold])}"
    )
    print(
        f"Cases with similar performance (|diff| <= {score_threshold}): {len(result[abs(result['score_difference']) <= score_threshold])}"
    )

    print(f"\n=== Top 10 cases where {loop1_name} performs much better ===")
    top_loop1 = result.head(10)
    if show_details:
        display_df(top_loop1)
    else:
        print(
            top_loop1[
                [
                    "agent",
                    "datapoint_name",
                    f"{loop1_name}_score",
                    f"{loop2_name}_score",
                    "score_difference",
                ]
            ].to_string(index=False)
        )

    print(f"\n=== Top 10 cases where {loop2_name} performs much better ===")
    top_loop2 = result.tail(10)
    if show_details:
        display_df(top_loop2)
    else:
        print(
            top_loop2[
                [
                    "agent",
                    "datapoint_name",
                    f"{loop1_name}_score",
                    f"{loop2_name}_score",
                    "score_difference",
                ]
            ].to_string(index=False)
        )

    print("\n=== Summary by Agent ===")
    agent_summary = (
        result.groupby("agent")
        .agg(
            {
                "score_difference": ["mean", "std", "count"],
                f"{loop1_name}_score": "mean",
                f"{loop2_name}_score": "mean",
            }
        )
        .round(3)
    )

    agent_summary.columns = [
        "avg_score_diff",
        "std_score_diff",
        "num_testcases",
        f"avg_{loop1_name}_score",
        f"avg_{loop2_name}_score",
    ]
    agent_summary = agent_summary.sort_values("avg_score_diff", ascending=False)

    if show_details:
        display_df(agent_summary)
    else:
        print(agent_summary.to_string())

    return result


def loops_testcase_comparison(df: DataSet[EvaluatorDatapointResultSchema] | None) -> None:
    """Run ``sort_testcases_by_loop_difference`` on the two loops with the highest mean score."""
    if df is None or df.empty:
        print("No data to compare")
        return

    df_copy = pd.DataFrame(df).copy()

    if "loop_key" not in df_copy.columns and "loop_version" in df_copy.columns:
        df_copy = df_copy.rename(columns={"loop_version": "loop_key"})

    available_loops = df_copy["loop_key"].unique()
    print(f"Available loops: {available_loops}")

    if len(available_loops) >= 2:
        if len(available_loops) == 2:
            first_loop = available_loops[0]
            second_loop = available_loops[1]
        else:
            loop_performance = {}
            for loop in available_loops:
                loop_data = df_copy[df_copy["loop_key"] == loop]
                avg_score = loop_data["score"].mean()
                loop_performance[loop] = avg_score

            top_2 = heapq.nlargest(2, loop_performance.items(), key=lambda x: x[1])
            first_loop = top_2[0][0]
            second_loop = top_2[1][0]

            print(
                f"\nSelected top 2 performing loops: {first_loop} ({top_2[0][1]:.4f}) vs {second_loop} ({top_2[1][1]:.4f})"
            )

        print(f"\nComparing {first_loop} vs {second_loop}")

        testcase_comparison = sort_testcases_by_loop_difference(
            df=df_copy,
            loop1_name=first_loop,
            loop2_name=second_loop,
            score_threshold=0.3,
            show_details=True,
        )

        print(f"\n=== Cases where {first_loop} performs perfectly but {second_loop} fails ===")
        perfect_vs_fail = testcase_comparison[
            (testcase_comparison[f"{first_loop}_score"] >= 0.9)
            & (testcase_comparison[f"{second_loop}_score"] <= 0.1)
        ]

        if not perfect_vs_fail.empty:
            print(f"Found {len(perfect_vs_fail)} such cases:")
            display_df(
                perfect_vs_fail[
                    [
                        "agent",
                        "agent_version",
                        "run",
                        "datapoint_name",
                        f"{first_loop}_score",
                        f"{second_loop}_score",
                        "score_difference",
                    ]
                ].head(20)
            )
        else:
            print(
                "No cases found where one loop performs perfectly and the other fails completely."
            )

        print(f"\n=== Cases where {second_loop} performs perfectly but {first_loop} fails ===")
        fail_vs_perfect = testcase_comparison[
            (testcase_comparison[f"{first_loop}_score"] <= 0.1)
            & (testcase_comparison[f"{second_loop}_score"] >= 0.9)
        ]

        if not fail_vs_perfect.empty:
            print(f"Found {len(fail_vs_perfect)} such cases:")
            display_df(
                fail_vs_perfect[
                    [
                        "agent",
                        "agent_version",
                        "run",
                        "datapoint_name",
                        f"{first_loop}_score",
                        f"{second_loop}_score",
                        "score_difference",
                    ]
                ].head(20)
            )
        else:
            print(
                "No cases found where one loop fails completely and the other performs perfectly."
            )

        print("\n=== Cases with largest score differences (top 10 each way) ===")
        print(f"Top 10 cases where {first_loop} performs much better:")
        top_positive = testcase_comparison.head(10)
        display_df(
            top_positive[
                [
                    "agent",
                    "agent_version",
                    "run",
                    "datapoint_name",
                    f"{first_loop}_score",
                    f"{second_loop}_score",
                    "score_difference",
                ]
            ]
        )

        print(f"\nTop 10 cases where {second_loop} performs much better:")
        top_negative = testcase_comparison.tail(10)
        display_df(
            top_negative[
                [
                    "agent",
                    "agent_version",
                    "run",
                    "datapoint_name",
                    f"{first_loop}_score",
                    f"{second_loop}_score",
                    "score_difference",
                ]
            ]
        )
    else:
        print("Need at least 2 loops to perform comparison analysis.")
