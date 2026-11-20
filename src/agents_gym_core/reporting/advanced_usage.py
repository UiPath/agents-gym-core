from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from strictly_typed_pandas.dataset import DataSet

from agents_gym_core.reporting.schemas import AgentScorePerRunSchema
from agents_gym_core.reporting.utils import (
    extract_optimization_algorithm,
    get_models_color_palette,
)


def run_correlation_analysis(per_run_df: DataSet[AgentScorePerRunSchema], variance: float = 0.05):
    """Scatter-plot mean agent scores for every pair of loops.

    Args:
        per_run_df: Per-run scores.
        variance: Absolute mean-score tolerance used for the diagonal band and counts.
    """
    groupby_cols = ["loop_name", "loop_version", "loop_key", "agent", "agent_version"]
    if "model" in per_run_df.columns:
        groupby_cols.append("model")

    df_runs = pd.DataFrame(per_run_df.groupby(groupby_cols, as_index=False)["score"].mean())
    df_runs.rename(columns={"score": "mean_score"}, inplace=True)

    df_runs_copy = df_runs.copy()
    df_runs_copy["optimization_algorithm"] = df_runs_copy["loop_key"].apply(
        extract_optimization_algorithm
    )

    loops = df_runs_copy["loop_key"].unique()
    if len(loops) < 2:
        print("Not enough loops to perform correlation analysis")
        return
    loop_pairs = list(combinations(loops, 2))

    for loop1, loop2 in loop_pairs:
        agent_executions = df_runs_copy[df_runs_copy["loop_key"].isin([loop1, loop2])].copy()
        if len(agent_executions) == 0:
            print(f"No executions for {loop1} or {loop2}")
            continue

        loop1_data = df_runs_copy[df_runs_copy["loop_key"] == loop1]
        loop2_data = df_runs_copy[df_runs_copy["loop_key"] == loop2]

        loop1_model_name = loop1_data["model"].iloc[0]
        loop2_model_name = loop2_data["model"].iloc[0]
        loop1_optim_algo = extract_optimization_algorithm(loop1)
        loop2_optim_algo = extract_optimization_algorithm(loop2)
        loop1_name = loop1_data["loop_name"].iloc[0]
        loop2_name = loop2_data["loop_name"].iloc[0]
        loop1_version = loop1_data["loop_version"].iloc[0]
        loop2_version = loop2_data["loop_version"].iloc[0]

        if loop1.removeprefix(loop1_optim_algo) == loop2.removeprefix(loop2_optim_algo):
            group_type = "optimization_algorithm"
            title = f"Prompt Optimization Correlation Analysis for {loop1_name}: {loop1_optim_algo} vs {loop2_optim_algo}"
            show_legend = False
        elif loop1_name == loop2_name and loop1_model_name == loop2_model_name:
            group_type = "version"
            title = f"Version Correlation Analysis for {loop1_name} ({loop1_model_name}): {loop1_version} vs {loop2_version}"
            show_legend = False
        elif loop1_name == loop2_name and loop1_version == loop2_version:
            group_type = "model"
            title = f"Model Correlation Analysis for {loop1_name} {loop1_version}: {loop1_model_name} vs {loop2_model_name}"
            show_legend = False
        elif loop1_name == loop2_name:
            # Version and model both differ.
            group_type = "model"
            title = f"Loop Correlation Analysis for {loop1_name}: {loop1_version}/{loop1_model_name} vs {loop2_version}/{loop2_model_name}"
            show_legend = False
        else:
            group_type = "loop_key"
            title = "Correlation Analysis: Loop1 vs Loop2"
            show_legend = True

        _, ax = plt.subplots(figsize=(12, 10))
        ax.set_title(
            title,
            fontsize=14,
            fontweight="bold",
        )

        loop1_data = agent_executions[agent_executions["loop_key"] == loop1]
        loop2_data = agent_executions[agent_executions["loop_key"] == loop2]

        loop1_subset = loop1_data[["agent", "agent_version", "mean_score"]].copy()
        loop1_subset = loop1_subset.rename(columns={"mean_score": loop1})
        loop2_subset = loop2_data[["agent", "agent_version", "mean_score"]].copy()
        loop2_subset = loop2_subset.rename(columns={"mean_score": loop2})
        merged_data = pd.merge(
            loop1_subset,
            loop2_subset,
            on=["agent", "agent_version"],
            how="inner",
        )

        if len(merged_data) == 0:
            print(f"No common agents between {loop1} and {loop2}")
            continue

        unique_versions = merged_data["agent_version"].unique()
        colors = plt.get_cmap("Set3")(np.linspace(0, 1, len(unique_versions)))
        version_color_map = dict(zip(unique_versions, colors))
        models_colors = (
            get_models_color_palette(list(df_runs_copy["model"].unique()))
            if "model" in df_runs_copy.columns
            else {}
        )

        for version in unique_versions:
            version_data = merged_data[merged_data["agent_version"] == version]
            ax.scatter(
                version_data[loop1],
                version_data[loop2],
                s=120,
                alpha=0.7,
                color=version_color_map[version],
                edgecolors="black",
                linewidths=1.5,
                label=f"Agent Version: {version}",
            )

        for _, row in merged_data.iterrows():
            ax.annotate(
                row["agent"],
                (row[loop1], row[loop2]),  # type: ignore
                fontsize=8,
                ha="right",
                va="bottom",
                alpha=0.8,
            )

        ax.plot(
            [0, 1.05],
            [0, 1.05],
            "r--",
            linewidth=2,
            alpha=0.5,
            label="Equal mean scores",
        )

        ax.plot(
            [0, 1.05],
            [0 + variance, 1.05 + variance],
            "r:",
            linewidth=1,
            alpha=0.3,
            label=f"Score difference ±{variance}",
        )
        ax.plot(
            [0, 1.05],
            [0 - variance, 1.05 - variance],
            "r:",
            linewidth=1,
            alpha=0.3,
        )

        ax.grid(True, linestyle="--", alpha=0.3)

        if group_type == "loop_key":
            ax.set_xlabel("Loop1 Score", fontsize=12, fontweight="bold")
            ax.set_ylabel("Loop2 Score", fontsize=12, fontweight="bold")
        elif group_type == "optimization_algorithm":
            ax.set_xlabel(f"{loop1_optim_algo} Score", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"{loop2_optim_algo} Score", fontsize=12, fontweight="bold")
        elif group_type == "version":
            ax.set_xlabel(f"{loop1_version} Score", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"{loop2_version} Score", fontsize=12, fontweight="bold")
        elif group_type == "model":
            ax.set_xlabel(f"{loop1_model_name} Score", fontsize=12, fontweight="bold")
            ax.set_ylabel(f"{loop2_model_name} Score", fontsize=12, fontweight="bold")

        ax.legend(loc="upper left", fontsize=10, bbox_to_anchor=(0, 1))

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

        if group_type == "loop_key":
            loop1_label = "Loop1"
            loop2_label = "Loop2"
        elif group_type == "optimization_algorithm":
            loop1_label = loop1_optim_algo
            loop2_label = loop2_optim_algo
        elif group_type == "version":
            loop1_label = loop1_version
            loop2_label = loop2_version
        elif group_type == "model":
            loop1_label = loop1_model_name
            loop2_label = loop2_model_name
        else:
            loop1_label = "Loop1"
            loop2_label = "Loop2"

        # Label the triangles above (y-axis loop better) and below the diagonal.
        top_left_x = 0.25
        top_left_y = 0.75
        ax.text(
            top_left_x,
            top_left_y,
            f"{loop2_label} better",
            fontsize=16,
            ha="center",
            va="center",
            color=models_colors[loop2_label] if loop2_label in models_colors else "green",
            fontweight="bold",
            alpha=0.3,
            rotation=45,
        )

        bottom_right_x = 0.75
        bottom_right_y = 0.25
        ax.text(
            bottom_right_x,
            bottom_right_y,
            f"{loop1_label} better",
            fontsize=16,
            ha="center",
            va="center",
            color=models_colors[loop1_label] if loop1_label in models_colors else "blue",
            fontweight="bold",
            alpha=0.3,
            rotation=45,
        )

        loop1_better = (merged_data[loop1] > merged_data[loop2] + variance).sum()
        loop2_better = (merged_data[loop2] > merged_data[loop1] + variance).sum()
        equal = (abs(merged_data[loop1] - merged_data[loop2]) <= variance).sum()

        stats_text = f"{loop1_label} higher by >{variance}: {loop1_better}\n{loop2_label} higher by >{variance}: {loop2_better}\nWithin ±{variance}: {equal}"
        ax.text(
            0.98,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            ha="right",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        )

        if show_legend:
            legend_text = f"Loop1: {loop1}\nLoop2: {loop2}"
            plt.figtext(
                0.5,
                -0.05,
                legend_text,
                ha="center",
                va="bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            )

        plt.tight_layout()
        plt.show()
