from itertools import combinations

import matplotlib.axes
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, display
from matplotlib import pyplot as plt
from matplotlib.transforms import Bbox
from scipy.stats import f_oneway, ttest_rel
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from agents_gym_core.reporting.utils import (
    display_df,
    extract_optimization_algorithm,
    get_models_color_palette,
)
from agents_gym_core.utils.bootstrap import (
    BootstrapStatistics,
    bootstrap_confidence_interval,
    hierarchical_bootstrap_confidence_interval,
)


def t_test_between_loops(
    df_runs: pd.DataFrame, loop1: str, loop2: str, p_value_threshold: float = 0.05
) -> None:
    """Paired t-test between two loop keys, plotted as a violin of per-run score differences.

    Runs are paired by (agent, agent_version, run); a run missing from one loop scores 0.
    """
    index_cols = ["agent", "agent_version", "run"]
    df_pivot = (
        df_runs[df_runs["loop_key"].isin([loop1, loop2])]
        .pivot_table(index=index_cols, columns=["loop_key"], values="score", aggfunc="mean")
        .fillna(0)
    )
    score_diff = df_pivot[loop1] - df_pivot[loop2]

    # Zero variance would make scipy emit precision warnings; handle it directly.
    if score_diff.std() == 0:
        t_value, p_value = (
            (float("inf") if score_diff.mean() != 0 else 0.0),
            (0.0 if score_diff.mean() != 0 else 1.0),
        )
    else:
        t_value, p_value = ttest_rel(
            df_pivot[loop1],
            df_pivot[loop2],
        )

    loop1_optim_algo = extract_optimization_algorithm(loop1)
    loop2_optim_algo = extract_optimization_algorithm(loop2)

    loop1_data = df_runs[df_runs["loop_key"] == loop1]
    loop2_data = df_runs[df_runs["loop_key"] == loop2]

    loop1_model_name = loop1_data["model"].iloc[0]
    loop2_model_name = loop2_data["model"].iloc[0]
    loop1_name = loop1_data["loop_name"].iloc[0]
    loop2_name = loop2_data["loop_name"].iloc[0]
    loop1_version = loop1_data["loop_version"].iloc[0]
    loop2_version = loop2_data["loop_version"].iloc[0]

    if loop1.removeprefix(loop1_optim_algo) == loop2.removeprefix(loop2_optim_algo):
        title = (
            f"Prompt Optimization T-Test for {loop1_name}: {loop1_optim_algo} vs {loop2_optim_algo}"
        )
        loop1_label = loop1_optim_algo
        loop2_label = loop2_optim_algo
        show_legend = False
    elif loop1_name == loop2_name and loop1_model_name == loop2_model_name:
        title = f"Version T-Test for {loop1_name} ({loop1_model_name}): {loop1_version} vs {loop2_version}"
        loop1_label = loop1_version
        loop2_label = loop2_version
        show_legend = False
    elif loop1_name == loop2_name and loop1_version == loop2_version:
        title = f"Model T-Test for {loop1_name} {loop1_version}: {loop1_model_name} vs {loop2_model_name}"
        loop1_label = loop1_model_name
        loop2_label = loop2_model_name
        show_legend = False
    elif loop1_name == loop2_name:
        # Version and model both differ.
        title = f"Loop T-Test for {loop1_name}: {loop1_version}/{loop1_model_name} vs {loop2_version}/{loop2_model_name}"
        loop1_label = f"{loop1_version}/{loop1_model_name}"
        loop2_label = f"{loop2_version}/{loop2_model_name}"
        show_legend = False
    else:
        title = "Loop T-Test: Loop1 vs Loop2"
        loop1_label = "Loop1"
        loop2_label = "Loop2"
        show_legend = True

    plt.figure(figsize=(16, 10))
    sns.violinplot(data=score_diff, color="skyblue", inner_kws={"box_width": 25, "whis_width": 5})  # pyright: ignore[reportArgumentType]
    plt.axhline(0, color="grey", linestyle="--")
    plt.axhline(0.1, color="grey", linestyle=":")
    plt.axhline(-0.1, color="grey", linestyle=":")
    plt.title(title, fontsize=14, weight="bold")

    stats = bootstrap_confidence_interval(score_diff.tolist())
    mean_score_diff = stats.mean
    ci_lower = stats.cilo
    ci_upper = stats.cihi
    # The CI may be asymmetric; report the larger side.
    error_upper = ci_upper - mean_score_diff
    error_lower = mean_score_diff - ci_lower
    error_margin = max(error_upper, error_lower)

    significance_text = r"significant" if p_value < p_value_threshold else r"not\ significant"
    newline = "\n"
    text_content = rf"Diff between mean scores: {mean_score_diff:.2f} ± {error_margin:.2f} (95% CI){newline}t: {t_value:.2f}, p={p_value:.3f}, Diff in favour of {loop1_label if mean_score_diff > 0 else loop2_label} is $\bf{{{significance_text}}}$"

    plt.text(
        0.95,
        0.95,
        text_content,
        ha="right",
        va="top",
        fontsize=12,
        transform=plt.gca().transAxes,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
    )
    plt.ylabel("Score Difference", fontsize=14)
    plt.ylim(-1.09, 1.09)

    if show_legend:
        legend_text = f"Loop1: {loop1}\nLoop2: {loop2}"
        num_lines = 2
        legend_y_position = 0.01 - (num_lines * 0.008)

        fig = plt.gcf()
        text_obj = fig.text(
            0.5,
            legend_y_position,
            legend_text,
            ha="center",
            va="bottom",
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9),
            transform=fig.transFigure,
        )

        # Render once to measure the legend box, then put a padded white rectangle behind it.
        fig.canvas.draw()
        renderer = fig.canvas.renderer  # pyright: ignore[reportAttributeAccessIssue]
        bbox = text_obj.get_window_extent(renderer=renderer)
        bbox_fig = bbox.transformed(fig.transFigure.inverted())

        padding = 0.02
        border_rect = mpatches.Rectangle(
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

    plt.show()


def plot_violin_distribution_between_loops(
    df_runs: pd.DataFrame,
    loop_names: list[str] | None = None,
    limit: int | None = None,
    p_value_threshold: float = 0.05,
) -> None:
    """Run ``t_test_between_loops`` for every pair of loop keys.

    Args:
        df_runs: Per-run scores.
        loop_names: If given (two or more), only loop keys whose ``loop_name`` is listed.
        limit: Compare only the ``limit`` loop keys with the highest mean score.
        p_value_threshold: Significance level.
    """
    df_runs_copy = df_runs.copy()

    if df_runs_copy["loop_key"].nunique() < 2:
        print("Not enough loops to perform t-test")
        return

    if not loop_names or len(loop_names) < 2:
        loop_means = df_runs_copy.groupby("loop_key")["score"].mean().sort_values(ascending=False)
    else:
        loop_means = (
            df_runs_copy[df_runs_copy["loop_name"].isin(loop_names)]
            .groupby("loop_key")["score"]
            .mean()
            .sort_values(ascending=False)
        )

    sorted_loops = loop_means.index.tolist()
    if limit:
        sorted_loops = sorted_loops[:limit]

    loop_pairs = list(combinations(sorted_loops, 2))

    for loop1, loop2 in loop_pairs:
        t_test_between_loops(df_runs_copy, loop1, loop2, p_value_threshold)
        display(HTML("<br>" * 10))


def _apply_bootstrap_ci_to_boxplot(
    ax: matplotlib.axes.Axes,
    boxes: list,
    all_lines: list,
    ci_data: dict[str, BootstrapStatistics],
    keys_order: list[str],
    is_vertical: bool = True,
    show_whiskers: bool = True,
) -> None:
    """Redraw seaborn or matplotlib boxplot elements in place to show bootstrap statistics.

    Boxes span the bootstrap CI instead of the IQR, the median line moves to the mean
    (drawn in white), and whiskers and caps are redrawn from the CI bounds to the data
    min and max (or hidden when ``show_whiskers`` is False). ``boxes`` and
    ``keys_order`` must be in the same order; ``ci_data`` maps each key to its
    ``BootstrapStatistics``.
    """
    for box, key in zip(boxes, keys_order):
        stats = ci_data[key]
        cilo = stats.cilo
        cihi = stats.cihi

        path = box.get_path()
        vertices = path.vertices

        if is_vertical:
            # Vertices: bottom-left, bottom-right, top-right, top-left, closing.
            for i, vertex in enumerate(vertices):
                if i in [0, 1, 4]:
                    vertices[i][1] = cilo
                elif i in [2, 3]:
                    vertices[i][1] = cihi
        else:
            # Vertices: left-bottom, left-top, right-top, right-bottom, closing.
            for i, vertex in enumerate(vertices):
                if i in [0, 1, 4]:
                    vertices[i][0] = cilo
                elif i in [2, 3]:
                    vertices[i][0] = cihi

        # A near-zero CI collapses the box; thicken its edge so it stays visible.
        ci_range = cihi - cilo
        if ci_range < 0.005:
            box.set_linewidth(3.0)
            box.set_edgecolor("black")

    for box, key in zip(boxes, keys_order):
        stats = ci_data[key]
        path = box.get_path()
        vertices = path.vertices
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        box_left = min(x_coords)
        box_right = max(x_coords)
        box_bottom = min(y_coords)
        box_top = max(y_coords)
        box_width = box_right - box_left
        box_height = box_top - box_bottom
        box_center_x = (box_left + box_right) / 2
        box_center_y = (box_bottom + box_top) / 2

        # Assign lines to this box by position.
        box_lines = []
        for line in all_lines:
            xdata = [float(x) for x in line.get_xdata()]  # type: ignore
            ydata = [float(y) for y in line.get_ydata()]  # type: ignore
            if len(xdata) == 2 and len(ydata) == 2:
                line_x_min, line_x_max = min(xdata), max(xdata)
                line_y_min, line_y_max = min(ydata), max(ydata)

                if is_vertical:
                    if (
                        box_left - 0.1 <= line_x_min <= box_right + 0.1
                        or box_left - 0.1 <= line_x_max <= box_right + 0.1
                    ):
                        box_lines.append(line)
                else:
                    if (
                        box_bottom - 0.1 <= line_y_min <= box_top + 0.1
                        or box_bottom - 0.1 <= line_y_max <= box_top + 0.1
                    ):
                        box_lines.append(line)

        # Whiskers run along the box axis; the median spans the full box width and
        # caps are narrower. Move the median to the mean and hide the rest.
        for line in box_lines:
            xdata = [float(x) for x in line.get_xdata()]  # type: ignore
            ydata = [float(y) for y in line.get_ydata()]  # type: ignore

            if len(xdata) == 2 and len(ydata) == 2:
                line_is_vertical = abs(xdata[0] - xdata[1]) < 0.01
                line_is_horizontal = abs(ydata[0] - ydata[1]) < 0.01

                if is_vertical:
                    if line_is_vertical:
                        line.set_visible(False)
                    elif line_is_horizontal:
                        line_width = abs(xdata[1] - xdata[0])
                        if line_width > box_width * 0.8:
                            line.set_ydata([stats.mean, stats.mean])
                            line.set_color("white")
                            line.set_linewidth(2.5)
                            line.set_zorder(4)
                        else:
                            line.set_visible(False)
                else:
                    if line_is_horizontal:
                        line.set_visible(False)
                    elif line_is_vertical:
                        line_height = abs(ydata[1] - ydata[0])
                        if line_height > box_height * 0.8:
                            line.set_xdata([stats.mean, stats.mean])
                            line.set_color("white")
                            line.set_linewidth(2.5)
                            line.set_zorder(4)
                        else:
                            line.set_visible(False)

        if show_whiskers:
            if is_vertical:
                cap_width = box_width * 0.5
                ax.plot(
                    [box_center_x, box_center_x],
                    [stats.whislo, stats.cilo],
                    color="black",
                    linewidth=1.5,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [box_center_x - cap_width / 2, box_center_x + cap_width / 2],
                    [stats.whislo, stats.whislo],
                    color="black",
                    linewidth=1.5,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [box_center_x, box_center_x],
                    [stats.cihi, stats.whishi],
                    color="black",
                    linewidth=1.5,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [box_center_x - cap_width / 2, box_center_x + cap_width / 2],
                    [stats.whishi, stats.whishi],
                    color="black",
                    linewidth=1.5,
                    solid_capstyle="butt",
                )
            else:
                cap_height = box_height * 0.5
                ax.plot(
                    [stats.whislo, stats.cilo],
                    [box_center_y, box_center_y],
                    color="black",
                    linewidth=1.5,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [stats.whislo, stats.whislo],
                    [box_center_y - cap_height / 2, box_center_y + cap_height / 2],
                    color="black",
                    linewidth=1.5,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [stats.cihi, stats.whishi],
                    [box_center_y, box_center_y],
                    color="black",
                    linewidth=1.5,
                    solid_capstyle="butt",
                )
                ax.plot(
                    [stats.whishi, stats.whishi],
                    [box_center_y - cap_height / 2, box_center_y + cap_height / 2],
                    color="black",
                    linewidth=1.5,
                    solid_capstyle="butt",
                )


def run_loop_analysis(
    df_runs: pd.DataFrame,
    p_value_threshold_anova: float = 0.05,
    p_value_threshold_tukey: float = 0.05,
    ci_confidence_level: float | None = 0.95,
    show_df: bool = True,
    show_means: bool = False,
    show_points: bool = True,
    show_whiskers: bool = True,
    show_anova: bool = True,
) -> None:
    """Compare all loop keys with a one-way ANOVA (Tukey HSD if significant) and a boxplot.

    Args:
        df_runs: Per-run scores.
        p_value_threshold_anova: Significance level for the ANOVA.
        p_value_threshold_tukey: Significance level for the Tukey HSD test.
        ci_confidence_level: Level of the hierarchical bootstrap CI (agents, then runs)
            drawn as the box. If None, boxes show the IQR and the table shows mean ± std.
        show_df: Display the loop-means and Tukey tables.
        show_anova: Print the ANOVA result on the plot.
    """
    data = df_runs.copy()
    data["optimization_algorithm"] = data["loop_key"].apply(extract_optimization_algorithm)

    loop_means = data.groupby("loop_key")["score"].mean().sort_values(ascending=False)
    sorted_loops = loop_means.index.tolist()

    loop_to_loop_name_dict: dict[str, str] = {
        str(k): str(v)
        for k, v in data[["loop_key", "loop_name"]]
        .drop_duplicates()
        .set_index("loop_key")["loop_name"]
        .to_dict()
        .items()
    }
    has_optimized_loops = any(
        extract_optimization_algorithm(loop_name) != "Default"
        for loop_name in loop_to_loop_name_dict.values()
    )
    has_model_comparison = len(data["model"].drop_duplicates()) > 1
    has_single_loop = (
        len(
            set(
                [
                    loop_name.removeprefix(extract_optimization_algorithm(loop_name))
                    for loop_name in loop_to_loop_name_dict.values()
                ]
            )
        )
        == 1
    )
    num_unique_agents = len(data[["agent", "agent_version"]].drop_duplicates())
    num_runs = len(data["run"].unique())

    title = f"Loop Comparison for {num_unique_agents} agents, {num_runs} runs per agent"
    if has_single_loop:
        single_loop_name = list(loop_to_loop_name_dict.values())[0]
        title += f"\nLoop: {single_loop_name.removeprefix(extract_optimization_algorithm(single_loop_name))}"

    if has_optimized_loops:
        grouped_loops = {}  # insertion order is the plot order
        for loop in sorted_loops:
            loop_name = loop_to_loop_name_dict[loop]
            optimization_algorithm = extract_optimization_algorithm(loop_name)
            base_loop = loop.removeprefix(optimization_algorithm)
            if base_loop not in grouped_loops:
                grouped_loops[base_loop] = []
            grouped_loops[base_loop].append(loop)

        loop_boundaries = [0]
        grouped_loops_order = []
        for loop_list in grouped_loops.values():
            loop_boundaries.append(len(loop_list) + loop_boundaries[-1])
            loop_list.sort(key=lambda item: data[data["loop_key"] == item]["score"].mean())
            grouped_loops_order.extend(loop_list)
    else:
        grouped_loops_order = sorted_loops
        grouped_loops = None
        loop_boundaries = None

    has_optimized_loops = (
        has_optimized_loops
        and grouped_loops
        and any(len(loop_list) > 1 for loop_list in grouped_loops.values())
    )

    loop_map = {loop: f"L{i + 1}" for i, loop in enumerate(grouped_loops_order)}
    ox_labels = [loop_map[loop] for loop in grouped_loops_order]

    loop_summary_df = pd.DataFrame(
        {"loop_name": sorted_loops, "score": [loop_means[loop] for loop in sorted_loops]},
        index=list(loop_map.values()),
    )
    print("Loop means (sorted by performance):")
    if show_df:
        display_df(loop_summary_df)

    anova_result_text = None
    if len(loop_means) >= 2:
        groups = []
        for loop in sorted_loops:
            group_data = data[data["loop_key"] == loop]["score"]
            groups.append(group_data)

        f_stat, p_value = f_oneway(*groups)

        anova_result_text = f"ANOVA: F={f_stat:.2f}, p={p_value:.3f}"
        best_loop = loop_means.idxmax()
        best_loop_label = loop_map[best_loop]
        anova_result_text += f"\nBest loop: {best_loop_label} (mean={loop_means[best_loop]:.3f})"

        print(f"ANOVA: F={f_stat:.2f}, p={p_value:.4f}")
        print(f"Best loop: {best_loop_label} (mean={loop_means[best_loop]:.3f})")

        if p_value < p_value_threshold_anova:
            data["loop_label"] = data["loop_key"].map(loop_map)
            tukey_result = pairwise_tukeyhsd(
                data["score"], data["loop_label"], alpha=p_value_threshold_tukey
            )

            tukey_df = pd.DataFrame(
                data=tukey_result._results_table.data[1:],
                columns=tukey_result._results_table.data[0],
            )

            print("Tukey test results:")
            if show_df:
                display_df(tukey_df)

    fig = plt.figure(figsize=(max(12, len(loop_means) * 1.3), 12))
    ax = fig.add_axes((0.1, 0.25, 0.8, 0.65))

    palette = None
    hue_key = "loop_key"
    if has_optimized_loops:
        hue_key = "optimization_algorithm"
    if has_model_comparison:
        hue_key = "model"
        models = list(data["model"].unique())
        palette = get_models_color_palette(models)

    ax = sns.boxplot(
        data=data,
        x="loop_key",
        y="score",
        hue=hue_key,
        order=grouped_loops_order,
        ax=ax,
        palette=palette,
        showmeans=show_means,
        showfliers=False,
        whis=(0 if not show_whiskers else 1.5),  # 0 hides the whiskers
    )

    ci_data: dict[str, BootstrapStatistics] = {}
    if ci_confidence_level:
        for loop in grouped_loops_order:
            loop_subset = data[data["loop_key"] == loop]
            # {agent_key: [run_scores]}
            hierarchical_data = {}
            for (agent, agent_version), agent_group in loop_subset.groupby(
                ["agent", "agent_version"]
            ):
                agent_key = f"{agent}_{agent_version}"
                hierarchical_data[agent_key] = agent_group["score"].tolist()
            stats = hierarchical_bootstrap_confidence_interval(
                hierarchical_data, confidence_level=ci_confidence_level
            )
            ci_data[loop] = stats

        # Align boxes with grouped_loops_order by sorting their horizontal positions.
        boxes = sorted(
            ax.patches[: len(grouped_loops_order)],
            key=lambda p: float(
                np.mean(
                    ax.transData.inverted().transform(
                        p.get_transform().transform(p.get_path().vertices)
                    )[:, 0]
                )
            ),
        )
        all_lines = ax.get_lines()

        _apply_bootstrap_ci_to_boxplot(
            ax,
            boxes,
            all_lines,
            ci_data,
            grouped_loops_order,
            is_vertical=True,
            show_whiskers=show_whiskers,
        )

    hatch_with_optim_algo_dict = {}
    hatch_patterns = ["/", "\\"]
    if has_optimized_loops:
        boxes = sorted(
            ax.patches[: len(grouped_loops_order)],
            key=lambda p: float(
                np.mean(
                    ax.transData.inverted().transform(
                        p.get_transform().transform(p.get_path().vertices)
                    )[:, 0]
                )
            ),
        )
        for box, loop in zip(boxes, grouped_loops_order):
            optim_algo = extract_optimization_algorithm(loop)
            if optim_algo != "Default":
                if optim_algo not in hatch_with_optim_algo_dict:
                    current_hatch = hatch_patterns.pop(0)
                    hatch_with_optim_algo_dict[optim_algo] = current_hatch
                else:
                    current_hatch = hatch_with_optim_algo_dict[optim_algo]

                box.set_hatch_linewidth(7)
                box.set_hatch(current_hatch)
                box.set_edgecolor("white")
                box.set_zorder(box.get_zorder() - 1)
                border_box = mpatches.PathPatch(
                    box.get_path(),
                    transform=box.get_transform(),
                    facecolor="black",
                    edgecolor="black",
                    linewidth=box.get_linewidth(),
                    linestyle=box.get_linestyle(),
                    fill=False,
                    zorder=box.get_zorder() + 1,
                )
                ax.add_patch(border_box)

    if show_points:
        sns.stripplot(
            data=data,
            x="loop_key",
            y="score",
            order=grouped_loops_order,
            color="black",
            alpha=0.5,
            jitter=True,
            size=5,
            ax=ax,
        )

    ax.set_xticks(range(len(ox_labels)))
    ax.set_xticklabels(ox_labels)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_title(title, pad=25, fontdict={"fontsize": 16})
    ax.set_ylim(0, 1.09)

    handles, labels = ax.get_legend_handles_labels()
    if hue_key == "model":
        # Order legend entries by first appearance along the x axis.
        ordered_hue_values = []
        for loop in grouped_loops_order:
            loop_data = data[data["loop_key"] == loop]
            if not loop_data.empty:
                hue_value = loop_data[hue_key].iloc[0]
                if hue_value not in ordered_hue_values:
                    ordered_hue_values.append(hue_value)

        if ordered_hue_values:
            reordered_handles = []
            reordered_labels = []
            for hue_value in ordered_hue_values:
                if hue_value in labels:
                    idx = labels.index(hue_value)
                    reordered_handles.append(handles[idx])
                    reordered_labels.append(labels[idx])

            for i, label in enumerate(labels):
                if label not in reordered_labels:
                    reordered_handles.append(handles[i])
                    reordered_labels.append(label)

            handles = reordered_handles
            labels = reordered_labels

    if has_optimized_loops:
        for optim_algo, hatch in hatch_with_optim_algo_dict.items():
            label = f"{optim_algo} Optimized Loop"
            hatched_patch = mpatches.Patch(
                facecolor="lightgray", edgecolor="black", hatch=hatch * 3, label=label
            )
            handles.append(hatched_patch)
            labels.append(label)

    ax.legend(
        handles,
        labels,
        loc="lower right",
        fontsize=9,
        framealpha=0.2,
        title="Legend",
        title_fontsize=10,
        ncol=1,
    )

    if has_optimized_loops and grouped_loops and loop_boundaries:
        for boundary in loop_boundaries[1:-1]:
            ax.axvline(x=boundary - 0.5, color="gray", linestyle="--", linewidth=1)

        # Arrow with the relative change for groups of exactly two loops.
        current_position = 0
        for _, loops_in_group in grouped_loops.items():
            if len(loops_in_group) == 2:
                loop1, loop2 = loops_in_group[0], loops_in_group[1]
                if ci_confidence_level:
                    mean1 = ci_data[loop1].mean
                    mean2 = ci_data[loop2].mean
                else:
                    mean1 = data[data["loop_key"] == loop1]["score"].mean()
                    mean2 = data[data["loop_key"] == loop2]["score"].mean()

                if mean1 != 0:
                    percent_change = ((mean2 - mean1) / mean1) * 100
                else:
                    percent_change = 0

                arrow_color = "green" if mean2 > mean1 else "red"
                x_start = current_position
                x_end = current_position + 1

                ax.annotate(
                    "",
                    xy=(x_end, mean2),
                    xytext=(x_start, mean1),
                    arrowprops={
                        "arrowstyle": "->",
                        "color": arrow_color,
                        "lw": 2,
                        "alpha": 0.7,
                    },
                )

                mid_x = (x_start + x_end) / 2
                mid_y = (mean1 + mean2) / 2
                ax.text(
                    mid_x,
                    mid_y,
                    f"{abs(percent_change):.1f}%",
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor=arrow_color,
                    ),
                    color=arrow_color,
                    fontweight="bold",
                )

            current_position += len(loops_in_group)

    table_data = []
    for loop in grouped_loops_order:
        label = loop_map[loop]
        loop_display = f"{label}: {loop}"

        if ci_confidence_level:
            stats = ci_data[loop]
            mean_score = stats.mean
            ci_lower = stats.cilo
            ci_upper = stats.cihi
            # The CI may be asymmetric; report the larger side.
            error_upper = ci_upper - mean_score
            error_lower = mean_score - ci_lower
            error_margin = max(error_upper, error_lower)
        else:
            loop_data = data[data["loop_key"] == loop]["score"]
            mean_score = loop_data.mean()
            error_margin = loop_data.std()

        table_data.append(
            {
                "Loop": loop_display,
                "Score": f"{mean_score:.3f}±{error_margin:.3f}",
            }
        )

    best_score = max(float(row["Score"].split("±")[0]) for row in table_data)
    cell_text = []
    for row in table_data:
        score_val = float(row["Score"].split("±")[0])
        if score_val == best_score:
            cell_text.append([row["Loop"], f"{row['Score']} ★"])
        else:
            cell_text.append([row["Loop"], row["Score"]])

    table_height = max(0.15, len(table_data) * 0.025 + 0.05)
    table = ax.table(
        cellText=cell_text,
        colLabels=["Loop", "Score"],
        cellLoc="left",
        loc="lower left",
        bbox=Bbox.from_bounds(0.0, -0.10 - table_height, 1.0, table_height),
        colWidths=[0.8, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for i in range(len(["Loop", "Score"])):
        table[(0, i)].set_text_props(weight="bold")

    for i, row in enumerate(table_data):
        score_val = float(row["Score"].split("±")[0])
        if score_val == best_score:
            table[(i + 1, 1)].set_facecolor("lightgreen")
            table[(i + 1, 1)].set_text_props(weight="bold")

    if show_anova and anova_result_text:
        ax.text(
            0.5,
            0.1,
            anova_result_text,
            fontsize=12,
            ha="center",
            va="top",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        )

    plt.show()
    display(HTML("<br>" * 10))
