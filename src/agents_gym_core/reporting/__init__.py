"""Score aggregation, statistical tests, and plots for evaluation results."""

from agents_gym_core.reporting.advanced_usage import run_correlation_analysis
from agents_gym_core.reporting.cost_reporting import (
    display_cost_table,
    load_cost_table,
)
from agents_gym_core.reporting.data_utils import load_data
from agents_gym_core.reporting.loop_analysis import (
    plot_violin_distribution_between_loops,
    run_loop_analysis,
    t_test_between_loops,
)
from agents_gym_core.reporting.loop_visualization import (
    draw_graphs,
    visualize_all_loops,
)
from agents_gym_core.reporting.main import (
    calculate_agent_scores_for_report,
    compare_loops_and_agent_versions,
    loops_testcase_comparison,
    pivot_agent_versions_average,
    pivot_on_loops,
    pivot_on_versions,
    sort_testcases_by_loop_difference,
)
from agents_gym_core.reporting.utils import display_df

__all__ = [
    "calculate_agent_scores_for_report",
    "compare_loops_and_agent_versions",
    "display_cost_table",
    "draw_graphs",
    "load_cost_table",
    "load_data",
    "loops_testcase_comparison",
    "pivot_agent_versions_average",
    "pivot_on_loops",
    "pivot_on_versions",
    "plot_violin_distribution_between_loops",
    "run_correlation_analysis",
    "run_loop_analysis",
    "display_df",
    "sort_testcases_by_loop_difference",
    "t_test_between_loops",
    "visualize_all_loops",
]
