# Reporting

Analysis reports for evaluation results:
- scores with bootstrap confidence intervals
- statistical comparisons between loops (ANOVA, Tukey HSD, paired t-tests)
- plots of agent performance across loops, agent versions and models
- correlation plots between loop pairs
- per-agent LLM cost tables

Report scores are computed with `calculate_agent_scores()` from `runtime/runners/result_aggregator.py`, the same function that writes `final_score_per_agent.json`.

## Module Structure

```
agents_gym_core/reporting/
├── __init__.py                 # Module exports
├── README.md                   # This file
├── main.py                     # Score tables, pivot tables, loop and version comparisons
├── loop_analysis.py            # Statistical tests (ANOVA, t-tests, violin plots)
├── advanced_usage.py           # Correlation analysis
├── cost_reporting.py           # Per-agent cost table
├── loop_visualization.py       # Loop graph diagrams
├── utils.py                    # Display, colors, highlighting
├── styling.py                  # CSS for HTML exports
├── schemas.py                  # DataFrame schemas
├── data_utils.py               # Data loading
├── run_report.py               # Executes the report notebook
└── report_template.ipynb       # Report notebook template
```

## Usage

### Programmatic Usage

```python
import agents_gym_core.reporting as rp
from agents_gym_core.reporting import load_data

# Load evaluation results
df = load_data(["path/to/results.json"])

# Calculate scores for reporting
summary, per_run = rp.calculate_agent_scores_for_report(
    df,
    ci_confidence_level=0.95,
    max_runs_to_plot=-1
)

# Generate pivot tables
rp.pivot_on_loops(summary, loop_on_column=False)
rp.pivot_on_versions(summary, pivot_type="loop")

# Statistical analysis
rp.run_loop_analysis(
    per_run,
    p_value_threshold_anova=0.05,
    p_value_threshold_tukey=0.05,
    ci_confidence_level=0.95
)

# Comparison visualizations
rp.compare_loops_and_agent_versions(
    summary,
    per_run,
    clustering="best",
    orientation="best",
    use_ci_boxplots=True
)

# Correlation analysis
rp.run_correlation_analysis(per_run)
```

### Running Notebook Reports

```bash
# Run from Python
python -m agents_gym_core.reporting.run_report \
    --results_paths /path/to/results.json \
    --confidence_level 0.95 \
    --output_path /path/to/report_executed.ipynb \
    --export_html

# Or set environment variable and run
export RESULTS_PATHS=/path/to/results.json
python -m agents_gym_core.reporting.run_report --output_path report_executed.ipynb --export_html
```

### Using the Jupyter Notebook

Open and run [report_template.ipynb](report_template.ipynb) in Jupyter. Set the `RESULTS_PATHS` environment variable or edit the notebook cells to point to your results.

## Score Calculation

Report scores and `final_score_per_agent.json` share one implementation:

1. Report generation (`calculate_agent_scores_for_report` in `main.py`) calls `calculate_agent_scores()` and formats the result as DataSets for plotting.
2. Final score JSON (`result_aggregator.py`) calls the same `calculate_agent_scores()` and saves `final_score_per_agent.json`.

Both rely on:
- `bootstrap_confidence_interval()` (`utils/bootstrap.py`) for confidence intervals
- `_aggregate_evaluator_run()` / `_aggregate_datapoint_run()` for score aggregation
- `_build_final_scores_with_ci()` for the final scores structure

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- strictly-typed-pandas
- IPython (for notebook display)
- nbformat and nbconvert (for notebook execution)
