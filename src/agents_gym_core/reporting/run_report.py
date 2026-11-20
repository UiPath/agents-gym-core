"""Execute the report notebook on evaluation results and optionally export it as HTML."""

import argparse
import os
from pathlib import Path

import nbformat as nbf
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor

from agents_gym_core.reporting.styling import STYLING_CSS_FOR_HTML_EXPORT

DEFAULT_RESULTS_PATHS: list[str] = []
if (env_results_paths := os.getenv("RESULTS_PATHS", None)) is not None:
    DEFAULT_RESULTS_PATHS = [p.strip() for p in env_results_paths.split(",") if p.strip()]

DEFAULT_NOTEBOOK_PATH = str(Path(__file__).parent / "report_template.ipynb")


def run_notebook(
    path: str = DEFAULT_NOTEBOOK_PATH,
    output_path: str = "",
    results_paths: list[str] = DEFAULT_RESULTS_PATHS,
    confidence_level: float = 0.95,
    max_runs_to_plot: int = 0,
    no_overwrite: bool = False,
    export_html: bool = False,
    aggregation_mode: str = "datapoint_run",
) -> None:
    """Execute the notebook at ``path`` and save it to ``output_path``.

    Parameters reach the notebook through the ``RESULTS_PATHS``, ``CONFIDENCE_LEVEL``,
    ``MAX_RUNS_TO_PLOT`` and ``AGGREGATION_MODE`` environment variables. ``output_path``
    defaults to ``report_executed.ipynb`` next to the input notebook; with ``export_html``
    an ``.html`` copy without code cells is also written.
    """
    if not 0 < confidence_level < 1:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    # The notebook runs in its own folder, so relative paths are resolved here.
    os.environ["RESULTS_PATHS"] = ",".join(str(Path(p).resolve()) for p in results_paths)
    # Always set, so a previous call or an exported value does not leak into this report.
    os.environ["CONFIDENCE_LEVEL"] = str(confidence_level)
    os.environ["MAX_RUNS_TO_PLOT"] = str(max_runs_to_plot)  # <= 0 plots all runs
    os.environ["AGGREGATION_MODE"] = aggregation_mode

    if not output_path:
        output_path = str(Path(path).parent / "report_executed.ipynb")

    if os.path.exists(output_path) and no_overwrite:
        print(
            f"Output file {output_path} already exists. Remove the --no_overwrite flag to replace it."
        )
        return

    with open(path) as f:
        nb = nbf.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(Path(path).parent)}})

    with open(output_path, "w") as f:
        nbf.write(nb, f)
    print(f"Notebook executed and saved to {output_path}")

    if export_html:
        html_exporter = HTMLExporter()
        html_exporter.template_name = "classic"
        html_exporter.exclude_input = True
        (body, _) = html_exporter.from_notebook_node(nb)
        body = body.replace("</head>", STYLING_CSS_FOR_HTML_EXPORT + "</head>")

        html_path = Path(output_path).with_suffix(".html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(body)

        print(f"HTML report exported to {html_path}")


if __name__ == "__main__":
    from agents_gym_core.runtime.run_utils import AggregationMode

    parser = argparse.ArgumentParser(
        description="Execute Jupyter notebook report with evaluation results"
    )
    parser.add_argument("--path", type=str, default=DEFAULT_NOTEBOOK_PATH, help="Path to notebook")
    parser.add_argument(
        "--output_path", type=str, default="", help="Output path for executed notebook"
    )
    parser.add_argument(
        "--results_paths",
        type=str,
        nargs="+",
        default=DEFAULT_RESULTS_PATHS,
        help="Paths to result files",
    )
    parser.add_argument(
        "--confidence_level", type=float, default=0.95, help="Confidence level for bootstrap CI"
    )
    parser.add_argument(
        "--max_runs_to_plot", type=int, default=-1, help="Maximum number of runs to plot"
    )
    parser.add_argument(
        "--no_overwrite", action="store_true", help="Do not overwrite existing output"
    )
    parser.add_argument("--export_html", action="store_true", help="Export notebook as HTML")
    parser.add_argument(
        "--aggregation_mode",
        choices=[m.value for m in AggregationMode],
        default=AggregationMode.DATAPOINT_RUN.value,
        help="How the scores are aggregated, as for the testbed's --aggregation_mode",
    )
    args = parser.parse_args()

    run_notebook(
        path=args.path,
        output_path=args.output_path,
        results_paths=args.results_paths,
        confidence_level=args.confidence_level,
        max_runs_to_plot=args.max_runs_to_plot,
        no_overwrite=args.no_overwrite,
        export_html=args.export_html,
        aggregation_mode=args.aggregation_mode,
    )
