"""Draw loop graphs, inline in reports or as PNG files.

Usage as script:
    python -m agents_gym_core.reporting.loop_visualization [--output-dir PATH]
"""

from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from warnings import catch_warnings, simplefilter

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from PIL import Image
from pydantic import BaseModel, Field
from strictly_typed_pandas.dataset import DataSet
from uipath.eval.models.evaluation_set import EvaluationSet

from agents_gym_core.loops.basic_loop import BasicLoop, BasicLoopAny
from agents_gym_core.loops.loop_factory import LoopName, get_loop
from agents_gym_core.loops.planning_loop import (
    PlanningLoop,
    PlanningSystemPlanLoop,
    PlanningUserPlanLoop,
)
from agents_gym_core.loops.summarizing_loop import SummarizingLoop
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION
from agents_gym_core.loops.validation_loop import ValidationLoop
from agents_gym_core.reporting.schemas import AgentScoreSummarySchema
from agents_gym_core.utils.gym_types import Agent


class VisualizationInput(BaseModel):
    """Input schema for the loop-diagram agent."""

    query: str = Field(default="")


class VisualizationOutput(BaseModel):
    """Output schema for the loop-diagram agent."""

    response: str = Field(default="")


@tool
def tools(query: str) -> str:
    """Representative tool used to construct loop diagrams."""
    return f"Result for: {query}"


def create_visualization_agent() -> Agent:
    """Create the representative agent used to build loop diagrams."""
    return Agent(
        name="visualization_agent",
        system_prompt="Minimal agent for visualization",
        user_prompt="{query}",
        tools=[tools],
        evaluation_set=EvaluationSet(id="viz", name="visualization", evaluations=[]),
        input_schema=VisualizationInput,
        output_schema=VisualizationOutput,
    )


def create_visualization_model() -> BaseChatModel:
    """Return a ``BaseChatModel`` mock, so graphs build without API credentials."""
    mock_model = MagicMock(spec=BaseChatModel)
    mock_model.bind_tools = MagicMock(return_value=mock_model)
    return mock_model


def _get_loop_enum_from_name(loop_name: str) -> LoopName | None:
    """Match a loop name to a ``LoopName``: exactly first, then by substring or prefix."""
    for loop_enum in LoopName:
        if loop_enum.value == loop_name or loop_enum.name == loop_name:
            return loop_enum

    loop_name_lower = loop_name.lower().replace("_", "-")
    for loop_enum in LoopName:
        if loop_enum.value in loop_name_lower or loop_name_lower.startswith(loop_enum.value):
            return loop_enum

    return None


def draw_loop_diagram(compiled_graph: Any, ax: Axes, title: str | None = None) -> bool:
    """Draw a compiled LangGraph graph on ``ax``; return False if drawing failed."""
    try:
        if title:
            ax.set_title(title, fontsize=18, pad=20, y=1.05)

        graph_drawable = compiled_graph.get_graph()

        if hasattr(graph_drawable, "draw_mermaid_png"):
            png_bytes = graph_drawable.draw_mermaid_png()
            img = Image.open(BytesIO(png_bytes))
            ax.imshow(img)
            ax.axis("off")
        elif hasattr(compiled_graph, "draw"):
            compiled_graph.draw(ax=ax)
        else:
            node_names = list(graph_drawable.nodes)
            ax.text(
                0.5,
                0.5,
                f"Graph: {title or 'Unknown'}\nNodes: {', '.join(node_names[:5])}{'...' if len(node_names) > 5 else ''}",
                ha="center",
                va="center",
                fontsize=12,
                transform=ax.transAxes,
                wrap=True,
            )
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        return True

    except Exception as e:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"Error drawing {title or 'diagram'}:\n{str(e)[:50]}",
            ha="center",
            va="center",
            fontsize=14,
            color="red",
        )
        return False


def draw_graphs(
    summary_df: DataSet[AgentScoreSummarySchema],
    num_columns: int = 3,
    single_width: int = 8,
    single_height: int = 10,
) -> None:
    """Draw a grid with one diagram per distinct ``loop_name`` in ``summary_df``."""
    loop_names = summary_df["loop_name"].unique()

    try:
        minimal_agent = create_visualization_agent()
        minimal_model = create_visualization_model()
    except Exception as e:
        print(f"Could not create minimal objects for visualization: {e}")
        return

    all_diagrams: list[tuple[str, Any]] = []
    for loop_name in loop_names:
        try:
            loop_enum = _get_loop_enum_from_name(loop_name)
            if loop_enum is not None:
                compiled_graph = get_loop(
                    loop_name=loop_enum,
                    version=DEFAULT_LOOP_VERSION,
                    scenario=minimal_agent,
                    models={"model": minimal_model},
                )
                all_diagrams.append((loop_name, compiled_graph))
            else:
                print(f"Could not find LoopName enum for: {loop_name}")
        except Exception as e:
            print(f"Could not create loop diagram for {loop_name}: {e}")

    if not all_diagrams:
        print("No loop diagrams could be loaded for visualization.")
        return

    num_loops = len(all_diagrams)
    rows = max(1, (num_loops + num_columns - 1) // num_columns)
    num_columns = min(num_columns, num_loops)
    fig = plt.figure(figsize=(single_width * num_columns, single_height * rows))

    gs = fig.add_gridspec(rows, num_columns, hspace=0.3, wspace=0.3)

    for i, (loop_name, compiled_graph) in enumerate(all_diagrams):
        if i >= rows * num_columns:
            break

        row_idx = i // num_columns
        col_idx = i % num_columns

        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.clear()

        if compiled_graph:
            draw_loop_diagram(compiled_graph, ax, title=loop_name)
        else:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"Loop {loop_name} not available",
                ha="center",
                va="center",
                fontsize=14,
                color="red",
            )

    for i in range(len(all_diagrams), rows * num_columns):
        row_idx = i // num_columns
        col_idx = i % num_columns
        ax = fig.add_subplot(gs[row_idx, col_idx])
        ax.axis("off")

    with catch_warnings():
        simplefilter("ignore", category=UserWarning)
        fig.tight_layout(pad=2.0)

    plt.show()


def get_all_loop_classes() -> list[tuple[str, type]]:
    """Return ``(name, class)`` pairs for the loops drawn by ``visualize_all_loops``."""
    return [
        ("basic_loop", BasicLoop),
        ("basic_loop_any", BasicLoopAny),
        ("planning_loop", PlanningLoop),
        ("planning_system_plan_loop", PlanningSystemPlanLoop),
        ("planning_user_plan_loop", PlanningUserPlanLoop),
        ("validation_loop", ValidationLoop),
        ("summarizing_loop", SummarizingLoop),
    ]


def visualize_all_loops(
    output_dir: Path | str | None = None,
    show_plots: bool = False,
) -> None:
    """Save a PNG diagram of each loop to ``output_dir`` (default ``reporting/loop_diagrams/``)."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "loop_diagrams"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    agent = create_visualization_agent()
    model = create_visualization_model()

    loops_to_visualize = get_all_loop_classes()

    print(f"Creating visualizations in: {output_dir}")
    print("-" * 60)

    for loop_name, loop_class in loops_to_visualize:
        try:
            loop_instance = loop_class(agent, model)
            graph = loop_instance.build_graph()
            compiled_graph = graph.compile()

            output_file = output_dir / f"{loop_name}.png"
            png_data = compiled_graph.get_graph().draw_mermaid_png()

            with open(output_file, "wb") as f:
                f.write(png_data)

            print(f"✓ {loop_name:40s} -> {output_file.name}")

            if show_plots:
                _, ax = plt.subplots(figsize=(10, 12))
                draw_loop_diagram(compiled_graph, ax, title=loop_name)
                plt.show()

        except Exception as e:
            import traceback

            print(f"✗ {loop_name:40s} -> ERROR: {e}")
            traceback.print_exc()

    print("-" * 60)
    print(f"Visualizations saved to: {output_dir.absolute()}")


def main() -> None:
    """Entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize all agent loop implementations as PNG diagrams."
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Output directory for PNG files (default: reporting/loop_diagrams/)",
    )
    parser.add_argument(
        "--show",
        "-s",
        action="store_true",
        help="Also display plots interactively",
    )

    args = parser.parse_args()
    visualize_all_loops(output_dir=args.output_dir, show_plots=args.show)


if __name__ == "__main__":
    main()
