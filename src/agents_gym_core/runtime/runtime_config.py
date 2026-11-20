"""Configuration for evaluation runs."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agents_gym_core.runtime.run_utils import AggregationMode, RunMode
from agents_gym_core.utils.gym_settings_core import GymSettingsCore


@dataclass
class RuntimeConfig:
    """Configuration for evaluation runs (shared across all managers and runners)."""

    models: list[dict[str, Any]] = field(default_factory=list)
    max_concurrency: int = 20
    max_datapoints: int = 0  # 0 = all
    datapoint_names: list[str] | None = None  # None = all; list = filter by exact name match
    splits: list[str] | None = None  # None = all; list = filter by split label (e.g. ["train"])

    verbose: bool = False
    print_trace: bool = False

    parallel_tool_calls: bool = False

    base_output_dir: Path = Path("./experiments")
    experiment_name: str = "experiment"

    mode: RunMode = "fill"

    num_runs: int = 1  # Times each datapoint is executed

    aggregation_mode: AggregationMode = AggregationMode.DATAPOINT_RUN

    # Return tool exceptions to the agent as messages instead of failing the run.
    handle_tool_errors: bool = False

    save_traces: bool = True  # Write output.json for each datapoint run

    settings: GymSettingsCore | None = None
