"""Per-agent cost table for the notebook report.

Costs are read from the ``final_score_per_agent.json`` that ``ResultAggregator``
writes next to each ``combined_results.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from IPython.display import HTML, display

FINAL_SCORES_FILENAME = "final_score_per_agent.json"

_COLUMNS_ORDER: list[str] = [
    "loop_key",
    "agent",
    "version",
    "split",
    "cost_per_run_usd",
    "cost_per_datapoint",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "num_llm_calls",
    "by_model",
    "num_datapoints",
    "num_runs",
    "total_experiment_cost_usd",
]


def _resolve_final_scores_paths(results_paths: list[str] | str | Path) -> list[Path]:
    """Map combined_results.json paths to their sibling final_score_per_agent.json."""
    if isinstance(results_paths, (str, Path)):
        results_paths = [str(results_paths)]

    seen: set[Path] = set()
    resolved: list[Path] = []
    for p in results_paths:
        candidate = Path(p).parent / FINAL_SCORES_FILENAME
        if candidate.exists() and candidate not in seen:
            seen.add(candidate)
            resolved.append(candidate)
    return resolved


def load_cost_table(results_paths: list[str] | str | Path) -> pd.DataFrame:
    """Load cost data from ``final_score_per_agent.json`` into a flat DataFrame.

    One row per (loop_key, agent, split); entries without a ``cost`` block are skipped.
    """
    rows: list[dict[str, Any]] = []
    for path in _resolve_final_scores_paths(results_paths):
        with open(path) as f:
            data = json.load(f)
        for loop_key, agents_dict in (data or {}).items():
            for agent_key, splits_dict in (agents_dict or {}).items():
                if "__" in agent_key:
                    agent_name, agent_version = agent_key.split("__", 1)
                else:
                    agent_name, agent_version = agent_key, "base"
                for split, score_data in (splits_dict or {}).items():
                    cost = (score_data or {}).get("cost")
                    if not cost:
                        continue
                    raw_cpd = cost.get("cost_per_datapoint", 0.0)
                    if isinstance(raw_cpd, dict):
                        raw_cpd = sum(raw_cpd.values()) / len(raw_cpd) if raw_cpd else 0.0
                    rows.append(
                        {
                            "source": str(path),  # each split of a run repeats its cost
                            "loop_key": loop_key,
                            "agent": agent_name,
                            "version": agent_version,
                            "split": split,
                            "cost_per_run_usd": float(cost.get("total_cost_usd", 0.0)),
                            "cost_per_datapoint": float(raw_cpd),
                            "input_tokens": float(cost.get("input_tokens", 0.0)),
                            "output_tokens": float(cost.get("output_tokens", 0.0)),
                            "cached_tokens": float(cost.get("cached_tokens", 0.0)),
                            "num_llm_calls": float(cost.get("num_llm_calls", 0.0)),
                            "by_model": cost.get("by_model", {}),
                            "num_datapoints": int(cost.get("num_datapoints", 0)),
                            "num_runs": int(cost.get("num_runs", 0)),
                            "total_experiment_cost_usd": float(
                                cost.get("total_experiment_cost_usd", 0.0)
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def _format_cost_df_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Format costs as dollars, round counts, flatten ``by_model`` and order the columns."""
    if df.empty:
        return df

    out = df.copy()
    for col in ("cost_per_run_usd", "cost_per_datapoint", "total_experiment_cost_usd"):
        out[col] = out[col].map(lambda v: f"${v:,.6f}" if isinstance(v, (int, float)) else v)
    out["input_tokens"] = out["input_tokens"].round(1)
    out["output_tokens"] = out["output_tokens"].round(1)
    out["cached_tokens"] = out["cached_tokens"].round(1)
    out["num_llm_calls"] = out["num_llm_calls"].round(2)

    def _models(d: dict[str, float]) -> str:
        if not d:
            return ""
        return ", ".join(f"{m}: ${v:,.6f}" for m, v in sorted(d.items()))

    out["by_model"] = out["by_model"].map(_models)
    return out[[c for c in _COLUMNS_ORDER if c in out.columns]]


def _append_total_row(df: pd.DataFrame) -> pd.DataFrame:
    """Append a TOTAL row summing the numeric columns; other columns are left blank."""
    if df.empty:
        return df

    once = df.drop_duplicates(["source", "loop_key", "agent", "version"])  # one cost per run
    total_row: dict[str, Any] = {col: "" for col in df.columns}
    total_row["loop_key"] = "TOTAL"
    total_row["cost_per_run_usd"] = float(once["cost_per_run_usd"].sum())
    total_row["cost_per_datapoint"] = float(once["cost_per_datapoint"].sum())
    total_row["input_tokens"] = float(once["input_tokens"].sum())
    total_row["output_tokens"] = float(once["output_tokens"].sum())
    total_row["cached_tokens"] = float(once["cached_tokens"].sum())
    total_row["num_llm_calls"] = float(once["num_llm_calls"].sum())
    total_row["num_datapoints"] = int(once["num_datapoints"].sum())
    total_row["num_runs"] = int(once["num_runs"].sum())
    total_row["total_experiment_cost_usd"] = float(once["total_experiment_cost_usd"].sum())
    return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)


def display_cost_table(results_paths: list[str] | str | Path) -> pd.DataFrame:
    """Display the cost table (or a note when no cost was recorded) and return it unformatted."""
    df = load_cost_table(results_paths)
    if df.empty:
        display(HTML("<em>No cost data recorded for these runs.</em>"))
        return df

    df_sorted = df.sort_values(["loop_key", "agent", "version", "split"]).reset_index(drop=True)
    df_with_total = _append_total_row(df_sorted)
    table_html = _format_cost_df_for_display(df_with_total).to_html(index=False, border=0)
    display(
        HTML(
            f'<div style="overflow-x:auto; overflow-y:visible; width:100%; '
            f'border:1px solid #ccc;">{table_html}</div>'
        )
    )
    return df_sorted
