"""Utility functions for reporting."""

from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import HTML, display
from pandas.io.formats.style import Styler

MODELS_COLOR_PALETTE: dict[str, tuple[float, float, float, float]] = {}


def get_models_color_palette(
    model_names: list[str],
) -> dict[str, tuple[float, float, float, float]]:
    """Return a model-to-RGBA mapping that stays fixed across plots in a session."""
    global MODELS_COLOR_PALETTE
    if MODELS_COLOR_PALETTE and all(model in MODELS_COLOR_PALETTE for model in model_names):
        return MODELS_COLOR_PALETTE

    cmap = plt.get_cmap("tab10")  # reports compare at most 10 models
    cmap_index = len(MODELS_COLOR_PALETTE)
    for model in sorted(model_names):
        if model in MODELS_COLOR_PALETTE:
            continue
        MODELS_COLOR_PALETTE[model] = cmap(cmap_index)
        cmap_index += 1

    return MODELS_COLOR_PALETTE


def extract_optimization_algorithm(loop_name: str) -> str:
    """Return ``"Default"`` for the report's optimization-algorithm grouping."""
    return "Default"


def display_df(
    df: pd.DataFrame | pd.Series | Styler,
    index: bool = True,
    height: int | str = 300,
    width: int | str = "fit-content",
    scrollable: bool = True,
) -> None:
    """Render ``df`` as an HTML table; with ``scrollable``, cap its height at ``height`` px."""
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if scrollable:
        html = f"""
        <div style="max-height:{height}px; overflow:auto; width:{width}; border:1px solid #ccc;">
            {df.to_html(index=index, border=0)}
        </div>
        """
        display(HTML(html))
    else:
        html = f"""
        <div style="width:{width}; max-width:100%; overflow-x:auto; border:1px solid #ccc;">
            {df.to_html(index=index, border=0)}
        </div>
        """
        display(HTML(html))


def highlight_max_per_series(s: pd.Series) -> list[str]:
    """Return Styler CSS that highlights the maximum of ``s``.

    Values may be numbers or ``"mean ± margin"`` strings; strings compare by mean.
    """
    try:
        means = s.astype(float)
    except (TypeError, ValueError):
        means = s.str.split(" ±").str[0].astype(float)

    is_max = means == means.max()
    return ["background-color: green" if v else "" for v in is_max]


def highlight_best_per_agent(df: pd.DataFrame) -> pd.DataFrame:
    """Return Styler CSS that highlights, per row, the best version of each agent.

    Columns are ``(agent, agent_version)`` pairs holding ``"mean ± margin"`` strings.
    """
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    means = pd.DataFrame(index=df.index, columns=df.columns)
    for idx in df.index:
        for col in df.columns:
            if pd.notna(df.loc[idx, col]):
                means.loc[idx, col] = float(cast(str, df.loc[idx, col]).split(" ±")[0])

    for idx in means.index:
        for agent in means.columns.get_level_values(0).unique():
            agent_cols = [col for col in means.columns if col[0] == agent]
            agent_values = means.loc[idx, agent_cols].dropna()
            if not agent_values.empty:
                best_col = agent_values.idxmax()
                styles.loc[idx, best_col] = "background-color: green"  # type: ignore[call-overload]

    return styles
