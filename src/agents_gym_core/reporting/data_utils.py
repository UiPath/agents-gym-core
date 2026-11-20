"""Data loading utilities for reporting."""

import json
from pathlib import Path

import pandas as pd
from strictly_typed_pandas.dataset import DataSet

from agents_gym_core.reporting.schemas import EvaluatorDatapointResultSchema
from agents_gym_core.utils.gym_types import EvaluatorDatapointResult


def load_data(path: str | Path | list[str]) -> DataSet[EvaluatorDatapointResultSchema]:
    """Load ``combined_results.json`` file(s), validating each item as an
    ``EvaluatorDatapointResult``.
    """
    filtered_data = []
    paths_to_load = [path] if isinstance(path, (str, Path)) else path

    for p in paths_to_load:
        with open(p) as f:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                # Accept files that store the loop key as ``loop_version``.
                if "loop_version" in item and "loop_key" not in item:
                    item["loop_key"] = item["loop_version"]
                result = EvaluatorDatapointResult.model_validate(item)
                filtered_data.append(result.model_dump())

    return DataSet[EvaluatorDatapointResultSchema](filtered_data)


def load_data_from_dataframe(df: pd.DataFrame) -> DataSet[EvaluatorDatapointResultSchema]:
    """Convert a pandas DataFrame to a DataSet[EvaluatorDatapointResultSchema]."""
    if "loop_key" not in df.columns and "loop_version" in df.columns:
        df = df.rename(columns={"loop_version": "loop_key"})

    return DataSet[EvaluatorDatapointResultSchema](df.to_dict("records"))
