from collections.abc import Generator
from itertools import islice
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from agents_gym_core.common_tools.vector_database import DEFAULT_EMBEDDINGS_MODEL
from agents_gym_core.loops import get_loop
from agents_gym_core.loops.types import DEFAULT_LOOP_VERSION
from agents_gym_core.models.chat import get_chat_model
from agents_gym_core.models.embeddings import get_embedding_model
from agents_gym_core.models.utils import ModelBackend, default_backend
from agents_gym_core.runtime.cost_tracking import DRIVER_TAG
from agents_gym_core.runtime.run_utils import GraphDatapoint
from agents_gym_core.utils.gym_settings_core import GymSettingsCore
from agents_gym_core.utils.gym_types import Agent
from agents_gym_core.utils.trace_utils import warning


def agent_graphs_with_datapoints(
    agents: list[Agent],
    loops: list[tuple[str, str]] | None = None,
    models: dict[str, Any] | None = None,
    max_datapoints: int = 0,
    print_trace: bool = False,
    parallel_tool_calls: bool = False,
    num_runs: int = 1,
    handle_tool_errors: bool = False,
    settings: GymSettingsCore | None = None,
    datapoint_names: list[str] | None = None,
    splits: list[str] | None = None,
) -> tuple[Generator[GraphDatapoint, None, None], int]:
    """Compile one graph per (agent, loop) for a model config and pair it with each datapoint run.

    A compiled graph is shared by all datapoints and runs: LangGraph creates fresh
    execution state per invocation.

    Args:
        loops: (loop_name, version) pairs, e.g. ``[("basic", "any"), ("planning", "auto")]``;
            defaults to ``[("basic", DEFAULT_LOOP_VERSION)]``. An agent's own ``loop:``
            replaces the default basic loop.
        models: Model names by role, e.g. ``{"model": "gpt-4o", "planning_llm": "gpt-4.1-mini"}``.
        max_datapoints: Datapoints per agent, or per split when ``splits`` is set (0 = all).
        num_runs: Runs per datapoint, numbered from 1.
        datapoint_names: Keep only these datapoints (exact names); overrides ``max_datapoints``
            and ``splits``.
        splits: Keep only datapoints whose ``agent.datapoint_splits`` label is listed, e.g.
            ``["train"]``. Labels come from eval-set filename suffixes (``_train.json`` ->
            ``"train"``); unlabelled datapoints (``"all_data"``) are kept only if
            ``"all_data"`` is listed.

    Returns:
        A generator of ``GraphDatapoint`` tuples and the number it yields. Loop metadata is
        on ``graph.loop_details``.
    """
    models = models or {}
    models.setdefault("embeddings", DEFAULT_EMBEDDINGS_MODEL)
    if loops is None:
        loops = [("basic", DEFAULT_LOOP_VERSION)]

    datapoint_names_set = set(datapoint_names) if datapoint_names else None
    splits_set = set(splits) if splits else None
    if datapoint_names_set:
        max_datapoints = 0
        splits_set = None

    total_datapoints = 0
    for agent in agents:
        if datapoint_names_set:
            agent_datapoints = sum(
                1 for dp in agent.evaluation_set.evaluations if dp.name in datapoint_names_set
            )
        elif splits_set:
            if max_datapoints > 0:
                agent_datapoints = 0
                for split_label in splits_set:
                    split_count = sum(
                        1
                        for dp in agent.evaluation_set.evaluations
                        if agent.datapoint_splits.get(dp.name, "all_data") == split_label
                    )
                    agent_datapoints += min(split_count, max_datapoints)
            else:
                agent_datapoints = sum(
                    1
                    for dp in agent.evaluation_set.evaluations
                    if agent.datapoint_splits.get(dp.name, "all_data") in splits_set
                )
        else:
            agent_datapoints = len(agent.evaluation_set.evaluations)
            if max_datapoints > 0:
                agent_datapoints = min(agent_datapoints, max_datapoints)
        total_datapoints += agent_datapoints * len(loops) * num_runs

    if datapoint_names_set and total_datapoints == 0:
        warning(
            f"No datapoints matched the provided filter {sorted(datapoint_names_set)!r}. "
            "Check that names are spelled correctly.",
        )
    if splits_set and total_datapoints == 0:
        warning(
            f"No datapoints matched the splits filter {sorted(splits_set)!r}. "
            "Check that the eval-set filenames use a split suffix (_train, _val, _test, _dev or __<custom>).",
        )

    def generator() -> Generator[GraphDatapoint, None, None]:
        # A copy, so popping "from_factory" and "backend" leaves the caller's dict intact.
        _models = dict(models)
        from_factory = _models.pop("from_factory", False)
        backend = ModelBackend(_models.pop("backend", None) or default_backend())

        model_objects: dict[str, BaseChatModel | Embeddings] = {
            k: get_embedding_model(v, backend=backend)
            if k == "embeddings"
            else get_chat_model(v, from_factory=from_factory, backend=backend, tags=[DRIVER_TAG])
            for k, v in _models.items()
            if v is not None
        }

        for agent in agents:
            # An agent's loop: key replaces the default basic loop (basic or basic:auto).
            if agent.loop:
                parts = agent.loop.split(":", 1)
                agent_loop = (parts[0], parts[1] if len(parts) > 1 else DEFAULT_LOOP_VERSION)
                agent_loops = [
                    agent_loop if lp == ("basic", DEFAULT_LOOP_VERSION) else lp for lp in loops
                ]
            else:
                agent_loops = loops
            for loop_name, version in agent_loops:
                graph = get_loop(
                    loop_name=loop_name,
                    version=version,
                    scenario=agent,
                    models=model_objects,
                    print_trace=print_trace,
                    parallel_tool_calls=parallel_tool_calls,
                    handle_tool_errors=handle_tool_errors,
                    settings=settings,
                )
                datapoints = agent.evaluation_set.evaluations
                if datapoint_names_set:
                    datapoints = [dp for dp in datapoints if dp.name in datapoint_names_set]
                elif splits_set:
                    if max_datapoints > 0:
                        capped: list = []
                        per_split_counts: dict[str, int] = {}
                        for dp in datapoints:
                            label = agent.datapoint_splits.get(dp.name, "all_data")
                            if label not in splits_set:
                                continue
                            count = per_split_counts.get(label, 0)
                            if count < max_datapoints:
                                capped.append(dp)
                                per_split_counts[label] = count + 1
                        datapoints = capped
                    else:
                        datapoints = [
                            dp
                            for dp in datapoints
                            if agent.datapoint_splits.get(dp.name, "all_data") in splits_set
                        ]
                elif max_datapoints > 0:
                    datapoints = islice(datapoints, max_datapoints)
                for datapoint in datapoints:
                    for run_number in range(1, num_runs + 1):
                        yield (graph, datapoint, agent, run_number)

    return generator(), total_datapoints
