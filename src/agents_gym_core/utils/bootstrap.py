from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, Field


class BootstrapStatistics(BaseModel):
    """Bootstrap confidence interval statistics."""

    mean: float = Field(description="Mean of the data")
    cilo: float = Field(description="Confidence interval lower bound")
    cihi: float = Field(description="Confidence interval upper bound")
    whislo: float = Field(description="Lower whisker (data minimum)")
    whishi: float = Field(description="Upper whisker (data maximum)")


def bootstrap_confidence_interval(
    data: Sequence[float],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapStatistics:
    """Percentile bootstrap confidence interval for the mean; whiskers are the data min and max."""
    if len(data) == 0:
        return BootstrapStatistics(mean=0.0, cilo=0.0, cihi=0.0, whislo=0.0, whishi=0.0)
    if len(data) == 1:
        val = float(data[0])
        return BootstrapStatistics(mean=val, cilo=val, cihi=val, whislo=val, whishi=val)

    rng = np.random.default_rng(random_seed)
    data_array = np.array(data)

    bootstrap_samples = rng.choice(data_array, size=(n_bootstrap, len(data_array)), replace=True)
    bootstrap_means = np.mean(bootstrap_samples, axis=1)

    alpha = 1 - confidence_level
    lower_bound, upper_bound = np.percentile(
        bootstrap_means, [(alpha / 2) * 100, (1 - alpha / 2) * 100]
    )
    whisker_lo = float(np.min(data_array))
    whisker_hi = float(np.max(data_array))
    mean = float(np.mean(data_array))

    return BootstrapStatistics(
        mean=mean,
        cilo=float(lower_bound),
        cihi=float(upper_bound),
        whislo=whisker_lo,
        whishi=whisker_hi,
    )


def hierarchical_bootstrap_confidence_interval(
    hierarchical_data: dict[str, Sequence[float]],
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> BootstrapStatistics:
    """Two-level bootstrap confidence interval for grouped data, e.g. run scores per agent.

    Each bootstrap sample resamples the groups with replacement, then the values within each
    chosen group, and takes the mean of all resampled values; this accounts for correlation
    within a group. ``mean`` and the whiskers come from the original values of all groups.
    """
    if len(hierarchical_data) == 0:
        return BootstrapStatistics(mean=0.0, cilo=0.0, cihi=0.0, whislo=0.0, whishi=0.0)

    group_ids = list(hierarchical_data.keys())
    group_arrays = [np.array(hierarchical_data[gid]) for gid in group_ids]
    n_groups = len(group_ids)

    if n_groups == 1:
        return bootstrap_confidence_interval(
            list(hierarchical_data.values())[0], n_bootstrap, confidence_level, random_seed
        )

    rng = np.random.default_rng(random_seed)

    group_sizes = [len(arr) for arr in group_arrays]
    equal_sizes = len(set(group_sizes)) == 1 and group_sizes[0] > 0

    if equal_sizes:
        # Equal group sizes allow a fully vectorized path.
        group_size = group_sizes[0]

        # (n_bootstrap, n_groups): which groups each bootstrap sample draws.
        group_indices = rng.integers(0, n_groups, size=(n_bootstrap, n_groups))

        # A fresh within-group resample for each draw, so a group drawn twice is resampled twice.
        within_indices = rng.integers(0, group_size, size=(n_bootstrap, n_groups, group_size))
        stacked_groups = np.stack(group_arrays)  # (n_groups, group_size)
        selected_resamples = stacked_groups[
            group_indices[:, :, None],  # (n_bootstrap, n_groups, 1)
            within_indices,  # (n_bootstrap, n_groups, group_size)
        ]  # (n_bootstrap, n_groups, group_size)
        bootstrap_means = selected_resamples.reshape(n_bootstrap, -1).mean(axis=1)

    else:
        group_indices = rng.integers(0, n_groups, size=(n_bootstrap, n_groups))

        # Groups padded to the largest size; a draw of a group uses its first len(group) slots.
        max_size = max(group_sizes)
        padded_groups = np.zeros((n_groups, max_size))
        for gidx, group_array in enumerate(group_arrays):
            padded_groups[gidx, : len(group_array)] = group_array
        drawn_sizes = np.array(group_sizes)[group_indices]  # (n_bootstrap, n_groups)

        # A fresh within-group resample for each draw, so a group drawn twice is resampled twice.
        within_indices = rng.integers(
            0, np.maximum(drawn_sizes, 1)[:, :, None], size=(n_bootstrap, n_groups, max_size)
        )
        selected_resamples = padded_groups[group_indices[:, :, None], within_indices]
        in_group = np.arange(max_size) < drawn_sizes[:, :, None]
        bootstrap_means = (selected_resamples * in_group).sum(axis=(1, 2)) / drawn_sizes.sum(axis=1)

    alpha = 1 - confidence_level
    lower_bound, upper_bound = np.percentile(
        bootstrap_means, [(alpha / 2) * 100, (1 - alpha / 2) * 100]
    )

    all_values = np.concatenate([arr for arr in group_arrays if len(arr) > 0])
    mean = float(np.mean(all_values)) if len(all_values) > 0 else 0.0

    whisker_lo = float(np.min(all_values)) if len(all_values) > 0 else 0.0
    whisker_hi = float(np.max(all_values)) if len(all_values) > 0 else 0.0

    return BootstrapStatistics(
        mean=mean,
        cilo=float(lower_bound),
        cihi=float(upper_bound),
        whislo=whisker_lo,
        whishi=whisker_hi,
    )
