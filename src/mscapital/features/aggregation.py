from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike


def local_group_ids(sample_ids: ArrayLike, sample_start: int) -> np.ndarray:
    ids = np.asarray(sample_ids, dtype=np.int64) - sample_start
    if ids.ndim != 1:
        raise ValueError("sample_ids must be one-dimensional")
    return ids


def transition_mask(
    group_ids: ArrayLike, seconds_before_predict: ArrayLike, window: float
) -> np.ndarray:
    ids = np.asarray(group_ids)
    seconds = np.asarray(seconds_before_predict, dtype=np.float64)
    if ids.ndim != 1 or ids.shape != seconds.shape:
        raise ValueError("group_ids and seconds_before_predict must be matching vectors")
    selected = np.zeros(len(ids), dtype=bool)
    if len(ids) > 1:
        selected[1:] = (
            (ids[1:] == ids[:-1])
            & (seconds[1:] <= window)
            & (seconds[:-1] <= window)
        )
    return selected


def group_count(group_ids: ArrayLike, n_groups: int, mask: ArrayLike | None = None) -> np.ndarray:
    ids = np.asarray(group_ids, dtype=np.int64)
    selected = _mask(mask, len(ids))
    return np.bincount(ids[selected], minlength=n_groups).astype(np.float64, copy=False)


def group_sum(
    group_ids: ArrayLike,
    values: ArrayLike,
    n_groups: int,
    mask: ArrayLike | None = None,
) -> np.ndarray:
    ids, data, selected = _selected(group_ids, values, mask)
    selected &= np.isfinite(data)
    return np.bincount(ids[selected], weights=data[selected], minlength=n_groups).astype(
        np.float64, copy=False
    )


def group_weighted_mean(
    group_ids: ArrayLike,
    values: ArrayLike,
    weights: ArrayLike,
    n_groups: int,
    mask: ArrayLike | None = None,
) -> np.ndarray:
    ids = np.asarray(group_ids, dtype=np.int64)
    data = np.asarray(values, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)
    if ids.shape != data.shape or ids.shape != weight_values.shape:
        raise ValueError("group_ids, values, and weights must have matching shapes")
    selected = _mask(mask, len(ids)) & np.isfinite(data) & np.isfinite(weight_values) & (weight_values > 0)
    numerator = np.bincount(
        ids[selected], weights=data[selected] * weight_values[selected], minlength=n_groups
    )
    denominator = np.bincount(ids[selected], weights=weight_values[selected], minlength=n_groups)
    result = np.full(n_groups, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def group_first(
    group_ids: ArrayLike,
    values: ArrayLike,
    n_groups: int,
    mask: ArrayLike | None = None,
) -> np.ndarray:
    return _group_edge(group_ids, values, n_groups, mask, first=True)


def group_last(
    group_ids: ArrayLike,
    values: ArrayLike,
    n_groups: int,
    mask: ArrayLike | None = None,
) -> np.ndarray:
    return _group_edge(group_ids, values, n_groups, mask, first=False)


def aggregate_series(
    prefix: str,
    group_ids: ArrayLike,
    values: ArrayLike,
    n_groups: int,
    *,
    mask: ArrayLike | None = None,
    seconds: ArrayLike | None = None,
    stats: Iterable[str] = ("last", "mean", "std", "min", "max", "delta"),
) -> dict[str, np.ndarray]:
    ids, data, selected = _selected(group_ids, values, mask)
    selected &= np.isfinite(data)
    selected_ids = ids[selected]
    selected_data = data[selected]
    counts = np.bincount(selected_ids, minlength=n_groups).astype(np.float64, copy=False)
    sums = np.bincount(selected_ids, weights=selected_data, minlength=n_groups)
    result: dict[str, np.ndarray] = {}
    requested = tuple(stats)

    mean = np.full(n_groups, np.nan, dtype=np.float64)
    if any(name in requested for name in ("mean", "std")):
        np.divide(sums, counts, out=mean, where=counts > 0)
    if "sum" in requested:
        result[f"{prefix}__sum"] = sums
    if "count" in requested:
        result[f"{prefix}__count"] = counts
    if "mean" in requested:
        result[f"{prefix}__mean"] = mean
    if "std" in requested:
        sum_squares = np.bincount(
            selected_ids, weights=selected_data * selected_data, minlength=n_groups
        )
        variance = np.zeros(n_groups, dtype=np.float64)
        np.divide(sum_squares, counts, out=variance, where=counts > 0)
        variance -= np.nan_to_num(mean, nan=0.0) ** 2
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)
        std[counts == 0] = np.nan
        result[f"{prefix}__std"] = std

    if any(name in requested for name in ("first", "last", "delta")):
        first = _group_edge(ids, data, n_groups, selected, first=True, mask_is_ready=True)
        last = _group_edge(ids, data, n_groups, selected, first=False, mask_is_ready=True)
        if "first" in requested:
            result[f"{prefix}__first"] = first
        if "last" in requested:
            result[f"{prefix}__last"] = last
        if "delta" in requested:
            result[f"{prefix}__delta"] = last - first

    if "min" in requested or "max" in requested:
        minimum = np.full(n_groups, np.inf, dtype=np.float64)
        maximum = np.full(n_groups, -np.inf, dtype=np.float64)
        np.minimum.at(minimum, selected_ids, selected_data)
        np.maximum.at(maximum, selected_ids, selected_data)
        minimum[counts == 0] = np.nan
        maximum[counts == 0] = np.nan
        if "min" in requested:
            result[f"{prefix}__min"] = minimum
        if "max" in requested:
            result[f"{prefix}__max"] = maximum

    if "slope" in requested:
        if seconds is None:
            raise ValueError("seconds are required for slope")
        time_values = -np.asarray(seconds, dtype=np.float64)
        if time_values.shape != ids.shape:
            raise ValueError("seconds and group_ids must have matching shapes")
        x = time_values[selected]
        sum_x = np.bincount(selected_ids, weights=x, minlength=n_groups)
        sum_xx = np.bincount(selected_ids, weights=x * x, minlength=n_groups)
        sum_xy = np.bincount(selected_ids, weights=x * selected_data, minlength=n_groups)
        denominator = counts * sum_xx - sum_x * sum_x
        numerator = counts * sum_xy - sum_x * sums
        slope = np.zeros(n_groups, dtype=np.float64)
        np.divide(numerator, denominator, out=slope, where=np.abs(denominator) > 1e-12)
        slope[counts == 0] = np.nan
        result[f"{prefix}__slope"] = slope
    return result


def fill_grouped_forward_backward(
    values: ArrayLike,
    group_ids: ArrayLike,
    fallback_by_group: ArrayLike,
) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    ids = np.asarray(group_ids, dtype=np.int64)
    fallback = np.asarray(fallback_by_group, dtype=np.float64)
    if data.shape != ids.shape:
        raise ValueError("values and group_ids must have matching shapes")
    if len(data) == 0:
        return data.copy()
    changes = np.concatenate(([0], np.flatnonzero(ids[1:] != ids[:-1]) + 1, [len(ids)]))
    starts = changes[:-1]
    ends = changes[1:]
    lengths = ends - starts
    group_starts = np.repeat(starts, lengths)
    group_ends = np.repeat(ends - 1, lengths)
    finite = np.isfinite(data)

    forward_index = np.where(finite, np.arange(len(data), dtype=np.int64), -1)
    np.maximum.accumulate(forward_index, out=forward_index)
    forward_valid = forward_index >= group_starts

    backward_index = np.where(finite, np.arange(len(data), dtype=np.int64), len(data))
    backward_index = np.minimum.accumulate(backward_index[::-1])[::-1]
    backward_valid = backward_index <= group_ends

    result = data.copy()
    missing = ~finite
    use_forward = missing & forward_valid
    result[use_forward] = data[forward_index[use_forward]]
    use_backward = missing & ~forward_valid & backward_valid
    result[use_backward] = data[backward_index[use_backward]]
    still_missing = ~np.isfinite(result)
    result[still_missing] = fallback[ids[still_missing]]
    return result


def feature_matrix(sample_ids: np.ndarray, features: dict[str, np.ndarray]):
    from mscapital.contracts import FeatureMatrix

    names = tuple(features)
    if not names:
        values = np.empty((len(sample_ids), 0), dtype=np.float32)
    else:
        values = np.column_stack([np.asarray(features[name], dtype=np.float32) for name in names])
    if np.isinf(values).any():
        raise ValueError("feature matrix contains infinite values")
    return FeatureMatrix(np.asarray(sample_ids, dtype=np.int32), values, names)


def _selected(group_ids: ArrayLike, values: ArrayLike, mask: ArrayLike | None):
    ids = np.asarray(group_ids, dtype=np.int64)
    data = np.asarray(values, dtype=np.float64)
    if ids.shape != data.shape:
        raise ValueError("group_ids and values must have matching shapes")
    return ids, data, _mask(mask, len(ids))


def _mask(mask: ArrayLike | None, length: int) -> np.ndarray:
    if mask is None:
        return np.ones(length, dtype=bool)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (length,):
        raise ValueError("mask must match input length")
    return selected.copy()


def _group_edge(
    group_ids: ArrayLike,
    values: ArrayLike,
    n_groups: int,
    mask: ArrayLike | None,
    *,
    first: bool,
    mask_is_ready: bool = False,
) -> np.ndarray:
    ids = np.asarray(group_ids, dtype=np.int64)
    data = np.asarray(values, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool) if mask_is_ready else _mask(mask, len(ids))
    selected &= np.isfinite(data)
    selected_ids = ids[selected]
    selected_data = data[selected]
    result = np.full(n_groups, np.nan, dtype=np.float64)
    if len(selected_ids) == 0:
        return result
    changes = np.concatenate(([0], np.flatnonzero(selected_ids[1:] != selected_ids[:-1]) + 1))
    positions = changes if first else np.concatenate((changes[1:] - 1, [len(selected_ids) - 1]))
    result[selected_ids[positions]] = selected_data[positions]
    return result
