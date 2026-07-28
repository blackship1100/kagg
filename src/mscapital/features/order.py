from __future__ import annotations

from typing import Mapping

import numpy as np

from mscapital.config import FeaturesConfig
from mscapital.contracts import FeatureMatrix, TableName
from mscapital.features.aggregation import (
    aggregate_series,
    feature_matrix,
    group_count,
    group_last,
    group_sum,
    local_group_ids,
    transition_mask,
)
from mscapital.features.base import FeatureContext


ORDER_COLUMNS = (
    "sample_id",
    "seconds_before_predict",
    "price",
    "volume",
    "side",
    "order_action",
)


class OrderFeatureBlock:
    name = "order"
    version = 2
    required_columns = ORDER_COLUMNS
    required_tables = (TableName.ORDER, TableName.MARKET)

    def __init__(self, config: FeaturesConfig) -> None:
        self.config = config

    def transform(
        self,
        columns: Mapping[str, np.ndarray],
        context: FeatureContext,
    ) -> FeatureMatrix:
        ids = local_group_ids(columns["sample_id"], context.shard.sample_start)
        seconds = np.asarray(columns["seconds_before_predict"], dtype=np.float64)
        raw_price = np.asarray(columns["price"], dtype=np.float64)
        volume = np.asarray(columns["volume"], dtype=np.float64)
        side = np.asarray(columns["side"], dtype=np.int8)
        action = np.asarray(columns["order_action"], dtype=np.int8)
        n_groups = context.shard.sample_count
        row_reference = np.asarray(context.reference_mid, dtype=np.float64)[ids]

        lower, upper = self.config.order_price_clip
        price = np.clip(raw_price, lower, upper)
        clipped = price != raw_price
        distance_bps = np.log(np.maximum(price, 1e-12) / row_reference) * 10_000.0
        side_sign = np.where(side == 0, 1.0, -1.0)
        action_sign = np.where(action == 0, 1.0, -1.0)
        direction = side_sign * action_sign
        signed_volume = direction * volume

        gap = np.full(len(ids), np.nan, dtype=np.float64)
        same_sample = np.zeros(len(ids), dtype=bool)
        same_sample[1:] = ids[1:] == ids[:-1]
        gap[1:] = np.where(same_sample[1:], seconds[:-1] - seconds[1:], np.nan)

        categories = {
            "buy_new": (side == 0) & (action == 0),
            "buy_cancel": (side == 0) & (action == 1),
            "sell_new": (side == 1) & (action == 0),
            "sell_cancel": (side == 1) & (action == 1),
        }
        absolute_distance = np.abs(distance_bps)
        distance_masks = _distance_masks(absolute_distance, self.config.order_distance_bins_bps)
        features: dict[str, np.ndarray] = {
            "order__row_count": group_count(ids, n_groups),
            "order__last_event_seconds": group_last(ids, seconds, n_groups),
        }

        for window in self.config.order_windows:
            mask = seconds <= window
            transition = transition_mask(ids, seconds, window)
            tag = f"w{window}"
            count_total = group_count(ids, n_groups, mask)
            volume_total = group_sum(ids, volume, n_groups, mask)
            signed_total = group_sum(ids, signed_volume, n_groups, mask)
            cancel_volume = group_sum(ids, volume, n_groups, mask & (action == 1))
            cancel_count = group_count(ids, n_groups, mask & (action == 1))
            features[f"order__{tag}__event_count"] = count_total
            features[f"order__{tag}__event_rate"] = count_total / float(window)
            features[f"order__{tag}__volume_logsum"] = np.log1p(volume_total)
            features[f"order__{tag}__signed_volume_imbalance"] = _safe_ratio(
                signed_total, volume_total
            )
            features[f"order__{tag}__cancel_count_rate"] = _safe_ratio(
                cancel_count, count_total
            )
            features[f"order__{tag}__cancel_volume_rate"] = _safe_ratio(
                cancel_volume, volume_total
            )
            features[f"order__{tag}__large_order_rate"] = _safe_ratio(
                group_count(ids, n_groups, mask & (volume >= self.config.large_order_volume)),
                count_total,
            )
            features[f"order__{tag}__price_clipped_rate"] = _safe_ratio(
                group_count(ids, n_groups, mask & clipped), count_total
            )
            features.update(
                aggregate_series(
                    f"order__{tag}__distance_bps",
                    ids,
                    distance_bps,
                    n_groups,
                    mask=mask,
                    stats=("last", "mean", "std", "max"),
                )
            )
            features.update(
                aggregate_series(
                    f"order__{tag}__event_gap",
                    ids,
                    gap,
                    n_groups,
                    mask=transition,
                    stats=("mean", "std", "max"),
                )
            )
            for name, category_mask in categories.items():
                category_count = group_count(ids, n_groups, mask & category_mask)
                category_volume = group_sum(ids, volume, n_groups, mask & category_mask)
                features[f"order__{tag}__{name}_count"] = category_count
                features[f"order__{tag}__{name}_volume_logsum"] = np.log1p(category_volume)
            for name, distance_mask in distance_masks.items():
                bucket_count = group_count(ids, n_groups, mask & distance_mask)
                bucket_volume = group_sum(ids, volume, n_groups, mask & distance_mask)
                features[f"order__{tag}__distance_{name}_count"] = bucket_count
                features[f"order__{tag}__distance_{name}_volume_logsum"] = np.log1p(
                    bucket_volume
                )
        return feature_matrix(context.sample_ids, features)


def _distance_masks(values: np.ndarray, bins: tuple[int, ...]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    lower = 0.0
    for upper in bins:
        result[f"{int(lower)}_{upper}"] = (values >= lower) & (values < upper)
        lower = float(upper)
    result[f"{int(lower)}_plus"] = values >= lower
    return result


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(np.asarray(numerator, dtype=np.float64))
    np.divide(numerator, denominator, out=result, where=np.asarray(denominator) != 0)
    return result
