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
    group_weighted_mean,
    local_group_ids,
    transition_mask,
)
from mscapital.features.base import FeatureContext


TRANSACTION_COLUMNS = (
    "sample_id",
    "seconds_before_predict",
    "price",
    "volume",
    "side",
)


class TransactionFeatureBlock:
    name = "transaction"
    version = 2
    required_columns = TRANSACTION_COLUMNS
    required_tables = (TableName.TRANSACTION, TableName.MARKET)

    def __init__(self, config: FeaturesConfig) -> None:
        self.config = config

    def transform(
        self,
        columns: Mapping[str, np.ndarray],
        context: FeatureContext,
    ) -> FeatureMatrix:
        ids = local_group_ids(columns["sample_id"], context.shard.sample_start)
        seconds = np.asarray(columns["seconds_before_predict"], dtype=np.float64)
        price = np.asarray(columns["price"], dtype=np.float64)
        volume = np.asarray(columns["volume"], dtype=np.float64)
        side = np.asarray(columns["side"], dtype=np.int8)
        n_groups = context.shard.sample_count
        reference = np.asarray(context.reference_mid, dtype=np.float64)[ids]
        price_bps = np.log(np.maximum(price, 1e-12) / reference) * 10_000.0
        direction = np.where(side == 0, 1.0, -1.0)
        signed_volume = direction * volume

        gap = np.full(len(ids), np.nan, dtype=np.float64)
        persistence = np.full(len(ids), np.nan, dtype=np.float64)
        same_sample = np.zeros(len(ids), dtype=bool)
        same_sample[1:] = ids[1:] == ids[:-1]
        gap[1:] = np.where(same_sample[1:], seconds[:-1] - seconds[1:], np.nan)
        persistence[1:] = np.where(same_sample[1:], side[1:] == side[:-1], np.nan)

        features: dict[str, np.ndarray] = {
            "transaction__row_count": group_count(ids, n_groups),
            "transaction__last_event_seconds": group_last(ids, seconds, n_groups),
        }
        for window in self.config.transaction_windows:
            mask = seconds <= window
            transition = transition_mask(ids, seconds, window)
            tag = f"w{window}"
            total_count = group_count(ids, n_groups, mask)
            total_volume = group_sum(ids, volume, n_groups, mask)
            buy_mask = mask & (side == 0)
            sell_mask = mask & (side == 1)
            buy_count = group_count(ids, n_groups, buy_mask)
            sell_count = group_count(ids, n_groups, sell_mask)
            buy_volume = group_sum(ids, volume, n_groups, buy_mask)
            sell_volume = group_sum(ids, volume, n_groups, sell_mask)
            signed_total = group_sum(ids, signed_volume, n_groups, mask)

            features[f"transaction__{tag}__event_count"] = total_count
            features[f"transaction__{tag}__event_rate"] = total_count / float(window)
            features[f"transaction__{tag}__volume_logsum"] = np.log1p(total_volume)
            features[f"transaction__{tag}__buy_count"] = buy_count
            features[f"transaction__{tag}__sell_count"] = sell_count
            features[f"transaction__{tag}__buy_volume_logsum"] = np.log1p(buy_volume)
            features[f"transaction__{tag}__sell_volume_logsum"] = np.log1p(sell_volume)
            features[f"transaction__{tag}__count_imbalance"] = _safe_ratio(
                buy_count - sell_count, total_count
            )
            features[f"transaction__{tag}__volume_imbalance"] = _safe_ratio(
                signed_total, total_volume
            )
            features[f"transaction__{tag}__large_trade_rate"] = _safe_ratio(
                group_count(ids, n_groups, mask & (volume >= self.config.large_trade_volume)),
                total_count,
            )
            features[f"transaction__{tag}__vwap_bps"] = group_weighted_mean(
                ids, price_bps, volume, n_groups, mask
            )
            features.update(
                aggregate_series(
                    f"transaction__{tag}__price_bps",
                    ids,
                    price_bps,
                    n_groups,
                    mask=mask,
                    seconds=seconds,
                    stats=("last", "mean", "std", "min", "max", "delta", "slope"),
                )
            )
            features.update(
                aggregate_series(
                    f"transaction__{tag}__event_gap",
                    ids,
                    gap,
                    n_groups,
                    mask=transition,
                    stats=("mean", "std", "max"),
                )
            )
            features[f"transaction__{tag}__direction_persistence"] = aggregate_series(
                "persistence",
                ids,
                persistence,
                n_groups,
                mask=transition,
                stats=("mean",),
            )["persistence__mean"]
        return feature_matrix(context.sample_ids, features)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(np.asarray(numerator, dtype=np.float64))
    np.divide(numerator, denominator, out=result, where=np.asarray(denominator) != 0)
    return result
