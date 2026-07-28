from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from mscapital.config import FeaturesConfig
from mscapital.contracts import FeatureMatrix, TableName
from mscapital.data.canonical import ShardSpec
from mscapital.data.cleaning import classify_book_level, classify_market_trades, signed_log1p
from mscapital.features.aggregation import (
    aggregate_series,
    feature_matrix,
    fill_grouped_forward_backward,
    group_count,
    group_first,
    group_last,
    group_sum,
    local_group_ids,
    transition_mask,
)
from mscapital.features.base import FeatureContext


MARKET_COLUMNS = (
    "sample_id",
    "seconds_before_predict",
    "transaction_avgprice",
    "transaction_volume",
    "transaction_count",
    "ask_price_1",
    "ask_volume_1",
    "bid_price_1",
    "bid_volume_1",
    "ask_price_2",
    "ask_volume_2",
    "bid_price_2",
    "bid_volume_2",
)


@dataclass(frozen=True)
class MarketDerived:
    context: FeatureContext
    group_ids: np.ndarray
    seconds: np.ndarray
    mid_bps: np.ndarray
    spread_bps: np.ndarray
    microprice_bps: np.ndarray
    imbalance_l1: np.ndarray
    imbalance_l2: np.ndarray
    log_depth_l1: np.ndarray
    log_depth_l2: np.ndarray
    book_valid_l1: np.ndarray
    book_valid_l2: np.ndarray
    trade_price_bps: np.ndarray
    positive_trade_volume: np.ndarray
    positive_trade_count: np.ndarray
    has_trade: np.ndarray
    no_trade: np.ndarray
    is_correction: np.ndarray
    correction_volume: np.ndarray
    correction_count: np.ndarray


class MarketFeatureBlock:
    name = "market"
    version = 3
    required_columns = MARKET_COLUMNS
    required_tables = (TableName.MARKET,)

    def __init__(self, config: FeaturesConfig, *, strict: bool = True) -> None:
        self.config = config
        self.strict = strict

    def derive(self, columns: Mapping[str, np.ndarray], shard: ShardSpec) -> MarketDerived:
        ids = np.asarray(columns["sample_id"])
        group_ids = local_group_ids(ids, shard.sample_start)
        n_groups = shard.sample_count
        ask1 = np.asarray(columns["ask_price_1"], dtype=np.float64)
        bid1 = np.asarray(columns["bid_price_1"], dtype=np.float64)
        ask2 = np.asarray(columns["ask_price_2"], dtype=np.float64)
        bid2 = np.asarray(columns["bid_price_2"], dtype=np.float64)
        ask_volume1 = np.asarray(columns["ask_volume_1"], dtype=np.float64)
        bid_volume1 = np.asarray(columns["bid_volume_1"], dtype=np.float64)
        ask_volume2 = np.asarray(columns["ask_volume_2"], dtype=np.float64)
        bid_volume2 = np.asarray(columns["bid_volume_2"], dtype=np.float64)

        ask1_state = classify_book_level(ask1, ask_volume1, strict=self.strict)
        bid1_state = classify_book_level(bid1, bid_volume1, strict=self.strict)
        ask2_state = classify_book_level(ask2, ask_volume2, strict=self.strict)
        bid2_state = classify_book_level(bid2, bid_volume2, strict=self.strict)
        ask_volume1 = np.where(ask1_state.valid, ask_volume1, 0.0)
        bid_volume1 = np.where(bid1_state.valid, bid_volume1, 0.0)
        ask_volume2 = np.where(ask2_state.valid, ask_volume2, 0.0)
        bid_volume2 = np.where(bid2_state.valid, bid_volume2, 0.0)
        two_sided = ask1_state.valid & bid1_state.valid
        level2_two_sided = ask2_state.valid & bid2_state.valid
        raw_mid = np.full(len(ids), np.nan, dtype=np.float64)
        raw_mid[two_sided] = (ask1[two_sided] + bid1[two_sided]) / 2.0
        reference = group_last(group_ids, raw_mid, n_groups)

        last_ask = group_last(group_ids, ask1, n_groups, mask=ask1 > 0)
        last_bid = group_last(group_ids, bid1, n_groups, mask=bid1 > 0)
        one_sided = np.where(
            np.isfinite(last_ask) & np.isfinite(last_bid),
            (last_ask + last_bid) / 2.0,
            np.where(np.isfinite(last_ask), last_ask, last_bid),
        )
        reference = np.where(np.isfinite(reference), reference, one_sided)
        no_valid_mid = ~np.isfinite(reference) | (reference <= 0)
        reference = np.where(no_valid_mid, 1.0, reference)
        filled_mid = fill_grouped_forward_backward(raw_mid, group_ids, reference)
        row_reference = reference[group_ids]
        mid_bps = np.log(np.maximum(filled_mid, 1e-12) / row_reference) * 10_000.0

        spread_bps = np.full(len(ids), np.nan, dtype=np.float64)
        spread_bps[two_sided] = (ask1[two_sided] - bid1[two_sided]) / raw_mid[two_sided] * 10_000.0
        depth1 = ask_volume1 + bid_volume1
        microprice = np.full(len(ids), np.nan, dtype=np.float64)
        micro_valid = two_sided & (depth1 > 0)
        microprice[micro_valid] = (
            ask1[micro_valid] * bid_volume1[micro_valid]
            + bid1[micro_valid] * ask_volume1[micro_valid]
        ) / depth1[micro_valid]
        microprice_bps = np.full(len(ids), np.nan, dtype=np.float64)
        microprice_bps[micro_valid] = (
            np.log(microprice[micro_valid] / filled_mid[micro_valid]) * 10_000.0
        )

        imbalance_l1 = _safe_imbalance(bid_volume1, ask_volume1)
        imbalance_l2 = _safe_imbalance(bid_volume1 + bid_volume2, ask_volume1 + ask_volume2)
        log_depth_l1 = np.log1p(np.maximum(depth1, 0.0))
        log_depth_l2 = np.log1p(
            np.maximum(ask_volume1 + ask_volume2 + bid_volume1 + bid_volume2, 0.0)
        )

        avgprice = np.asarray(columns["transaction_avgprice"], dtype=np.float64)
        volume = np.asarray(columns["transaction_volume"], dtype=np.float64)
        count = np.asarray(columns["transaction_count"], dtype=np.float64)
        trade_state = classify_market_trades(avgprice, volume, count, strict=self.strict)
        has_trade = trade_state.positive_trade
        no_trade = trade_state.no_trade
        is_correction = trade_state.correction
        trade_price_bps = np.zeros(len(ids), dtype=np.float64)
        trade_price_bps[has_trade] = (
            np.log(np.maximum(avgprice[has_trade], 1e-12) / filled_mid[has_trade]) * 10_000.0
        )

        context = FeatureContext(
            shard=shard,
            reference_mid=reference.astype(np.float32),
            no_valid_mid=no_valid_mid,
        )
        return MarketDerived(
            context=context,
            group_ids=group_ids,
            seconds=np.asarray(columns["seconds_before_predict"], dtype=np.float64),
            mid_bps=mid_bps,
            spread_bps=spread_bps,
            microprice_bps=microprice_bps,
            imbalance_l1=imbalance_l1,
            imbalance_l2=imbalance_l2,
            log_depth_l1=log_depth_l1,
            log_depth_l2=log_depth_l2,
            book_valid_l1=two_sided.astype(np.float64),
            book_valid_l2=level2_two_sided.astype(np.float64),
            trade_price_bps=trade_price_bps,
            positive_trade_volume=np.where(has_trade, volume, 0.0),
            positive_trade_count=np.where(has_trade, count, 0.0),
            has_trade=has_trade.astype(np.float64),
            no_trade=no_trade.astype(np.float64),
            is_correction=is_correction.astype(np.float64),
            correction_volume=np.where(is_correction, volume, 0.0),
            correction_count=np.where(is_correction, count, 0.0),
        )

    def transform_derived(self, derived: MarketDerived) -> FeatureMatrix:
        context = derived.context
        n_groups = context.shard.sample_count
        ids = derived.group_ids
        seconds = derived.seconds
        features: dict[str, np.ndarray] = {
            "market__reference_mid": context.reference_mid,
            "market__no_valid_mid": context.no_valid_mid.astype(np.float64),
            "market__row_count": group_count(ids, n_groups),
        }
        first_second = group_first(ids, seconds, n_groups)
        last_second = group_last(ids, seconds, n_groups)
        features["market__observed_span"] = first_second - last_second
        features["market__last_event_seconds"] = last_second

        same_sample = np.zeros(len(ids), dtype=bool)
        same_sample[1:] = ids[1:] == ids[:-1]
        mid_returns = np.full(len(ids), np.nan, dtype=np.float64)
        mid_returns[1:] = np.where(
            same_sample[1:], derived.mid_bps[1:] - derived.mid_bps[:-1], np.nan
        )

        dense_signals = {
            "mid_bps": derived.mid_bps,
            "spread_bps": derived.spread_bps,
            "microprice_bps": derived.microprice_bps,
            "imbalance_l1": derived.imbalance_l1,
            "imbalance_l2": derived.imbalance_l2,
            "trade_price_bps": derived.trade_price_bps,
        }
        depth_signals = {
            "log_depth_l1": derived.log_depth_l1,
            "log_depth_l2": derived.log_depth_l2,
        }
        for window in self.config.market_windows:
            mask = seconds <= window
            tag = f"w{window}"
            features[f"market__{tag}__rows"] = group_count(ids, n_groups, mask)
            features[f"market__{tag}__book_valid_l1_rate"] = group_sum(
                ids, derived.book_valid_l1, n_groups, mask
            ) / np.maximum(features[f"market__{tag}__rows"], 1.0)
            features[f"market__{tag}__book_valid_l2_rate"] = group_sum(
                ids, derived.book_valid_l2, n_groups, mask
            ) / np.maximum(features[f"market__{tag}__rows"], 1.0)
            for name, values in dense_signals.items():
                features.update(
                    aggregate_series(
                        f"market__{tag}__{name}",
                        ids,
                        values,
                        n_groups,
                        mask=mask,
                        seconds=seconds,
                        stats=("last", "mean", "std", "min", "max", "delta", "slope"),
                    )
                )
            for name, values in depth_signals.items():
                features.update(
                    aggregate_series(
                        f"market__{tag}__{name}",
                        ids,
                        values,
                        n_groups,
                        mask=mask,
                        stats=("last", "mean", "std", "max"),
                    )
                )
            volume_sum = group_sum(ids, derived.positive_trade_volume, n_groups, mask)
            count_sum = group_sum(ids, derived.positive_trade_count, n_groups, mask)
            row_count = np.maximum(group_count(ids, n_groups, mask), 1.0)
            features[f"market__{tag}__trade_volume_logsum"] = np.log1p(volume_sum)
            features[f"market__{tag}__trade_count_logsum"] = np.log1p(count_sum)
            features[f"market__{tag}__has_trade_rate"] = (
                group_sum(ids, derived.has_trade, n_groups, mask) / row_count
            )
            features[f"market__{tag}__no_trade_rate"] = (
                group_sum(ids, derived.no_trade, n_groups, mask) / row_count
            )
            features[f"market__{tag}__correction_count"] = group_sum(
                ids, derived.is_correction, n_groups, mask
            )
            features[f"market__{tag}__correction_volume_signed_logsum"] = signed_log1p(
                group_sum(ids, derived.correction_volume, n_groups, mask)
            )
            features[f"market__{tag}__correction_trade_count_signed_logsum"] = signed_log1p(
                group_sum(ids, derived.correction_count, n_groups, mask)
            )
            squared_returns = np.square(np.nan_to_num(mid_returns, nan=0.0))
            valid_return = transition_mask(ids, seconds, window) & np.isfinite(mid_returns)
            features[f"market__{tag}__realized_vol_bps"] = np.sqrt(
                group_sum(ids, squared_returns, n_groups, valid_return)
            )
        return feature_matrix(context.sample_ids, features)

    def transform(
        self,
        columns: Mapping[str, np.ndarray],
        context: FeatureContext,
    ) -> FeatureMatrix:
        derived = self.derive(columns, context.shard)
        return self.transform_derived(derived)


def _safe_imbalance(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    denominator = bid + ask
    result = np.zeros(len(bid), dtype=np.float64)
    np.divide(bid - ask, denominator, out=result, where=denominator > 0)
    return result
