from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from mscapital.contracts import FeatureMatrix, TableName
from mscapital.features.aggregation import feature_matrix
from mscapital.features.base import FeatureContext


class CrossFeatureBlock:
    name = "cross"
    version = 1
    required_tables = (TableName.MARKET, TableName.ORDER, TableName.TRANSACTION)

    def transform(
        self,
        blocks: Mapping[str, FeatureMatrix],
        context: FeatureContext,
    ) -> FeatureMatrix:
        market = _columns(blocks["market"])
        order = _columns(blocks["order"])
        transaction = _columns(blocks["transaction"])
        features: dict[str, np.ndarray] = {}
        for window in (1, 2, 5, 10, 30, 60):
            market_window = 5 if window < 5 else window
            order_flow = order[f"order__w{window}__signed_volume_imbalance"]
            trade_flow = transaction[f"transaction__w{window}__volume_imbalance"]
            order_volume = np.expm1(order[f"order__w{window}__volume_logsum"])
            trade_volume = np.expm1(transaction[f"transaction__w{window}__volume_logsum"])
            depth = np.expm1(market[f"market__w{market_window}__log_depth_l1__last"])
            price_delta = market[f"market__w{market_window}__mid_bps__delta"]
            microprice = market[f"market__w{market_window}__microprice_bps__last"]
            trade_price = transaction[f"transaction__w{window}__price_bps__last"]
            cancel_rate = order[f"order__w{window}__cancel_volume_rate"]

            tag = f"cross__w{window}"
            features[f"{tag}__flow_alignment"] = order_flow * trade_flow
            features[f"{tag}__flow_divergence"] = order_flow - trade_flow
            features[f"{tag}__order_volume_per_depth"] = _safe_ratio(order_volume, depth)
            features[f"{tag}__trade_volume_per_depth"] = _safe_ratio(trade_volume, depth)
            features[f"{tag}__price_response_per_trade"] = _safe_ratio(
                price_delta, np.log1p(trade_volume)
            )
            features[f"{tag}__microprice_trade_divergence"] = microprice - trade_price
            features[f"{tag}__cancel_trade_divergence"] = cancel_rate - np.abs(trade_flow)
        return feature_matrix(context.sample_ids, features)


def _columns(matrix: FeatureMatrix) -> dict[str, np.ndarray]:
    return {name: matrix.values[:, index] for index, name in enumerate(matrix.names)}


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros_like(np.asarray(numerator, dtype=np.float64))
    np.divide(numerator, denominator, out=result, where=np.abs(denominator) > 1e-12)
    return result
