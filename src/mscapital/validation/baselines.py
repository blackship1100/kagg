from __future__ import annotations

from dataclasses import asdict

import numpy as np

from mscapital.contracts import FeatureMatrix
from mscapital.validation.metrics import cosine_report


BASELINE_FEATURES = {
    "market_momentum_60": "market__w60__mid_bps__delta",
    "microprice_deviation_last": "market__w5__microprice_bps__last",
    "trade_imbalance_10": "transaction__w10__volume_imbalance",
}


def evaluate_baselines(
    matrix: FeatureMatrix,
    target: np.ndarray,
    months: np.ndarray,
) -> dict[str, dict]:
    index = {name: position for position, name in enumerate(matrix.names)}
    results = {}
    for baseline, feature_name in BASELINE_FEATURES.items():
        if feature_name not in index:
            raise KeyError(f"baseline feature is missing: {feature_name}")
        prediction = np.nan_to_num(matrix.values[:, index[feature_name]], nan=0.0)
        if np.linalg.norm(prediction) == 0:
            results[baseline] = {"error": "zero-norm prediction"}
            continue
        results[baseline] = asdict(cosine_report(target, prediction, months))
    return results
