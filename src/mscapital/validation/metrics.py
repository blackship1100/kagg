from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CosineReport:
    global_score: float
    monthly_scores: dict[int, float]
    monthly_mean: float
    monthly_std: float
    worst_month: int
    worst_score: float


def cosine_score(target: ArrayLike, prediction: ArrayLike) -> float:
    truth = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("target and prediction must be matching one-dimensional arrays")
    if not np.isfinite(truth).all() or not np.isfinite(predicted).all():
        raise ValueError("target and prediction must contain only finite values")
    truth_norm = float(np.linalg.norm(truth))
    prediction_norm = float(np.linalg.norm(predicted))
    if truth_norm == 0 or prediction_norm == 0:
        raise ValueError("cosine score is undefined for a zero-norm vector")
    return float(np.dot(truth, predicted) / (truth_norm * prediction_norm))


def cosine_report(target: ArrayLike, prediction: ArrayLike, months: ArrayLike) -> CosineReport:
    truth = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    month_values = np.asarray(months)
    if truth.shape != predicted.shape or truth.shape != month_values.shape:
        raise ValueError("target, prediction, and months must have matching shapes")
    global_score = cosine_score(truth, predicted)
    monthly = {
        int(month): cosine_score(truth[month_values == month], predicted[month_values == month])
        for month in np.unique(month_values)
    }
    scores = np.asarray(list(monthly.values()), dtype=np.float64)
    worst_month = min(monthly, key=monthly.get)
    return CosineReport(
        global_score=global_score,
        monthly_scores=monthly,
        monthly_mean=float(scores.mean()),
        monthly_std=float(scores.std()),
        worst_month=int(worst_month),
        worst_score=float(monthly[worst_month]),
    )
