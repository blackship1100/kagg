from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mscapital.artifacts import atomic_save_npy, atomic_write_json


PREPROCESSOR_VERSION = 1


@dataclass(frozen=True)
class FoldPreprocessor:
    feature_names: tuple[str, ...]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    missing_fraction: np.ndarray
    quantiles: tuple[float, float]

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        feature_names: tuple[str, ...],
        quantiles: tuple[float, float],
    ) -> "FoldPreprocessor":
        matrix = np.asarray(values)
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
            raise ValueError("values and feature_names do not have compatible shapes")
        lower_q, upper_q = quantiles
        if not 0.0 <= lower_q < upper_q <= 1.0:
            raise ValueError("feature clip quantiles must satisfy 0 <= lower < upper <= 1")
        lower = np.full(matrix.shape[1], np.nan, dtype=np.float32)
        upper = np.full(matrix.shape[1], np.nan, dtype=np.float32)
        missing = np.mean(~np.isfinite(matrix), axis=0, dtype=np.float64).astype(np.float32)
        for index in range(matrix.shape[1]):
            finite = matrix[:, index]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                bounds = np.quantile(finite, (lower_q, upper_q))
                lower[index], upper[index] = bounds
        return cls(feature_names, lower, upper, missing, quantiles)

    def transform(self, values: np.ndarray, *, copy: bool = True) -> np.ndarray:
        matrix = np.array(values, dtype=np.float32, copy=copy)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError("values do not match the fitted feature schema")
        for index, (lower, upper) in enumerate(zip(self.lower_bounds, self.upper_bounds)):
            if np.isfinite(lower) and np.isfinite(upper):
                np.clip(matrix[:, index], lower, upper, out=matrix[:, index])
        return matrix

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        atomic_save_npy(target / "lower_bounds.npy", self.lower_bounds)
        atomic_save_npy(target / "upper_bounds.npy", self.upper_bounds)
        atomic_save_npy(target / "missing_fraction.npy", self.missing_fraction)
        atomic_write_json(
            target / "manifest.json",
            {
                "version": PREPROCESSOR_VERSION,
                "feature_names": list(self.feature_names),
                "quantiles": list(self.quantiles),
                "feature_count": len(self.feature_names),
            },
        )

    @classmethod
    def load(cls, directory: str | Path) -> "FoldPreprocessor":
        source = Path(directory)
        raw = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
        if raw["version"] != PREPROCESSOR_VERSION:
            raise ValueError("unsupported fold preprocessor version")
        names = tuple(raw["feature_names"])
        lower = np.load(source / "lower_bounds.npy", allow_pickle=False)
        upper = np.load(source / "upper_bounds.npy", allow_pickle=False)
        missing = np.load(source / "missing_fraction.npy", allow_pickle=False)
        expected = (len(names),)
        if lower.shape != expected or upper.shape != expected or missing.shape != expected:
            raise ValueError("saved fold preprocessor arrays have invalid shapes")
        return cls(names, lower, upper, missing, tuple(raw["quantiles"]))
