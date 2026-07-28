from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mscapital.artifacts import (
    atomic_save_npy,
    atomic_write_json,
    atomic_write_parquet,
    fingerprint,
)
from mscapital.config import ProjectConfig
from mscapital.validation.metrics import cosine_report, cosine_score


def blend_tabular_deep(
    config: ProjectConfig,
    tabular_run_id: str,
    deep_run_id: str,
    *,
    deep_weight: float | None = None,
) -> tuple[str, dict]:
    tabular_dir = config.paths.artifacts_dir / "runs" / tabular_run_id
    deep_dir = config.paths.artifacts_dir / "runs" / deep_run_id
    tabular = pq.read_table(tabular_dir / "oof.parquet")
    deep = pq.read_table(deep_dir / "oof.parquet")
    tabular_indices = _align_oof(tabular, deep)
    target = deep["target"].to_numpy(zero_copy_only=False).astype(np.float64)
    months = deep["month"].to_numpy(zero_copy_only=False)
    tabular_prediction = (
        tabular["prediction"]
        .to_numpy(zero_copy_only=False)[tabular_indices]
        .astype(np.float64)
    )
    deep_prediction = (
        deep["prediction"].to_numpy(zero_copy_only=False).astype(np.float64)
    )
    tabular_unit = _unit_norm(tabular_prediction, "tabular OOF")
    deep_unit = _unit_norm(deep_prediction, "deep OOF")
    if deep_weight is None:
        candidates = np.linspace(0.0, 1.0, 101)
        scores = [
            cosine_score(target, (1.0 - weight) * tabular_unit + weight * deep_unit)
            for weight in candidates
        ]
        deep_weight = float(candidates[int(np.argmax(scores))])
    if not 0.0 <= deep_weight <= 1.0:
        raise ValueError("deep_weight must be between zero and one")
    prediction = ((1.0 - deep_weight) * tabular_unit + deep_weight * deep_unit).astype(
        np.float32
    )
    report = asdict(cosine_report(target, prediction, months))
    payload = {
        "kind": "deep_tabular_oof_blend",
        "tabular_run_id": tabular_run_id,
        "deep_run_id": deep_run_id,
        "deep_weight": deep_weight,
        "normalization": "per_component_l2",
    }
    run_id = f"blend-{fingerprint(payload)}"
    run_dir = config.paths.artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(
        run_dir / "oof.parquet",
        pa.table(
            {
                "sample_id": deep["sample_id"],
                "month": deep["month"],
                "target": deep["target"],
                "prediction": pa.array(prediction, type=pa.float32()),
                "fold": deep["fold"],
            }
        ),
        compression=config.cache.parquet_compression,
    )
    test_path = _blend_test(config, tabular_dir, deep_dir, run_dir, deep_weight)
    metrics = {
        "run_id": run_id,
        "overall": report,
        "tabular_oof_cosine": cosine_score(target, tabular_prediction),
        "deep_oof_cosine": cosine_score(target, deep_prediction),
        "deep_weight": deep_weight,
        "covered_rows": len(target),
    }
    atomic_write_json(run_dir / "metrics.json", metrics)
    atomic_write_json(
        run_dir / "manifest.json",
        {
            **payload,
            "run_id": run_id,
            "oof_path": "oof.parquet",
            "metrics_path": "metrics.json",
            "test_prediction_path": test_path.name if test_path is not None else None,
        },
    )
    return run_id, metrics


def _align_oof(tabular: pa.Table, deep: pa.Table) -> np.ndarray:
    tabular_ids = tabular["sample_id"].to_numpy(zero_copy_only=False)
    deep_ids = deep["sample_id"].to_numpy(zero_copy_only=False)
    if len(np.unique(tabular_ids)) != len(tabular_ids) or len(
        np.unique(deep_ids)
    ) != len(deep_ids):
        raise ValueError("OOF sample_id must be unique")
    positions = np.searchsorted(tabular_ids, deep_ids)
    if np.any(positions >= len(tabular_ids)) or not np.array_equal(
        tabular_ids[positions], deep_ids
    ):
        raise ValueError("deep OOF rows are not covered by the tabular OOF run")
    for name in ("month", "target", "fold"):
        tabular_values = tabular[name].to_numpy(zero_copy_only=False)[positions]
        deep_values = deep[name].to_numpy(zero_copy_only=False)
        if not np.array_equal(tabular_values, deep_values):
            raise ValueError(f"OOF alignment mismatch: {name}")
    return positions


def _blend_test(
    config: ProjectConfig,
    tabular_dir: Path,
    deep_dir: Path,
    output_dir: Path,
    deep_weight: float,
) -> Path | None:
    tabular_path = tabular_dir / "test_prediction.npy"
    deep_path = deep_dir / "test_prediction.npy"
    if not tabular_path.is_file() and not deep_path.is_file():
        return None
    if not tabular_path.is_file() or not deep_path.is_file():
        raise FileNotFoundError("both tabular and deep test predictions are required")
    tabular = np.load(tabular_path, allow_pickle=False).astype(np.float64)
    deep = np.load(deep_path, allow_pickle=False).astype(np.float64)
    expected = (config.dataset.test_sample_count,)
    if tabular.shape != expected or deep.shape != expected:
        raise ValueError("test prediction row count mismatch")
    prediction = (
        (1.0 - deep_weight) * _unit_norm(tabular, "tabular test")
        + deep_weight * _unit_norm(deep, "deep test")
    ).astype(np.float32)
    path = output_dir / "test_prediction.npy"
    atomic_save_npy(path, prediction)
    return path


def _unit_norm(values: np.ndarray, name: str) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")
    norm = np.linalg.norm(values)
    if norm == 0:
        raise ValueError(f"{name} has zero norm")
    return values / norm
