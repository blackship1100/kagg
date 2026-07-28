from __future__ import annotations

import json
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
from mscapital.validation.metrics import cosine_report


_CONTRACT_KEYS = (
    "model",
    "preprocessing",
    "experiment",
    "folds",
    "blocks",
    "features",
    "feature_names",
)


def ensemble_runs(
    config: ProjectConfig,
    run_ids: tuple[str, ...],
    *,
    weights: tuple[float, ...] | None = None,
) -> tuple[str, dict]:
    if len(run_ids) < 2 or len(set(run_ids)) != len(run_ids):
        raise ValueError("ensemble requires at least two distinct runs")
    run_dirs = [config.paths.artifacts_dir / "runs" / run_id for run_id in run_ids]
    manifests = [
        json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
    ]
    _validate_contracts(manifests)
    if weights is None:
        weights = tuple(float(len(item["runtime"]["seeds"])) for item in manifests)
    normalized_weights = _validate_weights(weights, len(run_ids))

    tables = [pq.read_table(run_dir / "oof.parquet") for run_dir in run_dirs]
    reference = tables[0]
    for table in tables[1:]:
        for column in ("sample_id", "month", "target", "fold"):
            if not table[column].equals(reference[column]):
                raise ValueError(f"ensemble OOF column mismatch: {column}")
    predictions = [
        table["prediction"].to_numpy(zero_copy_only=False).astype(np.float64)
        for table in tables
    ]
    prediction = np.average(np.vstack(predictions), axis=0, weights=normalized_weights)
    target = reference["target"].to_numpy(zero_copy_only=False).astype(np.float64)
    months = reference["month"].to_numpy(zero_copy_only=False)
    folds = reference["fold"].to_numpy(zero_copy_only=False)
    fold_reports = {
        str(fold): asdict(cosine_report(target[folds == fold], prediction[folds == fold], months[folds == fold]))
        for fold in np.unique(folds)
    }
    metrics = {
        "overall": asdict(cosine_report(target, prediction, months)),
        "folds": fold_reports,
        "fold_mean": float(
            np.mean([report["global_score"] for report in fold_reports.values()])
        ),
        "component_runs": list(run_ids),
        "weights": normalized_weights.tolist(),
    }
    payload = {
        "kind": "weighted_run_ensemble",
        "component_runs": list(run_ids),
        "weights": normalized_weights.tolist(),
        "contract": {key: manifests[0][key] for key in _CONTRACT_KEYS},
    }
    ensemble_id = f"ensemble-{fingerprint(payload)}"
    run_dir = config.paths.artifacts_dir / "runs" / ensemble_id
    oof_table = pa.table(
        {
            "sample_id": reference["sample_id"],
            "month": reference["month"],
            "target": reference["target"],
            "prediction": pa.array(prediction.astype(np.float32), type=pa.float32()),
            "fold": reference["fold"],
        }
    )
    atomic_write_parquet(
        run_dir / "oof.parquet",
        oof_table,
        compression=config.cache.parquet_compression,
    )
    test_path = _ensemble_test_predictions(config, run_dirs, run_dir, normalized_weights)
    manifest = {
        **payload,
        "run_id": ensemble_id,
        "oof_path": "oof.parquet",
        "metrics_path": "metrics.json",
        "test_prediction_path": test_path.name if test_path is not None else None,
    }
    atomic_write_json(run_dir / "metrics.json", metrics)
    atomic_write_json(run_dir / "manifest.json", manifest)
    return ensemble_id, metrics


def _validate_contracts(manifests: list[dict]) -> None:
    reference = manifests[0]
    for manifest in manifests[1:]:
        mismatched = [key for key in _CONTRACT_KEYS if manifest.get(key) != reference.get(key)]
        if mismatched:
            raise ValueError(f"ensemble run contracts differ: {mismatched}")


def _validate_weights(weights: tuple[float, ...], expected: int) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    if values.shape != (expected,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("ensemble weights must be finite, positive, and match the runs")
    return values / values.sum()


def _ensemble_test_predictions(
    config: ProjectConfig,
    component_dirs: list[Path],
    ensemble_dir: Path,
    weights: np.ndarray,
) -> Path | None:
    paths = [directory / "test_prediction.npy" for directory in component_dirs]
    if not any(path.is_file() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError("all component test predictions are required for an ensemble")
    predictions = [np.load(path, allow_pickle=False) for path in paths]
    expected_shape = (config.dataset.test_sample_count,)
    if any(values.shape != expected_shape for values in predictions):
        raise ValueError("component test prediction shapes do not match the dataset")
    combined = np.average(np.vstack(predictions), axis=0, weights=weights)
    if not np.isfinite(combined).all() or np.linalg.norm(combined) == 0:
        raise ValueError("ensemble test prediction must be finite and have non-zero norm")
    output = ensemble_dir / "test_prediction.npy"
    atomic_save_npy(output, combined)
    return output
