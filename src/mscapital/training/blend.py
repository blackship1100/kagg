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
from mscapital.validation.metrics import cosine_report


def blend_runs(
    config: ProjectConfig,
    run_ids: tuple[str, ...],
    *,
    weights: tuple[float, ...] | None = None,
    normalize: bool = True,
) -> tuple[str, dict]:
    if len(run_ids) < 2 or len(set(run_ids)) != len(run_ids):
        raise ValueError("blend requires at least two distinct runs")
    run_dirs = [config.paths.artifacts_dir / "runs" / run_id for run_id in run_ids]
    tables = [pq.read_table(directory / "oof.parquet") for directory in run_dirs]
    reference = tables[0]
    _validate_oof_alignment(tables)
    normalized_weights = _validated_weights(weights, len(run_ids))
    predictions = [
        table["prediction"].to_numpy(zero_copy_only=False).astype(np.float64)
        for table in tables
    ]
    prediction = _combine(predictions, normalized_weights, normalize, "OOF")
    target = reference["target"].to_numpy(zero_copy_only=False).astype(np.float64)
    months = reference["month"].to_numpy(zero_copy_only=False)
    folds = reference["fold"].to_numpy(zero_copy_only=False)
    fold_reports = {
        str(fold): asdict(
            cosine_report(
                target[folds == fold], prediction[folds == fold], months[folds == fold]
            )
        )
        for fold in np.unique(folds)
    }
    payload = {
        "kind": "run_blend",
        "component_runs": list(run_ids),
        "weights": normalized_weights.tolist(),
        "normalize": normalize,
    }
    run_id = f"blend-{fingerprint(payload)}"
    run_dir = config.paths.artifacts_dir / "runs" / run_id
    output_table = reference.set_column(
        reference.schema.get_field_index("prediction"),
        "prediction",
        pa.array(prediction.astype(np.float32), type=pa.float32()),
    )
    atomic_write_parquet(
        run_dir / "oof.parquet",
        output_table,
        compression=config.cache.parquet_compression,
    )
    test_path = _blend_test_predictions(
        config, run_dirs, run_dir, normalized_weights, normalize
    )
    metrics = {
        "overall": asdict(cosine_report(target, prediction, months)),
        "folds": fold_reports,
        "fold_mean": float(
            np.mean([report["global_score"] for report in fold_reports.values()])
        ),
        "component_runs": list(run_ids),
        "weights": normalized_weights.tolist(),
        "normalize": normalize,
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


def _validate_oof_alignment(tables: list[pa.Table]) -> None:
    required = ("sample_id", "month", "target", "prediction", "fold")
    for table in tables:
        missing = set(required) - set(table.column_names)
        if missing:
            raise ValueError(f"blend OOF is missing columns: {sorted(missing)}")
    reference = tables[0]
    for table in tables[1:]:
        for column in ("sample_id", "month", "target", "fold"):
            if not table[column].equals(reference[column]):
                raise ValueError(f"blend OOF column mismatch: {column}")


def _validated_weights(
    weights: tuple[float, ...] | None, component_count: int
) -> np.ndarray:
    values = (
        np.ones(component_count, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if (
        values.shape != (component_count,)
        or not np.isfinite(values).all()
        or np.any(values <= 0)
    ):
        raise ValueError("blend weights must be finite, positive, and match the runs")
    return values / values.sum()


def _combine(
    predictions: list[np.ndarray],
    weights: np.ndarray,
    normalize: bool,
    label: str,
) -> np.ndarray:
    values = []
    for index, prediction in enumerate(predictions):
        vector = np.asarray(prediction, dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError(f"{label} prediction {index} must be a finite vector")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError(f"{label} prediction {index} must have non-zero norm")
        values.append(vector / norm if normalize else vector)
    combined = np.average(np.vstack(values), axis=0, weights=weights)
    if not np.isfinite(combined).all() or np.linalg.norm(combined) == 0:
        raise ValueError(f"blended {label} prediction must be finite and non-zero")
    return combined


def _blend_test_predictions(
    config: ProjectConfig,
    component_dirs: list[Path],
    output_dir: Path,
    weights: np.ndarray,
    normalize: bool,
) -> Path | None:
    paths = [directory / "test_prediction.npy" for directory in component_dirs]
    if not any(path.is_file() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(
            "all component test predictions are required for a blend"
        )
    predictions = [np.load(path, allow_pickle=False) for path in paths]
    expected_shape = (config.dataset.test_sample_count,)
    if any(prediction.shape != expected_shape for prediction in predictions):
        raise ValueError("component test prediction shapes do not match the dataset")
    combined = _combine(predictions, weights, normalize, "test")
    path = output_dir / "test_prediction.npy"
    atomic_save_npy(path, combined)
    return path
