from __future__ import annotations

import json
from dataclasses import asdict

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


def postprocess_run(
    config: ProjectConfig,
    source_run_id: str,
    *,
    power: float,
    center: bool = False,
) -> tuple[str, dict]:
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    source_dir = config.paths.artifacts_dir / "runs" / source_run_id
    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(encoding="utf-8")
    )
    source_oof = pq.read_table(source_dir / "oof.parquet")
    required = {"sample_id", "month", "target", "prediction", "fold"}
    missing = required - set(source_oof.column_names)
    if missing:
        raise ValueError(f"source OOF is missing columns: {sorted(missing)}")

    source_prediction = (
        source_oof["prediction"].to_numpy(zero_copy_only=False).astype(np.float64)
    )
    prediction = signed_power_transform(source_prediction, power=power, center=center)
    prediction = prediction.astype(np.float32)
    target = source_oof["target"].to_numpy(zero_copy_only=False).astype(np.float64)
    months = source_oof["month"].to_numpy(zero_copy_only=False)
    folds = source_oof["fold"].to_numpy(zero_copy_only=False)
    fold_reports = {
        str(fold): asdict(
            cosine_report(
                target[folds == fold], prediction[folds == fold], months[folds == fold]
            )
        )
        for fold in np.unique(folds)
    }
    metrics = {
        "source_run_id": source_run_id,
        "power": float(power),
        "center": center,
        "source_overall": asdict(cosine_report(target, source_prediction, months)),
        "overall": asdict(cosine_report(target, prediction, months)),
        "folds": fold_reports,
        "fold_mean": float(
            np.mean([report["global_score"] for report in fold_reports.values()])
        ),
    }
    payload = {
        "kind": "signed_power_postprocess",
        "source_run_id": source_run_id,
        "power": float(power),
        "center": center,
    }
    run_id = f"postprocess-{fingerprint(payload)}"
    run_dir = config.paths.artifacts_dir / "runs" / run_id
    output_oof = source_oof.set_column(
        source_oof.schema.get_field_index("prediction"),
        "prediction",
        pa.array(prediction, type=pa.float32()),
    )
    atomic_write_parquet(
        run_dir / "oof.parquet",
        output_oof,
        compression=config.cache.parquet_compression,
    )

    source_test_path = source_dir / "test_prediction.npy"
    test_path = None
    if source_test_path.is_file():
        source_test = np.load(source_test_path, allow_pickle=False)
        expected_shape = (config.dataset.test_sample_count,)
        if source_test.shape != expected_shape:
            raise ValueError(
                f"source test prediction has shape {source_test.shape}; "
                f"expected {expected_shape}"
            )
        processed_test = signed_power_transform(source_test, power=power, center=center)
        test_path = run_dir / "test_prediction.npy"
        atomic_save_npy(test_path, processed_test)

    contract = source_manifest.get("contract")
    manifest = {
        **payload,
        "run_id": run_id,
        "source_manifest": source_manifest,
        "oof_path": "oof.parquet",
        "metrics_path": "metrics.json",
        "test_prediction_path": test_path.name if test_path is not None else None,
    }
    if contract is not None:
        manifest["contract"] = contract
    atomic_write_json(run_dir / "metrics.json", metrics)
    atomic_write_json(run_dir / "manifest.json", manifest)
    return run_id, metrics


def signed_power_transform(
    prediction: np.ndarray,
    *,
    power: float,
    center: bool,
) -> np.ndarray:
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("prediction must be a finite one-dimensional vector")
    if not np.isfinite(power) or power <= 0:
        raise ValueError("power must be finite and positive")
    shifted = values - values.mean() if center else values.copy()
    scale = float(shifted.std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("prediction must have non-zero variance after centering")
    transformed = np.sign(shifted) * np.power(np.abs(shifted) / scale, power)
    if not np.isfinite(transformed).all() or np.linalg.norm(transformed) == 0:
        raise ValueError("postprocessed prediction must be finite and non-zero")
    return transformed
