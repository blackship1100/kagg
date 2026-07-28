from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from mscapital.config import ProjectConfig


def make_submission(
    config: ProjectConfig,
    run_id: str,
    output: str | Path | None = None,
    *,
    resume: bool = False,
) -> Path:
    run_dir = config.paths.artifacts_dir / "runs" / run_id
    prediction = np.load(run_dir / "test_prediction.npy", allow_pickle=False)
    template = pd.read_csv(config.paths.data_dir / "submission.csv")
    if list(template.columns) != ["sample_id", "prediction"]:
        raise ValueError("submission template must contain sample_id and prediction")
    if len(template) != config.dataset.test_sample_count:
        raise ValueError(
            f"submission template has {len(template)} rows; "
            f"expected {config.dataset.test_sample_count}"
        )
    expected_ids = np.arange(len(template), dtype=template["sample_id"].to_numpy().dtype)
    if not np.array_equal(template["sample_id"].to_numpy(), expected_ids):
        raise ValueError("submission sample_id must be contiguous and ordered")
    if prediction.shape != (len(template),):
        raise ValueError(
            f"prediction has {len(prediction)} rows; expected {len(template)} rows"
        )
    if not np.isfinite(prediction).all() or np.linalg.norm(prediction) == 0:
        raise ValueError("prediction must be finite and have non-zero norm")
    template["prediction"] = prediction
    target = (
        Path(output)
        if output is not None
        else config.paths.artifacts_dir / "submissions" / f"{run_id}.csv"
    )
    if resume and target.is_file():
        existing = pd.read_csv(target)
        if (
            list(existing.columns) == ["sample_id", "prediction"]
            and np.array_equal(existing["sample_id"].to_numpy(), expected_ids)
            and np.allclose(
                existing["prediction"].to_numpy(), prediction, rtol=1e-12, atol=1e-15
            )
        ):
            return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    template.to_csv(temp, index=False)
    os.replace(temp, target)
    return target
