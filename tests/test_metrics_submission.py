from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mscapital.training.ensemble import ensemble_runs
from mscapital.training.submission import make_submission
from mscapital.validation.metrics import cosine_report, cosine_score
from tests.helpers import temporary_config, write_synthetic_dataset


class MetricsAndSubmissionTests(unittest.TestCase):
    def test_cosine_and_month_report(self) -> None:
        target = np.array([1.0, -1.0, 2.0, -2.0])
        report = cosine_report(target, target * 3.0, [0, 0, 1, 1])
        self.assertAlmostEqual(report.global_score, 1.0)
        self.assertAlmostEqual(report.monthly_scores[0], 1.0)

    def test_cosine_rejects_zero_norm(self) -> None:
        with self.assertRaises(ValueError):
            cosine_score([1.0, -1.0], [0.0, 0.0])

    def test_submission_is_validated_and_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = temporary_config(root)
            write_synthetic_dataset(config.paths.data_dir)
            run_id = "test-run"
            run_dir = config.paths.artifacts_dir / "runs" / run_id
            run_dir.mkdir(parents=True)
            np.save(run_dir / "test_prediction.npy", np.array([1.0, -1.0, 0.5, -0.5]))
            output = make_submission(config, run_id)
            self.assertTrue(output.is_file())
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 5)

    def test_compatible_runs_are_ensembled_by_seed_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = temporary_config(root)
            contract = {
                "model": {"objective": "regression_l2"},
                "preprocessing": {"version": 1},
                "experiment": {},
                "folds": [],
                "blocks": ["market"],
                "features": {"market": "digest"},
                "feature_names": ["feature"],
            }
            for run_id, seeds, prediction in (
                ("one", [17], np.asarray([1.0, 2.0, 3.0, 4.0])),
                ("two", [43, 97], np.asarray([4.0, 3.0, 2.0, 1.0])),
            ):
                run_dir = config.paths.artifacts_dir / "runs" / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "manifest.json").write_text(
                    json.dumps({**contract, "runtime": {"seeds": seeds}}),
                    encoding="utf-8",
                )
                pq.write_table(
                    pa.table(
                        {
                            "sample_id": pa.array([0, 1, 2, 3], type=pa.int32()),
                            "month": pa.array([0, 0, 1, 1], type=pa.int16()),
                            "target": pa.array([1.0, 1.0, -1.0, -1.0], type=pa.float32()),
                            "prediction": pa.array(prediction, type=pa.float32()),
                            "fold": pa.array(["fold_1"] * 4),
                        }
                    ),
                    run_dir / "oof.parquet",
                )
                np.save(run_dir / "test_prediction.npy", prediction)
            ensemble_id, metrics = ensemble_runs(config, ("one", "two"))
            combined = np.load(
                config.paths.artifacts_dir / "runs" / ensemble_id / "test_prediction.npy"
            )
            np.testing.assert_allclose(combined, [3.0, 8.0 / 3.0, 7.0 / 3.0, 2.0])
            self.assertEqual(metrics["weights"], [1.0 / 3.0, 2.0 / 3.0])


if __name__ == "__main__":
    unittest.main()
