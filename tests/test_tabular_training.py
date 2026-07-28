from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from mscapital.contracts import FeatureMatrix
from mscapital.contracts import Split, TableName
from mscapital.data.canonical import CanonicalStore
from mscapital.features.store import FeatureStore
from mscapital.training.tabular import (
    _clip_target,
    _add_derived_features,
    _exclude_features,
    _experiment_payload,
    _normalize_target,
    _recency_weights,
    _restrict_to_recent_months,
    read_metrics,
    train_oof,
)
from tests.helpers import temporary_config, write_synthetic_dataset


class TabularTrainingIntegrationTests(unittest.TestCase):
    def test_four_fold_oof_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = temporary_config(root, expected_samples=142)
            write_synthetic_dataset(config.paths.data_dir, 142, n_months=71)
            canonical = CanonicalStore(config)
            for table in (TableName.MARKET, TableName.ORDER, TableName.TRANSACTION):
                canonical.build(Split.TRAIN, table)
            FeatureStore(config, canonical).build(Split.TRAIN)
            run_id, metrics = train_oof(config)
            self.assertEqual(metrics["covered_rows"], 48)
            self.assertEqual(len(metrics["folds"]), 4)
            self.assertIn("passed", metrics["baseline_gate"])
            self.assertEqual(read_metrics(config, run_id)["run_id"], run_id)
            self.assertTrue(
                (config.paths.artifacts_dir / "runs" / run_id / "oof.parquet").is_file()
            )

    def test_objective_override_is_recorded_in_run_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = temporary_config(root, expected_samples=142)
            config = replace(
                config,
                lightgbm=replace(
                    config.lightgbm,
                    max_rounds=2,
                    early_stopping_rounds=1,
                ),
            )
            write_synthetic_dataset(config.paths.data_dir, 142, n_months=71)
            canonical = CanonicalStore(config)
            for table in (TableName.MARKET, TableName.ORDER, TableName.TRANSACTION):
                canonical.build(Split.TRAIN, table)
            FeatureStore(config, canonical).build(Split.TRAIN)
            default_run, _ = train_oof(config, seeds=(17,))
            huber_run, _ = train_oof(config, seeds=(17,), objective="huber")
            scaled_huber_run, _ = train_oof(
                config,
                seeds=(17,),
                objective="huber",
                objective_alpha=0.005,
            )
            self.assertNotEqual(default_run, huber_run)
            self.assertNotEqual(huber_run, scaled_huber_run)
            manifest_path = (
                config.paths.artifacts_dir / "runs" / huber_run / "manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model"]["objective"], "huber")
            scaled_manifest_path = (
                config.paths.artifacts_dir
                / "runs"
                / scaled_huber_run
                / "manifest.json"
            )
            scaled_manifest = json.loads(
                scaled_manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(scaled_manifest["model"]["alpha"], 0.005)

            with self.assertRaises(ValueError):
                train_oof(config, seeds=(17,), objective_alpha=0.005)


class TabularExperimentTests(unittest.TestCase):
    def test_experiment_payload_validates_settings(self) -> None:
        payload = _experiment_payload(("market__row_count",), 24, 12.0, (0.01, 0.99))
        self.assertEqual(payload["exclude_patterns"], ["market__row_count"])
        self.assertEqual(payload["target_clip_quantiles"], [0.01, 0.99])
        for recent_months, half_life, quantiles in (
            (0, None, None),
            (None, 0.0, None),
            (None, None, (0.9, 0.1)),
        ):
            with self.subTest(
                recent_months=recent_months,
                half_life=half_life,
                quantiles=quantiles,
            ):
                with self.assertRaises(ValueError):
                    _experiment_payload((), recent_months, half_life, quantiles)

    def test_feature_exclusion_uses_glob_patterns_and_preserves_order(self) -> None:
        matrix = FeatureMatrix(
            sample_ids=np.arange(3, dtype=np.int32),
            values=np.arange(12, dtype=np.float32).reshape(3, 4),
            names=(
                "market__row_count",
                "market__w10__mid_bps__std",
                "order__w1__event_rate",
                "transaction__w10__volume_imbalance",
            ),
        )
        filtered = _exclude_features(
            matrix, ("*__row_count", "order__w*__event_rate")
        )
        self.assertEqual(
            filtered.names,
            (
                "market__w10__mid_bps__std",
                "transaction__w10__volume_imbalance",
            ),
        )
        np.testing.assert_array_equal(filtered.values, matrix.values[:, (1, 3)])

    def test_recent_months_target_clipping_and_weights_are_fold_local(self) -> None:
        months = np.asarray([0, 1, 2, 3, 4], dtype=np.int16)
        indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
        selected = _restrict_to_recent_months(indices, months, 2)
        np.testing.assert_array_equal(selected, [2, 3])

        target = np.asarray([-100.0, -1.0, 1.0, 100.0], dtype=np.float32)
        clipped, bounds = _clip_target(target, (0.25, 0.75))
        np.testing.assert_allclose(clipped, [-25.75, -1.0, 1.0, 25.75])
        self.assertEqual(bounds, (-25.75, 25.75))

        weights = _recency_weights(months[:4], 1.0)
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertTrue(np.all(np.diff(weights) > 0))
        np.testing.assert_allclose(_recency_weights(months[:4], None), 1.0)

    def test_monthly_target_normalization_is_group_local(self) -> None:
        target = np.asarray([1.0, 3.0, 10.0, 14.0], dtype=np.float32)
        months = np.asarray([0, 0, 1, 1], dtype=np.int16)
        standardized, report = _normalize_target(target, months, "monthly_zscore")
        np.testing.assert_allclose(standardized, [-1.0, 1.0, -1.0, 1.0])
        self.assertEqual(report["method"], "monthly_zscore")
        self.assertEqual(report["scale_min"], 1.0)
        self.assertEqual(report["scale_max"], 2.0)

        scaled, _ = _normalize_target(target, months, "monthly_std")
        np.testing.assert_allclose(scaled, [1.0, 3.0, 5.0, 7.0])
        with self.assertRaises(ValueError):
            _normalize_target(target, months, "future_month")

    def test_order_category_ratio_features(self) -> None:
        names: list[str] = []
        columns: list[np.ndarray] = []
        for window in (1, 2, 5, 10, 30, 60):
            prefix = f"order__w{window}"
            names.extend((f"{prefix}__event_count", f"{prefix}__volume_logsum"))
            columns.extend(
                (
                    np.asarray([4.0, 0.0], dtype=np.float32),
                    np.log1p(np.asarray([100.0, 0.0], dtype=np.float32)),
                )
            )
            for index, category in enumerate(
                ("buy_new", "buy_cancel", "sell_new", "sell_cancel")
            ):
                names.extend(
                    (
                        f"{prefix}__{category}_count",
                        f"{prefix}__{category}_volume_logsum",
                    )
                )
                columns.extend(
                    (
                        np.asarray([1.0, 0.0], dtype=np.float32),
                        np.log1p(
                            np.asarray([10.0 * (index + 1), 0.0], dtype=np.float32)
                        ),
                    )
                )
        matrix = FeatureMatrix(
            np.asarray([0, 1], dtype=np.int32),
            np.column_stack(columns),
            tuple(names),
        )
        augmented = _add_derived_features(matrix, ("order_category_ratios",))
        self.assertEqual(augmented.values.shape[1], matrix.values.shape[1] + 48)
        result = dict(zip(augmented.names, augmented.values.T))
        np.testing.assert_allclose(result["order__w1__buy_new_count_share"], [0.25, 0.0])
        np.testing.assert_allclose(result["order__w1__sell_cancel_volume_share"], [0.4, 0.0])


if __name__ == "__main__":
    unittest.main()
