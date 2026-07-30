from __future__ import annotations

import unittest

import numpy as np

from mscapital.config import ProjectConfig
from mscapital.contracts import FeatureMatrix
from mscapital.training.tabular import _baseline_gate
from mscapital.validation.baselines import evaluate_baselines
from mscapital.validation.splits import MonthFold, folds_from_config


class ValidationTests(unittest.TestCase):
    def test_month_fold_has_no_future_leakage(self) -> None:
        fold = MonthFold("example", train_start=0, train_end=2, valid_start=3, valid_end=4)
        months = np.array([0, 1, 2, 3, 4, 5], dtype=np.int16)
        train, valid = fold.split(months)
        np.testing.assert_array_equal(train, [0, 1, 2])
        np.testing.assert_array_equal(valid, [3, 4])
        self.assertLess(months[train].max(), months[valid].min())

    def test_overlapping_month_ranges_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MonthFold("bad", train_start=0, train_end=3, valid_start=3, valid_end=4)

    def test_base_config_builds_four_folds(self) -> None:
        config = ProjectConfig.from_toml("configs/base.toml")
        folds = folds_from_config(config.folds)
        self.assertEqual([fold.name for fold in folds], ["fold_1", "fold_2", "fold_3", "fold_4"])
        self.assertEqual(folds[-1].valid_end, 70)

    def test_partial_feature_matrix_only_evaluates_available_baselines(self) -> None:
        matrix = FeatureMatrix(
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([[0.1], [-0.1]], dtype=np.float32),
            ("market__w60__mid_bps__delta",),
        )
        report = evaluate_baselines(
            matrix,
            np.asarray([0.2, -0.2], dtype=np.float32),
            np.asarray([1, 1], dtype=np.int16),
        )
        self.assertEqual(tuple(report), ("market_momentum_60",))

        no_baseline_gate = _baseline_gate(
            {"fold_1": {"global_score": 0.1}}, {"fold_1": {}}
        )
        self.assertIsNone(no_baseline_gate["passed"])


if __name__ == "__main__":
    unittest.main()
