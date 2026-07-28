from __future__ import annotations

import unittest

import numpy as np

from mscapital.config import ProjectConfig
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


if __name__ == "__main__":
    unittest.main()

