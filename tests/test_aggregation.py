from __future__ import annotations

import unittest

import numpy as np

from mscapital.features.aggregation import (
    aggregate_series,
    fill_grouped_forward_backward,
    group_weighted_mean,
    transition_mask,
)


class AggregationTests(unittest.TestCase):
    def test_aggregate_series_handles_missing_groups_and_slope(self) -> None:
        result = aggregate_series(
            "x",
            group_ids=[0, 0, 2, 2],
            values=[1.0, 3.0, 10.0, 6.0],
            n_groups=3,
            seconds=[2.0, 1.0, 2.0, 1.0],
            stats=("first", "last", "mean", "std", "min", "max", "delta", "slope"),
        )
        self.assertAlmostEqual(result["x__mean"][0], 2.0)
        self.assertTrue(np.isnan(result["x__mean"][1]))
        self.assertAlmostEqual(result["x__delta"][2], -4.0)
        self.assertAlmostEqual(result["x__slope"][0], 2.0)

    def test_grouped_fill_does_not_cross_sample_boundaries(self) -> None:
        filled = fill_grouped_forward_backward(
            [np.nan, 2.0, np.nan, np.nan, 5.0],
            [0, 0, 0, 1, 1],
            [9.0, 8.0],
        )
        np.testing.assert_allclose(filled, [2.0, 2.0, 2.0, 5.0, 5.0])

    def test_weighted_mean(self) -> None:
        result = group_weighted_mean([0, 0, 1], [1.0, 3.0, 5.0], [1.0, 3.0, 2.0], 2)
        np.testing.assert_allclose(result, [2.5, 5.0])

    def test_window_transition_excludes_the_outside_boundary(self) -> None:
        selected = transition_mask([0, 0, 0, 1, 1], [6.0, 5.0, 0.0, 5.0, 0.0], 5)
        np.testing.assert_array_equal(selected, [False, False, True, False, True])


if __name__ == "__main__":
    unittest.main()
