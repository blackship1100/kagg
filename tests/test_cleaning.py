from __future__ import annotations

import unittest

import numpy as np

from mscapital.data.cleaning import (
    DataQualityError,
    classify_book_level,
    classify_market_trades,
    clip_with_mask,
    compute_midprice,
    signed_log1p,
)


class CleaningTests(unittest.TestCase):
    def test_book_level_separates_valid_and_missing_rows(self) -> None:
        state = classify_book_level([1.01, 0.0], [100, 0])
        np.testing.assert_array_equal(state.valid, [True, False])
        np.testing.assert_array_equal(state.missing, [False, True])
        self.assertFalse(state.inconsistent.any())

    def test_book_level_rejects_mismatched_zero_sentinel(self) -> None:
        with self.assertRaises(DataQualityError):
            classify_book_level([0.0], [100])

    def test_market_trade_states_cover_positive_missing_and_correction(self) -> None:
        state = classify_market_trades(
            avgprice=[1.0, np.nan, 0.99],
            volume=[100, 0, -400],
            count=[2, 0, -1],
        )
        np.testing.assert_array_equal(state.positive_trade, [True, False, False])
        np.testing.assert_array_equal(state.no_trade, [False, True, False])
        np.testing.assert_array_equal(state.correction, [False, False, True])
        self.assertFalse(state.inconsistent.any())

    def test_market_trade_rejects_zero_volume_with_price(self) -> None:
        with self.assertRaises(DataQualityError):
            classify_market_trades([1.0], [0], [0])

    def test_midprice_preserves_missing_rows(self) -> None:
        result = compute_midprice([1.02, 0.0], [0.98, 0.97])
        self.assertAlmostEqual(result[0], 1.0)
        self.assertTrue(np.isnan(result[1]))

    def test_signed_log_handles_corrections(self) -> None:
        values = signed_log1p([-9, 0, 9])
        np.testing.assert_allclose(values, [-np.log(10), 0.0, np.log(10)])

    def test_clip_returns_an_explicit_mask(self) -> None:
        result = clip_with_mask([-2.0, 0.5, 3.0], -1.0, 1.0)
        np.testing.assert_allclose(result.values, [-1.0, 0.5, 1.0])
        np.testing.assert_array_equal(result.was_clipped, [True, False, True])


if __name__ == "__main__":
    unittest.main()

