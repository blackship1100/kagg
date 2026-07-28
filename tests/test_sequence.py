from __future__ import annotations

import unittest

import numpy as np

from mscapital.data.sequence import build_sample_spans, validate_sequence_order


class SequenceTests(unittest.TestCase):
    def test_build_sample_spans(self) -> None:
        spans = build_sample_spans(np.array([0, 0, 1, 3, 3, 3], dtype=np.int32))
        np.testing.assert_array_equal(spans.sample_ids, [0, 1, 3])
        np.testing.assert_array_equal(spans.starts, [0, 2, 3])
        np.testing.assert_array_equal(spans.lengths, [2, 1, 3])
        np.testing.assert_array_equal(spans.ends, [2, 3, 6])

    def test_build_sample_spans_rejects_unsorted_ids(self) -> None:
        with self.assertRaises(ValueError):
            build_sample_spans([0, 2, 1])

    def test_descending_seconds_and_ties_are_valid(self) -> None:
        report = validate_sequence_order(
            sample_ids=[0, 0, 0, 1, 1],
            seconds_before_predict=[59.0, 20.0, 20.0, 55.0, 1.0],
        )
        self.assertTrue(report.ok)
        self.assertEqual(report.time_ties_within_sample, 1)

    def test_increasing_seconds_within_sample_is_invalid(self) -> None:
        report = validate_sequence_order([0, 0], [1.0, 59.0])
        self.assertFalse(report.ok)
        self.assertEqual(report.time_increases_within_sample, 1)


if __name__ == "__main__":
    unittest.main()

