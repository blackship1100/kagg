from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mscapital.models.preprocessing import FoldPreprocessor


class FoldPreprocessorTests(unittest.TestCase):
    def test_fit_transform_and_round_trip(self) -> None:
        values = np.array(
            [[0.0, np.nan, 5.0], [1.0, np.nan, 5.0], [100.0, np.nan, 5.0]],
            dtype=np.float32,
        )
        processor = FoldPreprocessor.fit(values, ("a", "b", "c"), (0.0, 0.5))
        transformed = processor.transform(values)
        self.assertEqual(transformed[2, 0], 1.0)
        self.assertTrue(np.isnan(transformed[:, 1]).all())
        self.assertEqual(processor.missing_fraction[1], 1.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            processor.save(Path(temp_dir))
            restored = FoldPreprocessor.load(Path(temp_dir))
            np.testing.assert_allclose(
                restored.transform(values), transformed, equal_nan=True
            )

    def test_validation_only_extreme_cannot_change_training_bounds(self) -> None:
        train = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
        processor = FoldPreprocessor.fit(train, ("x",), (0.0, 1.0))
        validation = processor.transform(np.array([[1000.0]], dtype=np.float32))
        self.assertEqual(validation[0, 0], 2.0)


if __name__ == "__main__":
    unittest.main()
