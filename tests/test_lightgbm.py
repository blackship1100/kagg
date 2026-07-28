from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mscapital.models.lightgbm import LightGBMRegressor
from tests.helpers import temporary_config


class LightGBMTests(unittest.TestCase):
    def test_fit_save_load_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = temporary_config(Path(temp_dir))
            rng = np.random.default_rng(17)
            values = rng.normal(size=(200, 5)).astype(np.float32)
            target = (values[:, 0] - 0.5 * values[:, 1]).astype(np.float32)
            model = LightGBMRegressor(config.lightgbm, seed=17, threads=2).fit(
                values[:150],
                target[:150],
                values[150:],
                target[150:],
                tuple(f"f{i}" for i in range(5)),
            )
            prediction = model.predict(values[150:])
            self.assertEqual(prediction.shape, (50,))
            self.assertTrue(np.isfinite(prediction).all())
            path = Path(temp_dir) / "model.txt"
            model.save(path)
            loaded = LightGBMRegressor(config.lightgbm, seed=17, threads=2).load(path)
            np.testing.assert_allclose(loaded.predict(values[150:]), prediction)


if __name__ == "__main__":
    unittest.main()
