from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

import numpy as np
from numpy.typing import NDArray

@runtime_checkable
class TabularRegressor(Protocol):
    def fit(
        self,
        train_values: NDArray[np.floating],
        train_target: NDArray[np.floating],
        valid_values: NDArray[np.floating],
        valid_target: NDArray[np.floating],
        feature_names: tuple[str, ...],
    ) -> Self: ...

    def predict(self, values: NDArray[np.floating]) -> NDArray[np.floating]: ...
