from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class Split(StrEnum):
    TRAIN = "train"
    TEST = "test"


class TableName(StrEnum):
    LABEL = "label"
    MARKET = "market"
    ORDER = "order"
    TRANSACTION = "transaction"
    SUBMISSION = "submission"


@dataclass(frozen=True)
class FeatureMatrix:
    sample_ids: NDArray[np.integer]
    values: NDArray[np.floating]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.sample_ids.ndim != 1:
            raise ValueError("sample_ids must be one-dimensional")
        if self.values.ndim != 2:
            raise ValueError("values must be two-dimensional")
        if len(self.sample_ids) != self.values.shape[0]:
            raise ValueError("sample_ids and values must have matching rows")
        if len(self.names) != self.values.shape[1]:
            raise ValueError("feature names and values must have matching columns")
        if len(set(self.names)) != len(self.names):
            raise ValueError("feature names must be unique")

