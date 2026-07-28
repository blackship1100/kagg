from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mscapital.config import FoldConfig


@dataclass(frozen=True)
class MonthFold:
    name: str
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int

    def __post_init__(self) -> None:
        if self.train_start > self.train_end or self.valid_start > self.valid_end:
            raise ValueError("month ranges must be ordered")
        if self.train_end >= self.valid_start:
            raise ValueError("training months must end before validation months")

    def split(self, months: ArrayLike) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        values = np.asarray(months)
        if values.ndim != 1:
            raise ValueError("months must be one-dimensional")
        train = np.flatnonzero((values >= self.train_start) & (values <= self.train_end))
        valid = np.flatnonzero((values >= self.valid_start) & (values <= self.valid_end))
        if len(train) == 0 or len(valid) == 0:
            raise ValueError(f"fold {self.name} produced an empty train or validation split")
        return train.astype(np.int64, copy=False), valid.astype(np.int64, copy=False)


def folds_from_config(configs: tuple[FoldConfig, ...]) -> tuple[MonthFold, ...]:
    names = [config.name for config in configs]
    if len(names) != len(set(names)):
        raise ValueError("fold names must be unique")
    return tuple(
        MonthFold(
            name=config.name,
            train_start=config.train_months[0],
            train_end=config.train_months[1],
            valid_start=config.valid_months[0],
            valid_end=config.valid_months[1],
        )
        for config in configs
    )

