from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from mscapital.contracts import FeatureMatrix, TableName
from mscapital.data.canonical import ShardSpec


@dataclass(frozen=True)
class FeatureContext:
    shard: ShardSpec
    reference_mid: np.ndarray
    no_valid_mid: np.ndarray

    @property
    def sample_ids(self) -> np.ndarray:
        return np.arange(self.shard.sample_start, self.shard.sample_end, dtype=np.int32)


@runtime_checkable
class FeatureBlock(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> int: ...

    @property
    def required_columns(self) -> tuple[str, ...]: ...

    @property
    def required_tables(self) -> tuple[TableName, ...]: ...

    def transform(
        self,
        columns: Mapping[str, np.ndarray],
        context: FeatureContext,
    ) -> FeatureMatrix: ...
