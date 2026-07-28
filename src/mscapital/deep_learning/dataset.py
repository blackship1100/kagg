from __future__ import annotations

import bisect
import math
from collections.abc import Iterator, Sequence
from typing import Self

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from mscapital.config import ProjectConfig
from mscapital.contracts import Split
from mscapital.deep_learning.sequences import SequenceStore


class SequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Memory-mapped sequence dataset that opens only the shard being sampled."""

    def __init__(
        self,
        config: ProjectConfig,
        split: Split,
        *,
        max_samples: int | None = None,
        target: np.ndarray | None = None,
        months: np.ndarray | None = None,
    ) -> None:
        self.directory, self.manifest = SequenceStore(config).manifest(
            split, max_samples=max_samples
        )
        self.split = split
        self.max_samples = max_samples
        self._part_ends = [part.sample_end for part in self.manifest.parts]
        self._open_part_index: int | None = None
        self._open_arrays: dict[str, np.ndarray] = {}
        self.target = _validate_optional_vector(
            target, self.manifest.sample_count, "target"
        )
        self.months = _validate_optional_vector(
            months, self.manifest.sample_count, "months"
        )

    def __len__(self) -> int:
        return self.manifest.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        part_index = bisect.bisect_right(self._part_ends, index)
        part = self.manifest.parts[part_index]
        arrays = self._arrays_for_part(part_index)
        local = index - part.sample_start
        item = {
            "sample_id": torch.tensor(
                int(arrays["sample_id"][local]), dtype=torch.int64
            ),
            "market_values": _float_tensor(arrays["market_values"][local]),
            "market_mask": _bool_tensor(arrays["market_mask"][local]),
            "transaction_values": _float_tensor(arrays["transaction_values"][local]),
            "transaction_side": torch.tensor(
                np.array(arrays["transaction_side"][local], dtype=np.int64, copy=True)
            ),
            "transaction_mask": _bool_tensor(arrays["transaction_mask"][local]),
            "transaction_grid": _float_tensor(arrays["transaction_grid"][local]),
        }
        if self.target is not None:
            item["target"] = torch.tensor(
                float(self.target[index]), dtype=torch.float32
            )
        if self.months is not None:
            item["month"] = torch.tensor(int(self.months[index]), dtype=torch.int64)
        return item

    def part_for_index(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return bisect.bisect_right(self._part_ends, index)

    def _arrays_for_part(self, part_index: int) -> dict[str, np.ndarray]:
        if self._open_part_index != part_index:
            self.close()
            part = self.manifest.parts[part_index]
            part_dir = self.directory / part.directory
            self._open_arrays = {
                name: np.load(
                    part_dir / f"{name}.npy", mmap_mode="r", allow_pickle=False
                )
                for name in self.manifest.array_specs
            }
            self._open_part_index = part_index
        return self._open_arrays

    def close(self) -> None:
        arrays = getattr(self, "_open_arrays", {})
        for values in arrays.values():
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()
        arrays.clear()
        self._open_part_index = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_open_part_index"] = None
        state["_open_arrays"] = {}
        return state


class ShardBatchSampler(Sampler[list[int]]):
    """Shuffle samples within cache shards to avoid cross-file random I/O."""

    def __init__(
        self,
        dataset: SequenceDataset,
        indices: Sequence[int] | np.ndarray,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        if (
            self.indices.ndim != 1
            or np.any(self.indices < 0)
            or np.any(self.indices >= len(dataset))
        ):
            raise ValueError("indices must be valid one-dimensional dataset positions")
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        self._groups = self._group_by_part()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        group_order = np.arange(len(self._groups))
        if self.shuffle:
            rng.shuffle(group_order)
        for group_index in group_order:
            group = self._groups[group_index]
            current = group.copy()
            if self.shuffle:
                rng.shuffle(current)
            for start in range(0, len(current), self.batch_size):
                batch = current[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch.tolist()

    def __len__(self) -> int:
        if self.drop_last:
            return sum(len(group) // self.batch_size for group in self._groups)
        return sum(math.ceil(len(group) / self.batch_size) for group in self._groups)

    def _group_by_part(self) -> list[np.ndarray]:
        if len(self.indices) == 0:
            return []
        ends = np.asarray(self._part_ends(), dtype=np.int64)
        part_indices = np.searchsorted(ends, self.indices, side="right")
        return [
            self.indices[part_indices == part_index]
            for part_index in np.unique(part_indices)
        ]

    def _part_ends(self) -> list[int]:
        return [part.sample_end for part in self.dataset.manifest.parts]


def _float_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(values, dtype=np.float32, copy=True))


def _bool_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.array(values, dtype=bool, copy=True))


def _validate_optional_vector(
    values: np.ndarray | None, expected: int, name: str
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},)")
    return array
