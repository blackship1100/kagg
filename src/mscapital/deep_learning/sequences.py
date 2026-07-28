from __future__ import annotations

import gc
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from mscapital.artifacts import (
    atomic_save_npy,
    atomic_write_json,
    atomic_write_text,
    fingerprint,
    sha256_file,
    validate_npy,
)
from mscapital.config import ProjectConfig
from mscapital.contracts import Split, TableName
from mscapital.data.canonical import CanonicalStore, ShardSpec
from mscapital.data.sequence import build_sample_spans
from mscapital.features.market import MARKET_COLUMNS, MarketFeatureBlock
from mscapital.features.transaction import TRANSACTION_COLUMNS

SEQUENCE_BUILDER_VERSION = 1

MARKET_FEATURE_NAMES = (
    "seconds_norm",
    "mid_bps_100",
    "mid_return_bps_10",
    "spread_bps_100",
    "microprice_bps_10",
    "imbalance_l1",
    "imbalance_l2",
    "log_depth_l1_10",
    "log_depth_l2_10",
    "trade_price_bps_100",
    "log_trade_volume_10",
    "log_trade_count_10",
    "has_trade",
    "book_valid_l1",
    "delta_seconds_60",
)

TRANSACTION_FEATURE_NAMES = (
    "seconds_norm",
    "gap_norm",
    "price_bps_100",
    "price_delta_bps_10",
    "log_volume_10",
    "same_side",
)

TRANSACTION_GRID_FEATURE_NAMES = (
    "log_buy_volume_10",
    "log_sell_volume_10",
    "log_buy_count_5",
    "log_sell_count_5",
    "volume_imbalance",
    "count_imbalance",
    "vwap_bps_100",
    "log_max_volume_10",
    "last_price_bps_100",
    "has_trade",
)


@dataclass(frozen=True)
class SequencePart:
    index: int
    sample_start: int
    sample_end: int
    directory: str
    checksums: dict[str, str]

    @property
    def sample_count(self) -> int:
        return self.sample_end - self.sample_start


@dataclass(frozen=True)
class SequenceManifest:
    builder_version: int
    format_version: int
    split: str
    scope: str
    dataset_fingerprint: str
    sample_count: int
    market_steps: int
    event_steps: int
    grid_steps: int
    market_features: tuple[str, ...]
    transaction_features: tuple[str, ...]
    transaction_grid_features: tuple[str, ...]
    array_specs: dict[str, dict]
    sources: dict[str, str]
    parts: tuple[SequencePart, ...]
    content_digest: str

    @classmethod
    def from_json(cls, path: str | Path) -> SequenceManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        raw["market_features"] = tuple(raw["market_features"])
        raw["transaction_features"] = tuple(raw["transaction_features"])
        raw["transaction_grid_features"] = tuple(raw["transaction_grid_features"])
        raw["parts"] = tuple(SequencePart(**item) for item in raw["parts"])
        return cls(**raw)


class SequenceStore:
    def __init__(
        self,
        config: ProjectConfig,
        canonical: CanonicalStore | None = None,
    ) -> None:
        self.config = config
        self.canonical = canonical or CanonicalStore(config)
        self.root = (
            config.paths.cache_dir / "sequences" / f"v{SEQUENCE_BUILDER_VERSION}"
        )
        self.market_block = MarketFeatureBlock(
            config.features, strict=config.cleaning.strict_schema
        )

    def build(
        self,
        split: Split,
        *,
        resume: bool = False,
        max_samples: int | None = None,
    ) -> SequenceManifest:
        output_dir, identity, sources = self._location(split, max_samples)
        manifest_path = output_dir / "manifest.json"
        if resume and manifest_path.is_file():
            manifest = SequenceManifest.from_json(manifest_path)
            if manifest.dataset_fingerprint == identity and self.validate(
                manifest, output_dir
            ):
                return manifest

        output_dir.mkdir(parents=True, exist_ok=True)
        aligned = self.canonical.aligned_shards(
            split,
            (TableName.MARKET, TableName.TRANSACTION),
            max_samples=max_samples,
        )
        array_specs = self._array_specs()
        parts: list[SequencePart] = []
        for shard in aligned:
            if len(aligned) > 4:
                print(
                    f"[sequence {split.value}] part {shard.index + 1}/{len(aligned)}",
                    flush=True,
                )
            part_dir = output_dir / f"part_{shard.index:05d}"
            if resume and self._part_valid(part_dir, shard, array_specs):
                checksums = self._read_checksums(part_dir, array_specs)
            else:
                arrays = self._build_part(split, shard, max_samples=max_samples)
                checksums = self._write_part(part_dir, arrays)
                del arrays
                gc.collect()
            parts.append(
                SequencePart(
                    index=shard.index,
                    sample_start=shard.sample_start,
                    sample_end=shard.sample_end,
                    directory=part_dir.name,
                    checksums=checksums,
                )
            )

        sample_count = sum(part.sample_count for part in parts)
        expected = (
            self.config.dataset.train_sample_count
            if split is Split.TRAIN
            else self.config.dataset.test_sample_count
        )
        if max_samples is None and sample_count != expected:
            raise ValueError(
                f"{split.value} sequence cache has {sample_count} samples; expected {expected}"
            )
        digest_records = [
            (part.directory, name, checksum)
            for part in parts
            for name, checksum in sorted(part.checksums.items())
        ]
        manifest = SequenceManifest(
            builder_version=SEQUENCE_BUILDER_VERSION,
            format_version=self.config.cache.format_version,
            split=split.value,
            scope="full" if max_samples is None else f"sample_limit_{max_samples}",
            dataset_fingerprint=identity,
            sample_count=sample_count,
            market_steps=self.config.deep_learning.market_max_steps,
            event_steps=self.config.deep_learning.event_max_steps,
            grid_steps=self.config.deep_learning.event_grid_steps,
            market_features=MARKET_FEATURE_NAMES,
            transaction_features=TRANSACTION_FEATURE_NAMES,
            transaction_grid_features=TRANSACTION_GRID_FEATURE_NAMES,
            array_specs=array_specs,
            sources=sources,
            parts=tuple(parts),
            content_digest=fingerprint(digest_records, length=64),
        )
        atomic_write_json(manifest_path, asdict(manifest))
        return manifest

    def manifest(
        self,
        split: Split,
        *,
        max_samples: int | None = None,
    ) -> tuple[Path, SequenceManifest]:
        output_dir, identity, _ = self._location(split, max_samples)
        path = output_dir / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError(f"sequence cache is missing: {path}")
        manifest = SequenceManifest.from_json(path)
        if manifest.dataset_fingerprint != identity or not self.validate(
            manifest, output_dir
        ):
            raise ValueError(f"sequence cache is invalid: {path}")
        return output_dir, manifest

    def load_part(
        self,
        split: Split,
        part_index: int,
        *,
        max_samples: int | None = None,
        mmap_mode: str | None = "r",
    ) -> dict[str, np.ndarray]:
        directory, manifest = self.manifest(split, max_samples=max_samples)
        try:
            part = manifest.parts[part_index]
        except IndexError as exc:
            raise IndexError(f"sequence part {part_index} does not exist") from exc
        return {
            name: np.load(
                directory / part.directory / f"{name}.npy",
                mmap_mode=mmap_mode,
                allow_pickle=False,
            )
            for name in manifest.array_specs
        }

    def validate(self, manifest: SequenceManifest, directory: Path) -> bool:
        if manifest.builder_version != SEQUENCE_BUILDER_VERSION:
            return False
        if sum(part.sample_count for part in manifest.parts) != manifest.sample_count:
            return False
        records = []
        for part in manifest.parts:
            shard = ShardSpec(
                part.index,
                part.sample_start,
                part.sample_end,
                0,
                0,
            )
            part_dir = directory / part.directory
            if not self._part_valid(part_dir, shard, manifest.array_specs):
                return False
            checksums = self._read_checksums(part_dir, manifest.array_specs)
            if checksums != part.checksums:
                return False
            records.extend(
                (part.directory, name, checksum)
                for name, checksum in sorted(checksums.items())
            )
        return fingerprint(records, length=64) == manifest.content_digest

    def _location(
        self, split: Split, max_samples: int | None
    ) -> tuple[Path, str, dict[str, str]]:
        market = self.canonical.manifest(
            split, TableName.MARKET, max_samples=max_samples
        )[1]
        transaction = self.canonical.manifest(
            split, TableName.TRANSACTION, max_samples=max_samples
        )[1]
        sources = {
            "market": market.content_digest,
            "transaction": transaction.content_digest,
        }
        payload = {
            "builder_version": SEQUENCE_BUILDER_VERSION,
            "sources": sources,
            "market_steps": self.config.deep_learning.market_max_steps,
            "event_steps": self.config.deep_learning.event_max_steps,
            "grid_steps": self.config.deep_learning.event_grid_steps,
            "market_features": MARKET_FEATURE_NAMES,
            "transaction_features": TRANSACTION_FEATURE_NAMES,
            "transaction_grid_features": TRANSACTION_GRID_FEATURE_NAMES,
        }
        identity = fingerprint(payload)
        scope = "full" if max_samples is None else f"sample_limit_{max_samples}"
        return self.root / scope / split.value / identity, identity, sources

    def _array_specs(self) -> dict[str, dict]:
        deep = self.config.deep_learning
        return {
            "sample_id": {"shape": [], "dtype": "int32"},
            "market_values": {
                "shape": [deep.market_max_steps, len(MARKET_FEATURE_NAMES)],
                "dtype": "float16",
            },
            "market_mask": {
                "shape": [deep.market_max_steps],
                "dtype": "bool",
            },
            "transaction_values": {
                "shape": [deep.event_max_steps, len(TRANSACTION_FEATURE_NAMES)],
                "dtype": "float16",
            },
            "transaction_side": {
                "shape": [deep.event_max_steps],
                "dtype": "int8",
            },
            "transaction_mask": {
                "shape": [deep.event_max_steps],
                "dtype": "bool",
            },
            "transaction_grid": {
                "shape": [deep.event_grid_steps, len(TRANSACTION_GRID_FEATURE_NAMES)],
                "dtype": "float16",
            },
        }

    def _build_part(
        self,
        split: Split,
        shard: ShardSpec,
        *,
        max_samples: int | None,
    ) -> dict[str, np.ndarray]:
        _, market_columns = self.canonical.load_shard(
            split,
            TableName.MARKET,
            shard.index,
            MARKET_COLUMNS,
            max_samples=max_samples,
        )
        market_derived = self.market_block.derive(market_columns, shard)
        market_rows = _market_rows(market_derived)
        market_values, market_mask = _left_pad_recent(
            np.asarray(market_columns["sample_id"]),
            market_rows,
            shard,
            self.config.deep_learning.market_max_steps,
        )
        reference_mid = np.asarray(
            market_derived.context.reference_mid, dtype=np.float64
        )
        del market_rows, market_derived, market_columns
        gc.collect()

        _, transaction_columns = self.canonical.load_shard(
            split,
            TableName.TRANSACTION,
            shard.index,
            TRANSACTION_COLUMNS,
            max_samples=max_samples,
        )
        transaction_rows, price_bps = _transaction_rows(
            transaction_columns, shard, reference_mid
        )
        transaction_values, transaction_mask, transaction_side = _left_pad_transaction(
            transaction_columns,
            transaction_rows,
            shard,
            self.config.deep_learning.event_max_steps,
        )
        transaction_grid = _transaction_grid(
            transaction_columns,
            price_bps,
            shard,
            self.config.deep_learning.event_grid_steps,
        )
        sample_ids = np.arange(shard.sample_start, shard.sample_end, dtype=np.int32)
        arrays = {
            "sample_id": sample_ids,
            "market_values": market_values.astype(np.float16),
            "market_mask": market_mask,
            "transaction_values": transaction_values.astype(np.float16),
            "transaction_side": transaction_side,
            "transaction_mask": transaction_mask,
            "transaction_grid": transaction_grid.astype(np.float16),
        }
        _validate_finite_arrays(arrays)
        return arrays

    @staticmethod
    def _write_part(part_dir: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, str]:
        checksums = {}
        for name, values in arrays.items():
            path = part_dir / f"{name}.npy"
            atomic_save_npy(path, np.asarray(values))
            checksum = sha256_file(path)
            atomic_write_text(path.with_suffix(".npy.sha256"), checksum)
            checksums[name] = checksum
        return checksums

    @staticmethod
    def _read_checksums(part_dir: Path, specs: Mapping[str, dict]) -> dict[str, str]:
        return {
            name: (part_dir / f"{name}.npy.sha256").read_text(encoding="ascii").strip()
            for name in specs
        }

    @staticmethod
    def _part_valid(
        part_dir: Path,
        shard: ShardSpec,
        specs: Mapping[str, dict],
    ) -> bool:
        for name, spec in specs.items():
            shape = (shard.sample_count, *spec["shape"])
            if not validate_npy(
                part_dir / f"{name}.npy", shape, np.dtype(spec["dtype"])
            ):
                return False
        return True


def _market_rows(derived) -> np.ndarray:
    ids = np.asarray(derived.group_ids)
    seconds = np.asarray(derived.seconds, dtype=np.float64)
    same = np.zeros(len(ids), dtype=bool)
    same[1:] = ids[1:] == ids[:-1]
    mid_return = np.zeros(len(ids), dtype=np.float64)
    mid_return[1:] = np.where(same[1:], derived.mid_bps[1:] - derived.mid_bps[:-1], 0.0)
    delta_seconds = np.zeros(len(ids), dtype=np.float64)
    delta_seconds[1:] = np.where(same[1:], seconds[:-1] - seconds[1:], 0.0)
    values = np.column_stack(
        (
            np.clip(seconds / 600.0, 0.0, 1.0),
            np.clip(derived.mid_bps / 100.0, -10.0, 10.0),
            np.clip(mid_return / 10.0, -10.0, 10.0),
            np.clip(derived.spread_bps / 100.0, -10.0, 10.0),
            np.clip(derived.microprice_bps / 10.0, -10.0, 10.0),
            derived.imbalance_l1,
            derived.imbalance_l2,
            derived.log_depth_l1 / 10.0,
            derived.log_depth_l2 / 10.0,
            np.clip(derived.trade_price_bps / 100.0, -10.0, 10.0),
            np.log1p(np.maximum(derived.positive_trade_volume, 0.0)) / 10.0,
            np.log1p(np.maximum(derived.positive_trade_count, 0.0)) / 10.0,
            derived.has_trade,
            derived.book_valid_l1,
            np.clip(delta_seconds / 60.0, 0.0, 10.0),
        )
    )
    return np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0)


def _transaction_rows(
    columns: Mapping[str, np.ndarray],
    shard: ShardSpec,
    reference_mid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(columns["sample_id"], dtype=np.int64) - shard.sample_start
    seconds = np.asarray(columns["seconds_before_predict"], dtype=np.float64)
    price = np.asarray(columns["price"], dtype=np.float64)
    volume = np.asarray(columns["volume"], dtype=np.float64)
    side = np.asarray(columns["side"], dtype=np.int8)
    price_bps = np.log(np.maximum(price, 1e-12) / reference_mid[ids]) * 10_000.0
    same = np.zeros(len(ids), dtype=bool)
    same[1:] = ids[1:] == ids[:-1]
    gap = np.zeros(len(ids), dtype=np.float64)
    gap[1:] = np.where(same[1:], seconds[:-1] - seconds[1:], 0.0)
    delta = np.zeros(len(ids), dtype=np.float64)
    delta[1:] = np.where(same[1:], price_bps[1:] - price_bps[:-1], 0.0)
    persistence = np.zeros(len(ids), dtype=np.float64)
    persistence[1:] = np.where(same[1:], side[1:] == side[:-1], 0.0)
    values = np.column_stack(
        (
            np.clip(seconds / 60.0, 0.0, 1.0),
            np.clip(gap / 60.0, 0.0, 1.0),
            np.clip(price_bps / 100.0, -10.0, 10.0),
            np.clip(delta / 10.0, -10.0, 10.0),
            np.log1p(np.maximum(volume, 0.0)) / 10.0,
            persistence,
        )
    )
    return np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0), price_bps


def _left_pad_recent(
    sample_ids: np.ndarray,
    row_values: np.ndarray,
    shard: ShardSpec,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    spans = build_sample_spans(sample_ids)
    values = np.zeros(
        (shard.sample_count, max_steps, row_values.shape[1]), dtype=np.float32
    )
    mask = np.zeros((shard.sample_count, max_steps), dtype=bool)
    for sample_id, start, length in zip(spans.sample_ids, spans.starts, spans.lengths):
        local = int(sample_id) - shard.sample_start
        keep = min(int(length), max_steps)
        source = slice(int(start + length - keep), int(start + length))
        values[local, max_steps - keep :] = row_values[source]
        mask[local, max_steps - keep :] = True
    return values, mask


def _left_pad_transaction(
    columns: Mapping[str, np.ndarray],
    row_values: np.ndarray,
    shard: ShardSpec,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(columns["sample_id"])
    values, mask = _left_pad_recent(ids, row_values, shard, max_steps)
    side_rows = np.asarray(columns["side"], dtype=np.int8) + 1
    side_rows = np.clip(side_rows, 1, 2).astype(np.int8)
    side = np.zeros((shard.sample_count, max_steps), dtype=np.int8)
    spans = build_sample_spans(ids)
    for sample_id, start, length in zip(spans.sample_ids, spans.starts, spans.lengths):
        local = int(sample_id) - shard.sample_start
        keep = min(int(length), max_steps)
        source = slice(int(start + length - keep), int(start + length))
        side[local, max_steps - keep :] = side_rows[source]
    return values, mask, side


def _transaction_grid(
    columns: Mapping[str, np.ndarray],
    price_bps: np.ndarray,
    shard: ShardSpec,
    grid_steps: int,
) -> np.ndarray:
    sample_ids = np.asarray(columns["sample_id"], dtype=np.int64)
    seconds = np.asarray(columns["seconds_before_predict"], dtype=np.float64)
    volume = np.asarray(columns["volume"], dtype=np.float64)
    side = np.asarray(columns["side"], dtype=np.int8)
    local_ids = sample_ids - shard.sample_start
    bins = (
        grid_steps
        - 1
        - np.floor(np.clip(seconds, 0.0, np.nextafter(float(grid_steps), 0.0))).astype(
            np.int64
        )
    )
    bins = np.clip(bins, 0, grid_steps - 1)
    flat_index = local_ids * grid_steps + bins
    flat_size = shard.sample_count * grid_steps
    output = np.zeros(
        (shard.sample_count, grid_steps, len(TRANSACTION_GRID_FEATURE_NAMES)),
        dtype=np.float32,
    )
    flat_output = output.reshape(flat_size, -1)
    buy = side == 0
    sell = side == 1
    buy_volume = np.bincount(
        flat_index, weights=np.where(buy, volume, 0.0), minlength=flat_size
    )
    sell_volume = np.bincount(
        flat_index, weights=np.where(sell, volume, 0.0), minlength=flat_size
    )
    buy_count = np.bincount(flat_index[buy], minlength=flat_size)
    sell_count = np.bincount(flat_index[sell], minlength=flat_size)
    total_volume = buy_volume + sell_volume
    total_count = buy_count + sell_count
    flat_output[:, 0] = np.log1p(buy_volume) / 10.0
    flat_output[:, 1] = np.log1p(sell_volume) / 10.0
    flat_output[:, 2] = np.log1p(buy_count) / 5.0
    flat_output[:, 3] = np.log1p(sell_count) / 5.0
    np.divide(
        buy_volume - sell_volume,
        total_volume,
        out=flat_output[:, 4],
        where=total_volume > 0,
    )
    np.divide(
        buy_count - sell_count,
        total_count,
        out=flat_output[:, 5],
        where=total_count > 0,
    )
    weighted_price = np.bincount(
        flat_index, weights=price_bps * volume, minlength=flat_size
    )
    np.divide(
        weighted_price,
        total_volume,
        out=flat_output[:, 6],
        where=total_volume > 0,
    )
    flat_output[:, 6] = np.clip(flat_output[:, 6] / 100.0, -10.0, 10.0)
    max_volume = np.zeros(flat_size, dtype=np.float64)
    np.maximum.at(max_volume, flat_index, volume)
    flat_output[:, 7] = np.log1p(max_volume) / 10.0
    reversed_index = flat_index[::-1]
    _, reversed_positions = np.unique(reversed_index, return_index=True)
    last_rows = len(flat_index) - 1 - reversed_positions
    flat_output[flat_index[last_rows], 8] = np.clip(
        price_bps[last_rows] / 100.0, -10.0, 10.0
    )
    flat_output[total_count > 0, 9] = 1.0
    return np.nan_to_num(output, nan=0.0, posinf=10.0, neginf=-10.0)


def _validate_finite_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    sample_ids = arrays["sample_id"]
    if not np.array_equal(
        sample_ids,
        np.arange(
            sample_ids[0], sample_ids[0] + len(sample_ids), dtype=sample_ids.dtype
        ),
    ):
        raise ValueError("sequence sample_id must be contiguous and ordered")
    for name, values in arrays.items():
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
            raise ValueError(f"sequence array contains non-finite values: {name}")
