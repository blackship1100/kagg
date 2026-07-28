from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mscapital.artifacts import (
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    fingerprint,
    sha256_file,
)
from mscapital.config import ProjectConfig
from mscapital.contracts import FeatureMatrix, Split, TableName
from mscapital.data.canonical import CanonicalManifest, CanonicalStore, ShardSpec
from mscapital.features.cross import CrossFeatureBlock
from mscapital.features.market import MARKET_COLUMNS, MarketFeatureBlock
from mscapital.features.order import ORDER_COLUMNS, OrderFeatureBlock
from mscapital.features.transaction import TRANSACTION_COLUMNS, TransactionFeatureBlock


FEATURE_BLOCKS = ("market", "order", "transaction", "cross")


@dataclass(frozen=True)
class FeatureManifest:
    format_version: int
    block: str
    block_version: int
    split: str
    scope: str
    dataset_fingerprint: str
    row_count: int
    columns: tuple[str, ...]
    column_types: dict[str, str]
    allowed_nan_columns: tuple[str, ...]
    parts: tuple[str, ...]
    content_digest: str

    @classmethod
    def from_json(cls, path: str | Path) -> "FeatureManifest":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("columns", "allowed_nan_columns", "parts"):
            raw[key] = tuple(raw[key])
        return cls(**raw)


class FeatureStore:
    def __init__(self, config: ProjectConfig, canonical: CanonicalStore | None = None) -> None:
        self.config = config
        self.canonical = canonical or CanonicalStore(config)
        self.root = config.paths.cache_dir / "features" / f"v{config.cache.format_version}"
        self.market = MarketFeatureBlock(
            config.features, strict=config.cleaning.strict_schema
        )
        self.order = OrderFeatureBlock(config.features)
        self.transaction = TransactionFeatureBlock(config.features)
        self.cross = CrossFeatureBlock()

    def build(
        self,
        split: Split,
        *,
        resume: bool = False,
        max_samples: int | None = None,
    ) -> dict[str, FeatureManifest]:
        canonical_manifests = self._canonical_manifests(split, max_samples)
        dataset_fingerprint = self._dataset_fingerprint(canonical_manifests)
        scope = "full" if max_samples is None else f"sample_limit_{max_samples}"
        versions = self._block_versions()
        directories = {
            name: self.root / scope / split.value / dataset_fingerprint / name / f"v{versions[name]}"
            for name in FEATURE_BLOCKS
        }
        if resume:
            completed = self._completed_manifests(directories, versions)
            if len(completed) == len(FEATURE_BLOCKS):
                return completed

        aligned = self.canonical.aligned_shards(
            split,
            (TableName.MARKET, TableName.ORDER, TableName.TRANSACTION),
            max_samples=max_samples,
        )
        names_by_block: dict[str, tuple[str, ...]] = {}
        nan_by_block: dict[str, set[str]] = {name: set() for name in FEATURE_BLOCKS}
        part_checksums: dict[str, list[tuple[str, str]]] = {
            name: [] for name in FEATURE_BLOCKS
        }
        row_count = 0

        for shard in aligned:
            _, market_columns = self.canonical.load_shard(
                split,
                TableName.MARKET,
                shard.index,
                MARKET_COLUMNS,
                max_samples=max_samples,
            )
            derived = self.market.derive(market_columns, shard)
            market_matrix = self.market.transform_derived(derived)
            del market_columns, derived
            gc.collect()

            _, order_columns = self.canonical.load_shard(
                split,
                TableName.ORDER,
                shard.index,
                ORDER_COLUMNS,
                max_samples=max_samples,
            )
            order_matrix = self.order.transform(order_columns, market_matrix_context(shard, market_matrix))
            del order_columns
            gc.collect()

            # The context reference is persisted as the first two market features.
            context = market_matrix_context(shard, market_matrix)
            _, transaction_columns = self.canonical.load_shard(
                split,
                TableName.TRANSACTION,
                shard.index,
                TRANSACTION_COLUMNS,
                max_samples=max_samples,
            )
            transaction_matrix = self.transaction.transform(transaction_columns, context)
            del transaction_columns
            gc.collect()
            cross_matrix = self.cross.transform(
                {
                    "market": market_matrix,
                    "order": order_matrix,
                    "transaction": transaction_matrix,
                },
                context,
            )
            matrices = {
                "market": market_matrix,
                "order": order_matrix,
                "transaction": transaction_matrix,
                "cross": cross_matrix,
            }
            for name, matrix in matrices.items():
                previous = names_by_block.setdefault(name, matrix.names)
                if previous != matrix.names:
                    raise ValueError(f"feature schema changed between shards for block {name}")
                nan_columns = np.any(np.isnan(matrix.values), axis=0)
                nan_by_block[name].update(
                    feature_name
                    for feature_name, has_nan in zip(matrix.names, nan_columns)
                    if has_nan
                )
                part_name = f"part_{shard.index:05d}.parquet"
                part_path = directories[name] / part_name
                if not (resume and _valid_parquet(part_path, len(matrix.sample_ids), matrix.names)):
                    atomic_write_parquet(
                        part_path,
                        _to_arrow(matrix),
                        compression=self.config.cache.parquet_compression,
                    )
                    checksum = sha256_file(part_path)
                    atomic_write_text(part_path.with_suffix(".parquet.sha256"), checksum)
                else:
                    checksum = part_path.with_suffix(".parquet.sha256").read_text(
                        encoding="ascii"
                    ).strip()
                part_checksums[name].append((part_name, checksum))
            row_count += shard.sample_count
            del matrices, market_matrix, order_matrix, transaction_matrix, cross_matrix, context
            gc.collect()

        if max_samples is None:
            expected_rows = (
                self.config.dataset.train_sample_count
                if split is Split.TRAIN
                else self.config.dataset.test_sample_count
            )
            if row_count != expected_rows:
                raise ValueError(
                    f"{split.value} features have {row_count} rows; expected {expected_rows}"
                )

        manifests: dict[str, FeatureManifest] = {}
        for name in FEATURE_BLOCKS:
            manifest = FeatureManifest(
                format_version=self.config.cache.format_version,
                block=name,
                block_version=versions[name],
                split=split.value,
                scope=scope,
                dataset_fingerprint=dataset_fingerprint,
                row_count=row_count,
                columns=names_by_block[name],
                column_types={feature: "float32" for feature in names_by_block[name]},
                allowed_nan_columns=tuple(
                    feature for feature in names_by_block[name] if feature in nan_by_block[name]
                ),
                parts=tuple(part for part, _ in part_checksums[name]),
                content_digest=fingerprint(part_checksums[name], length=64),
            )
            payload = asdict(manifest)
            atomic_write_json(directories[name] / "manifest.json", payload)
            manifests[name] = manifest
        return manifests

    def load_matrix(
        self,
        split: Split,
        blocks: Sequence[str] = FEATURE_BLOCKS,
        *,
        max_samples: int | None = None,
    ) -> FeatureMatrix:
        manifests, directories = self._feature_locations(split, max_samples)
        if not blocks:
            raise ValueError("at least one feature block is required")
        row_counts = {manifests[block].row_count for block in blocks}
        if len(row_counts) != 1:
            raise ValueError("feature blocks have inconsistent row counts")
        row_count = row_counts.pop()
        total_columns = sum(len(manifests[block].columns) for block in blocks)
        sample_ids = np.empty(row_count, dtype=np.int32)
        joined_values = np.empty((row_count, total_columns), dtype=np.float32)
        names: list[str] = []
        column_offset = 0
        for block in blocks:
            if block not in manifests:
                raise KeyError(f"unknown or missing feature block: {block}")
            manifest = manifests[block]
            row_offset = 0
            for part in manifest.parts:
                table = pq.read_table(directories[block] / part)
                part_rows = len(table)
                row_slice = slice(row_offset, row_offset + part_rows)
                current_ids = table["sample_id"].to_numpy(zero_copy_only=False)
                if column_offset == 0:
                    sample_ids[row_slice] = current_ids
                elif not np.array_equal(sample_ids[row_slice], current_ids):
                    raise ValueError(f"sample_id mismatch while joining feature block {block}")
                for local_index, name in enumerate(manifest.columns):
                    joined_values[row_slice, column_offset + local_index] = table[
                        name
                    ].to_numpy(zero_copy_only=False)
                row_offset += part_rows
                del table, current_ids
            if row_offset != row_count:
                raise ValueError(f"feature block {block} has an invalid part row total")
            names.extend(manifest.columns)
            column_offset += len(manifest.columns)
        return FeatureMatrix(sample_ids, joined_values, tuple(names))

    def manifests(
        self,
        split: Split,
        *,
        max_samples: int | None = None,
    ) -> dict[str, FeatureManifest]:
        manifests, _ = self._feature_locations(split, max_samples)
        return manifests

    def validate_train_test_schema(self, *, max_samples: int | None = None) -> None:
        train = self.manifests(Split.TRAIN, max_samples=max_samples)
        test = self.manifests(Split.TEST, max_samples=max_samples)
        for block in FEATURE_BLOCKS:
            if train[block].columns != test[block].columns:
                raise ValueError(f"train/test feature names differ for block {block}")
            if train[block].column_types != test[block].column_types:
                raise ValueError(f"train/test feature types differ for block {block}")

    def _canonical_manifests(
        self, split: Split, max_samples: int | None
    ) -> dict[str, CanonicalManifest]:
        return {
            table.value: self.canonical.manifest(split, table, max_samples=max_samples)[1]
            for table in (TableName.MARKET, TableName.ORDER, TableName.TRANSACTION)
        }

    def _dataset_fingerprint(self, manifests: dict[str, CanonicalManifest]) -> str:
        return fingerprint(
            {
                "sources": {name: manifest.content_digest for name, manifest in manifests.items()},
                "features": asdict(self.config.features),
                "block_versions": self._block_versions(),
                "format_version": self.config.cache.format_version,
            }
        )

    def _feature_locations(self, split: Split, max_samples: int | None):
        canonical_manifests = self._canonical_manifests(split, max_samples)
        dataset_fingerprint = self._dataset_fingerprint(canonical_manifests)
        scope = "full" if max_samples is None else f"sample_limit_{max_samples}"
        versions = self._block_versions()
        directories = {
            name: self.root / scope / split.value / dataset_fingerprint / name / f"v{versions[name]}"
            for name in FEATURE_BLOCKS
        }
        manifests = self._completed_manifests(directories, versions)
        if len(manifests) != len(FEATURE_BLOCKS):
            missing = sorted(set(FEATURE_BLOCKS) - set(manifests))
            raise FileNotFoundError(f"feature manifests are missing: {missing}")
        return manifests, directories

    def _block_versions(self) -> dict[str, int]:
        return {
            "market": self.market.version,
            "order": self.order.version,
            "transaction": self.transaction.version,
            "cross": self.cross.version,
        }

    @staticmethod
    def _completed_manifests(
        directories: dict[str, Path], versions: dict[str, int]
    ) -> dict[str, FeatureManifest]:
        completed = {}
        for name, directory in directories.items():
            path = directory / "manifest.json"
            if path.is_file():
                try:
                    manifest = FeatureManifest.from_json(path)
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue
                if (
                    manifest.block == name
                    and manifest.block_version == versions[name]
                    and _feature_manifest_valid(directory, manifest)
                ):
                    completed[name] = manifest
        return completed


def market_matrix_context(shard: ShardSpec, matrix: FeatureMatrix):
    from mscapital.features.base import FeatureContext

    columns = {name: matrix.values[:, index] for index, name in enumerate(matrix.names)}
    return FeatureContext(
        shard=shard,
        reference_mid=columns["market__reference_mid"],
        no_valid_mid=columns["market__no_valid_mid"].astype(bool),
    )


def _to_arrow(matrix: FeatureMatrix) -> pa.Table:
    arrays: dict[str, pa.Array] = {
        "sample_id": pa.array(matrix.sample_ids, type=pa.int32())
    }
    arrays.update(
        {
            name: pa.array(matrix.values[:, index], type=pa.float32())
            for index, name in enumerate(matrix.names)
        }
    )
    return pa.table(arrays)


def _valid_parquet(path: Path, row_count: int, names: Iterable[str]) -> bool:
    checksum_path = path.with_suffix(".parquet.sha256")
    if not path.is_file() or not checksum_path.is_file():
        return False
    try:
        metadata = pq.read_metadata(path)
        expected_names = ["sample_id", *names]
        if metadata.num_rows != row_count or metadata.schema.names != expected_names:
            return False
        return sha256_file(path) == checksum_path.read_text(encoding="ascii").strip()
    except (OSError, pa.ArrowException):
        return False


def _feature_manifest_valid(directory: Path, manifest: FeatureManifest) -> bool:
    expected_names = ["sample_id", *manifest.columns]
    if manifest.column_types != {name: "float32" for name in manifest.columns}:
        return False
    row_count = 0
    records: list[tuple[str, str]] = []
    for part in manifest.parts:
        path = directory / part
        checksum_path = path.with_suffix(".parquet.sha256")
        if not path.is_file() or not checksum_path.is_file():
            return False
        try:
            metadata = pq.read_metadata(path)
            if metadata.schema.names != expected_names:
                return False
            checksum = checksum_path.read_text(encoding="ascii").strip()
            if not checksum or sha256_file(path) != checksum:
                return False
        except (OSError, pa.ArrowException):
            return False
        row_count += metadata.num_rows
        records.append((part, checksum))
    return row_count == manifest.row_count and fingerprint(records, length=64) == manifest.content_digest
