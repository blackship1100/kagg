from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mscapital.artifacts import atomic_write_parquet
from mscapital.config import FoldConfig
from mscapital.contracts import Split, TableName
from mscapital.data.canonical import CanonicalStore
from mscapital.deep_learning.blend import blend_tabular_deep
from mscapital.deep_learning.dataset import SequenceDataset, ShardBatchSampler
from mscapital.deep_learning.model import (
    MSCapitalSequenceModel,
    SequenceModelConfig,
    competition_loss,
)
from mscapital.deep_learning.sequences import SequenceStore
from mscapital.deep_learning.training import smoke_deep, train_deep_oof
from tests.helpers import temporary_config, write_synthetic_dataset


class DeepLearningTests(unittest.TestCase):
    def test_sequence_cache_is_aligned_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = temporary_config(root, shard_size=2, expected_samples=6)
            write_synthetic_dataset(config.paths.data_dir, 6, n_months=3)
            canonical = CanonicalStore(config)
            for table in (TableName.MARKET, TableName.TRANSACTION):
                canonical.build(Split.TRAIN, table)
            store = SequenceStore(config, canonical)
            manifest = store.build(Split.TRAIN)
            self.assertEqual(manifest.sample_count, 6)
            self.assertEqual(len(manifest.parts), 3)
            with SequenceDataset(config, Split.TRAIN) as dataset:
                first = dataset[0]
                self.assertEqual(tuple(first["market_values"].shape), (212, 15))
                self.assertEqual(tuple(first["transaction_values"].shape), (256, 6))
                self.assertEqual(tuple(first["transaction_grid"].shape), (60, 10))
                self.assertEqual(int(first["market_mask"].sum()), 3)
                self.assertEqual(int(first["transaction_mask"].sum()), 3)
                sampler = ShardBatchSampler(
                    dataset, np.arange(6), 2, shuffle=False, seed=17
                )
                self.assertEqual(list(sampler), [[0, 1], [2, 3], [4, 5]])

            directory, _ = store.manifest(Split.TRAIN)
            damaged = directory / manifest.parts[0].directory / "market_values.npy"
            damaged.write_bytes(b"damaged")
            repaired = store.build(Split.TRAIN, resume=True)
            self.assertEqual(manifest.content_digest, repaired.content_digest)

    def test_model_forward_backward_and_masking(self) -> None:
        config = SequenceModelConfig(
            market_features=15,
            transaction_features=6,
            transaction_grid_features=10,
            market_steps=12,
            event_steps=16,
            grid_steps=8,
            hidden_size=32,
            attention_layers=1,
            attention_heads=4,
            dropout=0.0,
        )
        model = MSCapitalSequenceModel(config)
        market_mask = torch.zeros(4, 12, dtype=torch.bool)
        market_mask[:, -5:] = True
        transaction_mask = torch.zeros(4, 16, dtype=torch.bool)
        transaction_mask[:, -7:] = True
        prediction = model(
            torch.randn(4, 12, 15),
            market_mask,
            torch.randn(4, 16, 6),
            torch.ones(4, 16, dtype=torch.long),
            transaction_mask,
            torch.randn(4, 8, 10),
        )
        target = torch.randn(4)
        loss, parts = competition_loss(prediction, target, cosine_weight=0.1)
        loss.backward()
        self.assertEqual(tuple(prediction.shape), (4,))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(parts["mse"]), 0.0)
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.parameters())
        )

    def test_real_pipeline_smoke_on_synthetic_feather(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = temporary_config(root, shard_size=4, expected_samples=8)
            write_synthetic_dataset(config.paths.data_dir, 8, n_months=4)
            output_dir, report = smoke_deep(
                config,
                max_samples=8,
                device="cpu",
                max_train_batches=1,
            )
            self.assertTrue((output_dir / "model.pt").is_file())
            self.assertTrue((output_dir / "valid_prediction.npy").is_file())
            self.assertEqual(report["train_rows"], 6)
            self.assertEqual(report["valid_rows"], 2)
            self.assertTrue(np.isfinite(report["valid_cosine"]))

    def test_rolling_oof_and_tabular_blend(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = temporary_config(root, shard_size=4, expected_samples=8)
            config = replace(
                config,
                folds=(
                    FoldConfig("fold_a", (0, 1), (2, 2)),
                    FoldConfig("fold_b", (0, 2), (3, 3)),
                ),
                deep_learning=replace(
                    config.deep_learning,
                    hidden_size=32,
                    attention_layers=1,
                    physical_batch_size=4,
                    effective_batch_size=8,
                    epochs=1,
                    num_workers=0,
                    early_stopping_patience=1,
                ),
            )
            write_synthetic_dataset(config.paths.data_dir, 8, n_months=4)
            canonical = CanonicalStore(config)
            for table in (TableName.MARKET, TableName.TRANSACTION):
                canonical.build(Split.TRAIN, table)
            SequenceStore(config, canonical).build(Split.TRAIN)
            deep_run, metrics = train_deep_oof(config, seed=17, epochs=1, device="cpu")
            self.assertEqual(metrics["covered_rows"], 4)

            deep_table = pq.read_table(
                config.paths.artifacts_dir / "runs" / deep_run / "oof.parquet"
            )
            tabular_run = "tabular-test"
            tabular_dir = config.paths.artifacts_dir / "runs" / tabular_run
            atomic_write_parquet(
                tabular_dir / "oof.parquet",
                pa.table(
                    {
                        "sample_id": deep_table["sample_id"],
                        "month": deep_table["month"],
                        "target": deep_table["target"],
                        "prediction": deep_table["target"],
                        "fold": deep_table["fold"],
                    }
                ),
            )
            blend_run, blend_metrics = blend_tabular_deep(
                config, tabular_run, deep_run, deep_weight=0.25
            )
            self.assertTrue(
                (
                    config.paths.artifacts_dir / "runs" / blend_run / "oof.parquet"
                ).is_file()
            )
            self.assertEqual(blend_metrics["covered_rows"], 4)


if __name__ == "__main__":
    unittest.main()
