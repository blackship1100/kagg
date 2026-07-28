from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from mscapital.contracts import Split, TableName
from mscapital.data.canonical import CanonicalStore
from mscapital.features.store import FEATURE_BLOCKS, FeatureStore
from tests.helpers import temporary_config, write_synthetic_dataset


class PipelineIntegrationTests(unittest.TestCase):
    def test_canonical_resume_repairs_a_corrupt_part(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = temporary_config(root)
            write_synthetic_dataset(config.paths.data_dir)
            store = CanonicalStore(config)
            manifest = store.build(Split.TRAIN, TableName.MARKET)
            directory, _ = store.manifest(Split.TRAIN, TableName.MARKET)
            part = directory / "columns" / "sample_id" / "part_00000.npy"
            part.write_bytes(b"corrupt")
            repaired = store.build(Split.TRAIN, TableName.MARKET, resume=True)
            values = np.load(part, allow_pickle=False)
            self.assertEqual(len(values), repaired.shards[0].row_count)
            self.assertEqual(manifest.content_digest, repaired.content_digest)

    def test_feature_blocks_are_aligned_for_train_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = temporary_config(root)
            write_synthetic_dataset(config.paths.data_dir)
            canonical = CanonicalStore(config)
            for split in (Split.TRAIN, Split.TEST):
                for table in (TableName.MARKET, TableName.ORDER, TableName.TRANSACTION):
                    canonical.build(split, table)
            features = FeatureStore(config, canonical)
            train_manifests = features.build(Split.TRAIN)
            test_manifests = features.build(Split.TEST)
            self.assertEqual(set(train_manifests), set(FEATURE_BLOCKS))
            train = features.load_matrix(Split.TRAIN)
            test = features.load_matrix(Split.TEST)
            self.assertEqual(train.names, test.names)
            self.assertEqual(train.values.shape[0], 4)
            self.assertGreater(train.values.shape[1], 300)
            self.assertFalse(np.isinf(train.values).any())
            train_directory = features._feature_locations(Split.TRAIN, None)[1]["market"]
            corrupt_part = train_directory / train_manifests["market"].parts[0]
            corrupt_part.write_bytes(b"corrupt")
            resumed = features.build(Split.TRAIN, resume=True)
            self.assertEqual(
                train_manifests["cross"].content_digest,
                resumed["cross"].content_digest,
            )
            self.assertGreater(corrupt_part.stat().st_size, len(b"corrupt"))


if __name__ == "__main__":
    unittest.main()
