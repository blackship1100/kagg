from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

from mscapital.config import ProjectConfig


def temporary_config(
    root: Path, *, shard_size: int = 2, expected_samples: int = 4
) -> ProjectConfig:
    config = ProjectConfig.from_toml("configs/base.toml")
    paths = replace(
        config.paths,
        data_dir=root / "data",
        artifacts_dir=root / "artifacts",
        cache_dir=root / "artifacts" / "cache",
    )
    cache = replace(config.cache, shard_size=shard_size)
    dataset = replace(
        config.dataset,
        train_sample_count=expected_samples,
        test_sample_count=expected_samples,
    )
    lightgbm = replace(
        config.lightgbm,
        num_leaves=7,
        min_data_in_leaf=1,
        max_rounds=20,
        early_stopping_rounds=5,
    )
    runtime = replace(config.runtime, threads=2, seeds=(17,))
    return replace(
        config,
        paths=paths,
        dataset=dataset,
        cache=cache,
        lightgbm=lightgbm,
        runtime=runtime,
    )


def write_synthetic_dataset(
    data_dir: Path, n_samples: int = 4, *, n_months: int = 2
) -> None:
    (data_dir / "train").mkdir(parents=True, exist_ok=True)
    (data_dir / "test").mkdir(parents=True, exist_ok=True)
    label = pa.table(
        {
            "month": pa.array(np.arange(n_samples) % n_months, type=pa.int16()),
            "sample_id": pa.array(np.arange(n_samples), type=pa.int32()),
            "target": pa.array(np.linspace(-0.01, 0.01, n_samples), type=pa.float32()),
        }
    )
    feather.write_feather(label, data_dir / "train" / "label.feather")
    for split in ("train", "test"):
        feather.write_feather(_market_table(n_samples), data_dir / split / "market.feather")
        feather.write_feather(_order_table(n_samples), data_dir / split / "order.feather")
        feather.write_feather(
            _transaction_table(n_samples), data_dir / split / "transaction.feather"
        )
    pd.DataFrame(
        {"sample_id": np.arange(n_samples), "prediction": np.zeros(n_samples)}
    ).to_csv(data_dir / "submission.csv", index=False)


def _market_table(n_samples: int) -> pa.Table:
    ids = np.repeat(np.arange(n_samples, dtype=np.int32), 3)
    seconds = np.tile(np.array([10.0, 5.0, 0.0], dtype=np.float32), n_samples)
    step = np.tile(np.array([-0.001, 0.0, 0.001], dtype=np.float32), n_samples)
    ask1 = 1.001 + step
    bid1 = 0.999 + step
    ask2 = 1.002 + step
    bid2 = 0.998 + step
    avg = (1.0 + step).astype(np.float32)
    avg[::3] = np.nan
    volume = np.tile(np.array([0, 100, 200], dtype=np.int32), n_samples)
    count = np.tile(np.array([0, 1, 2], dtype=np.int32), n_samples)
    return pa.table(
        {
            "sample_id": pa.array(ids, type=pa.int32()),
            "seconds_before_predict": pa.array(seconds, type=pa.float32()),
            "transaction_avgprice": pa.array(avg, type=pa.float32(), from_pandas=True),
            "transaction_volume": pa.array(volume, type=pa.int32()),
            "transaction_count": pa.array(count, type=pa.int32()),
            "ask_price_1": pa.array(ask1, type=pa.float32()),
            "ask_volume_1": pa.array(np.full(len(ids), 1000), type=pa.int32()),
            "bid_price_1": pa.array(bid1, type=pa.float32()),
            "bid_volume_1": pa.array(np.full(len(ids), 1200), type=pa.int32()),
            "ask_price_2": pa.array(ask2, type=pa.float32()),
            "ask_volume_2": pa.array(np.full(len(ids), 900), type=pa.int32()),
            "bid_price_2": pa.array(bid2, type=pa.float32()),
            "bid_volume_2": pa.array(np.full(len(ids), 1100), type=pa.int32()),
        }
    )


def _order_table(n_samples: int) -> pa.Table:
    ids = np.repeat(np.arange(n_samples, dtype=np.int32), 4)
    return pa.table(
        {
            "sample_id": pa.array(ids, type=pa.int32()),
            "seconds_before_predict": pa.array(
                np.tile([9.0, 5.0, 2.0, 0.0], n_samples), type=pa.float32()
            ),
            "price": pa.array(np.tile([0.999, 1.001, 1.0, 1.002], n_samples), type=pa.float32()),
            "volume": pa.array(np.tile([100, 200, 300, 400], n_samples), type=pa.int32()),
            "side": pa.array(np.tile([0, 1, 0, 1], n_samples), type=pa.int8()),
            "order_action": pa.array(np.tile([0, 0, 1, 1], n_samples), type=pa.int8()),
        }
    )


def _transaction_table(n_samples: int) -> pa.Table:
    ids = np.repeat(np.arange(n_samples, dtype=np.int32), 3)
    return pa.table(
        {
            "sample_id": pa.array(ids, type=pa.int32()),
            "seconds_before_predict": pa.array(
                np.tile([8.0, 3.0, 0.0], n_samples), type=pa.float32()
            ),
            "price": pa.array(np.tile([0.999, 1.0, 1.001], n_samples), type=pa.float32()),
            "volume": pa.array(np.tile([100, 200, 300], n_samples), type=pa.int32()),
            "side": pa.array(np.tile([1, 0, 0], n_samples), type=pa.int8()),
        }
    )
