from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    artifacts_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class DatasetConfig:
    train_sample_count: int
    test_sample_count: int


@dataclass(frozen=True)
class CleaningConfig:
    strict_schema: bool
    feature_clip_quantiles: tuple[float, float]
    volume_transform: str


@dataclass(frozen=True)
class SequenceConfig:
    time_order: str
    market_max_length: int
    order_max_length: int
    transaction_max_length: int


@dataclass(frozen=True)
class CacheConfig:
    format_version: int
    shard_size: int
    checksum_algorithm: str
    parquet_compression: str


@dataclass(frozen=True)
class FeaturesConfig:
    market_windows: tuple[int, ...]
    order_windows: tuple[int, ...]
    transaction_windows: tuple[int, ...]
    order_distance_bins_bps: tuple[int, ...]
    large_order_volume: int
    large_trade_volume: int
    order_price_clip: tuple[float, float]


@dataclass(frozen=True)
class RuntimeConfig:
    threads: int
    seeds: tuple[int, ...]


@dataclass(frozen=True)
class LightGBMConfig:
    objective: str
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l2: float
    max_bin: int
    max_rounds: int
    early_stopping_rounds: int
    alpha: float | None = None


@dataclass(frozen=True)
class DeepLearningConfig:
    device_profile: str
    market_max_steps: int
    event_max_steps: int
    event_grid_steps: int
    hidden_size: int
    attention_layers: int
    attention_heads: int
    physical_batch_size: int
    effective_batch_size: int
    max_vram_gb: float


@dataclass(frozen=True)
class FoldConfig:
    name: str
    train_months: tuple[int, int]
    valid_months: tuple[int, int]


@dataclass(frozen=True)
class ProjectConfig:
    paths: PathsConfig
    dataset: DatasetConfig
    cleaning: CleaningConfig
    sequence: SequenceConfig
    cache: CacheConfig
    features: FeaturesConfig
    runtime: RuntimeConfig
    lightgbm: LightGBMConfig
    deep_learning: DeepLearningConfig
    folds: tuple[FoldConfig, ...]

    @classmethod
    def from_toml(cls, path: str | Path) -> "ProjectConfig":
        config_path = Path(path).resolve()
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)

        base_dir = config_path.parent
        paths = raw["paths"]
        dataset = raw["dataset"]
        cleaning = raw["cleaning"]
        sequence = raw["sequence"]
        cache = raw["cache"]
        features = raw["features"]
        runtime = raw["runtime"]
        lightgbm = raw["model"]["lightgbm"]
        deep_learning = raw["deep_learning"]
        folds = raw["validation"]["folds"]

        return cls(
            paths=PathsConfig(
                data_dir=_resolve_path(base_dir, paths["data_dir"]),
                artifacts_dir=_resolve_path(base_dir, paths["artifacts_dir"]),
                cache_dir=_resolve_path(base_dir, paths["cache_dir"]),
            ),
            dataset=DatasetConfig(
                train_sample_count=int(dataset["train_sample_count"]),
                test_sample_count=int(dataset["test_sample_count"]),
            ),
            cleaning=CleaningConfig(
                strict_schema=bool(cleaning["strict_schema"]),
                feature_clip_quantiles=_pair(
                    cleaning["feature_clip_quantiles"],
                    "feature_clip_quantiles",
                    float,
                ),
                volume_transform=str(cleaning["volume_transform"]),
            ),
            sequence=SequenceConfig(
                time_order=str(sequence["time_order"]),
                market_max_length=int(sequence["market_max_length"]),
                order_max_length=int(sequence["order_max_length"]),
                transaction_max_length=int(sequence["transaction_max_length"]),
            ),
            cache=CacheConfig(
                format_version=int(cache["format_version"]),
                shard_size=int(cache["shard_size"]),
                checksum_algorithm=str(cache["checksum_algorithm"]),
                parquet_compression=str(cache["parquet_compression"]),
            ),
            features=FeaturesConfig(
                market_windows=_tuple(features["market_windows"], "market_windows", int),
                order_windows=_tuple(features["order_windows"], "order_windows", int),
                transaction_windows=_tuple(features["transaction_windows"], "transaction_windows", int),
                order_distance_bins_bps=_tuple(
                    features["order_distance_bins_bps"], "order_distance_bins_bps", int
                ),
                large_order_volume=int(features["large_order_volume"]),
                large_trade_volume=int(features["large_trade_volume"]),
                order_price_clip=_pair(features["order_price_clip"], "order_price_clip", float),
            ),
            runtime=RuntimeConfig(
                threads=int(runtime["threads"]),
                seeds=_tuple(runtime["seeds"], "seeds", int),
            ),
            lightgbm=LightGBMConfig(
                objective=str(lightgbm["objective"]),
                learning_rate=float(lightgbm["learning_rate"]),
                num_leaves=int(lightgbm["num_leaves"]),
                min_data_in_leaf=int(lightgbm["min_data_in_leaf"]),
                feature_fraction=float(lightgbm["feature_fraction"]),
                bagging_fraction=float(lightgbm["bagging_fraction"]),
                bagging_freq=int(lightgbm["bagging_freq"]),
                lambda_l2=float(lightgbm["lambda_l2"]),
                max_bin=int(lightgbm["max_bin"]),
                max_rounds=int(lightgbm["max_rounds"]),
                early_stopping_rounds=int(lightgbm["early_stopping_rounds"]),
                alpha=(
                    float(lightgbm["alpha"])
                    if lightgbm.get("alpha") is not None
                    else None
                ),
            ),
            deep_learning=DeepLearningConfig(
                device_profile=str(deep_learning["device_profile"]),
                market_max_steps=int(deep_learning["market_max_steps"]),
                event_max_steps=int(deep_learning["event_max_steps"]),
                event_grid_steps=int(deep_learning["event_grid_steps"]),
                hidden_size=int(deep_learning["hidden_size"]),
                attention_layers=int(deep_learning["attention_layers"]),
                attention_heads=int(deep_learning["attention_heads"]),
                physical_batch_size=int(deep_learning["physical_batch_size"]),
                effective_batch_size=int(deep_learning["effective_batch_size"]),
                max_vram_gb=float(deep_learning["max_vram_gb"]),
            ),
            folds=tuple(
                FoldConfig(
                    name=str(item["name"]),
                    train_months=_pair(item["train_months"], "train_months", int),
                    valid_months=_pair(item["valid_months"], "valid_months", int),
                )
                for item in folds
            ),
        )


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _pair(values: Any, name: str, converter: type) -> tuple[Any, Any]:
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return converter(values[0]), converter(values[1])


def _tuple(values: Any, name: str, converter: type) -> tuple[Any, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(converter(value) for value in values)
