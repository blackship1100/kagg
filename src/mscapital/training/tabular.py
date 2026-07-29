from __future__ import annotations

import gc
import json
from dataclasses import asdict, replace
from fnmatch import fnmatchcase
from pathlib import Path
from statistics import median

import numpy as np
import pyarrow as pa

from mscapital.artifacts import (
    atomic_save_npy,
    atomic_write_json,
    atomic_write_parquet,
    fingerprint,
)
from mscapital.config import LightGBMConfig, ProjectConfig
from mscapital.contracts import FeatureMatrix, Split, TableName
from mscapital.data.catalog import DataCatalog
from mscapital.features.store import FEATURE_BLOCKS, FeatureStore
from mscapital.models.lightgbm import LightGBMRegressor
from mscapital.models.preprocessing import PREPROCESSOR_VERSION, FoldPreprocessor
from mscapital.validation.baselines import BASELINE_FEATURES, evaluate_baselines
from mscapital.validation.metrics import cosine_report, cosine_score
from mscapital.validation.splits import folds_from_config

DERIVED_FEATURE_SETS = (
    "order_category_ratios",
    "order_pressure",
    "temporal_dynamics",
)


def load_labels(
    config: ProjectConfig,
    *,
    max_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = DataCatalog(config.paths.data_dir).read_columns(
        Split.TRAIN, TableName.LABEL, ["month", "sample_id", "target"]
    )
    if max_samples is not None:
        table = table.slice(0, max_samples)
    months = table["month"].to_numpy(zero_copy_only=False)
    sample_ids = table["sample_id"].to_numpy(zero_copy_only=False)
    target = table["target"].to_numpy(zero_copy_only=False)
    if not np.array_equal(
        sample_ids, np.arange(len(sample_ids), dtype=sample_ids.dtype)
    ):
        raise ValueError("label sample_id must be contiguous and ordered")
    return (
        sample_ids.astype(np.int32),
        months.astype(np.int16),
        target.astype(np.float32),
    )


def run_baselines(
    config: ProjectConfig,
    *,
    max_samples: int | None = None,
    resume: bool = False,
) -> tuple[Path, dict]:
    store = FeatureStore(config)
    matrix = store.load_matrix(
        Split.TRAIN, ("market", "transaction"), max_samples=max_samples
    )
    sample_ids, months, target = load_labels(config, max_samples=max_samples)
    _validate_ids(matrix.sample_ids, sample_ids)
    report = evaluate_baselines(matrix, target, months)
    manifests = store.manifests(Split.TRAIN, max_samples=max_samples)
    identity = fingerprint(
        {
            "features": {
                name: manifest.content_digest for name, manifest in manifests.items()
            },
            "scope": "full" if max_samples is None else f"sample_limit_{max_samples}",
        }
    )
    path = config.paths.artifacts_dir / "baselines" / f"{identity}.json"
    if resume and path.is_file():
        return path, json.loads(path.read_text(encoding="utf-8"))
    atomic_write_json(path, report)
    return path, report


def train_oof(
    config: ProjectConfig,
    *,
    resume: bool = False,
    blocks: tuple[str, ...] = FEATURE_BLOCKS,
    seeds: tuple[int, ...] | None = None,
    exclude_patterns: tuple[str, ...] = (),
    recent_months: int | None = None,
    recency_half_life: float | None = None,
    target_clip_quantiles: tuple[float, float] | None = None,
    objective: str | None = None,
    objective_alpha: float | None = None,
    target_normalization: str = "none",
    derived_feature_sets: tuple[str, ...] = (),
    learning_rate: float | None = None,
    num_leaves: int | None = None,
    min_data_in_leaf: int | None = None,
    feature_fraction: float | None = None,
    bagging_fraction: float | None = None,
    lambda_l2: float | None = None,
    max_bin: int | None = None,
    max_rounds: int | None = None,
    early_stopping_rounds: int | None = None,
) -> tuple[str, dict]:
    experiment = _experiment_payload(
        exclude_patterns,
        recent_months,
        recency_half_life,
        target_clip_quantiles,
        target_normalization,
        derived_feature_sets,
    )
    run_seeds = tuple(config.runtime.seeds if seeds is None else seeds)
    if not run_seeds:
        raise ValueError("at least one seed is required")
    if objective_alpha is not None and (objective != "huber" or objective_alpha <= 0):
        raise ValueError(
            "objective_alpha must be positive and requires objective='huber'"
        )
    _validate_model_overrides(
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_data_in_leaf=min_data_in_leaf,
        feature_fraction=feature_fraction,
        bagging_fraction=bagging_fraction,
        lambda_l2=lambda_l2,
        max_bin=max_bin,
        max_rounds=max_rounds,
        early_stopping_rounds=early_stopping_rounds,
    )
    model_config = replace(
        config.lightgbm,
        objective=config.lightgbm.objective if objective is None else objective,
        alpha=config.lightgbm.alpha if objective_alpha is None else objective_alpha,
        learning_rate=(
            config.lightgbm.learning_rate if learning_rate is None else learning_rate
        ),
        num_leaves=config.lightgbm.num_leaves if num_leaves is None else num_leaves,
        min_data_in_leaf=(
            config.lightgbm.min_data_in_leaf
            if min_data_in_leaf is None
            else min_data_in_leaf
        ),
        feature_fraction=(
            config.lightgbm.feature_fraction
            if feature_fraction is None
            else feature_fraction
        ),
        bagging_fraction=(
            config.lightgbm.bagging_fraction
            if bagging_fraction is None
            else bagging_fraction
        ),
        lambda_l2=config.lightgbm.lambda_l2 if lambda_l2 is None else lambda_l2,
        max_bin=config.lightgbm.max_bin if max_bin is None else max_bin,
        max_rounds=config.lightgbm.max_rounds if max_rounds is None else max_rounds,
        early_stopping_rounds=(
            config.lightgbm.early_stopping_rounds
            if early_stopping_rounds is None
            else early_stopping_rounds
        ),
    )
    model_payload = {
        name: value for name, value in asdict(model_config).items() if value is not None
    }
    feature_store = FeatureStore(config)
    loaded_matrix = feature_store.load_matrix(Split.TRAIN, blocks)
    sample_ids, months, target = load_labels(config)
    _validate_ids(loaded_matrix.sample_ids, sample_ids)
    baseline_matrix = _select_exact_features(
        loaded_matrix, tuple(BASELINE_FEATURES.values())
    )
    augmented_matrix = _add_derived_features(
        loaded_matrix, tuple(experiment["derived_feature_sets"])
    )
    matrix = _exclude_features(augmented_matrix, exclude_patterns)
    if augmented_matrix is not loaded_matrix:
        del loaded_matrix
        gc.collect()
    if matrix is not augmented_matrix:
        del augmented_matrix
        gc.collect()
    manifests = feature_store.manifests(Split.TRAIN)
    run_payload = {
        "model": model_payload,
        "preprocessing": {
            "version": PREPROCESSOR_VERSION,
            "feature_clip_quantiles": list(config.cleaning.feature_clip_quantiles),
            "missing_policy": "native_lightgbm_nan",
        },
        "runtime": {**asdict(config.runtime), "seeds": list(run_seeds)},
        "experiment": experiment,
        "folds": [asdict(fold) for fold in config.folds],
        "blocks": list(blocks),
        "features": {name: manifests[name].content_digest for name in blocks},
    }
    run_id = f"lgbm-{fingerprint(run_payload)}"
    run_dir = config.paths.artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "config.json", run_payload)

    folds = folds_from_config(config.folds)
    oof = np.full(len(target), np.nan, dtype=np.float64)
    fold_labels = np.full(len(target), "", dtype="<U32")
    best_iterations: list[int] = []
    fold_reports: dict[str, dict] = {}
    seed_reports: dict[str, dict] = {}
    baseline_fold_reports: dict[str, dict] = {}
    training_reports: dict[str, dict] = {}

    for fold in folds:
        train_index, valid_index = fold.split(months)
        train_index = _restrict_to_recent_months(
            train_index, months, experiment["recent_months"]
        )
        normalized_target, target_normalization_report = _normalize_target(
            target[train_index],
            months[train_index],
            experiment["target_normalization"],
        )
        train_target, target_bounds = _clip_target(
            normalized_target, experiment["target_clip_quantiles"]
        )
        train_weight = _recency_weights(
            months[train_index], experiment["recency_half_life"]
        )
        preprocessor_dir = run_dir / "preprocessors" / fold.name
        if resume and (preprocessor_dir / "manifest.json").is_file():
            preprocessor = FoldPreprocessor.load(preprocessor_dir)
            if preprocessor.feature_names != matrix.names:
                raise ValueError(f"saved preprocessor schema mismatch for {fold.name}")
        else:
            preprocessor = FoldPreprocessor.fit(
                matrix.values[train_index],
                matrix.names,
                config.cleaning.feature_clip_quantiles,
            )
            preprocessor.save(preprocessor_dir)
        train_values = preprocessor.transform(matrix.values[train_index], copy=False)
        valid_values = preprocessor.transform(matrix.values[valid_index], copy=False)
        seed_predictions = []
        for seed in run_seeds:
            model_dir = run_dir / "models" / fold.name / f"seed_{seed}"
            model_path = model_dir / "model.txt"
            prediction_path = model_dir / "valid_prediction.npy"
            metadata_path = model_dir / "metadata.json"
            if (
                resume
                and model_path.is_file()
                and prediction_path.is_file()
                and metadata_path.is_file()
            ):
                prediction = np.load(prediction_path, allow_pickle=False)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                best_iteration = int(metadata["best_iteration"])
            else:
                model = LightGBMRegressor(
                    model_config, seed=seed, threads=config.runtime.threads
                ).fit(
                    train_values,
                    train_target,
                    valid_values,
                    target[valid_index],
                    matrix.names,
                    train_weight,
                )
                prediction = model.predict(valid_values)
                best_iteration = model.best_iteration
                model.save(model_path)
                atomic_save_npy(prediction_path, prediction)
                atomic_write_json(
                    model_dir / "feature_importance.json",
                    model.feature_importance(matrix.names),
                )
                atomic_write_json(
                    metadata_path,
                    {
                        "fold": fold.name,
                        "seed": seed,
                        "best_iteration": best_iteration,
                        "valid_cosine": cosine_score(target[valid_index], prediction),
                        "train_rows": len(train_index),
                        "train_months": [
                            int(months[train_index].min()),
                            int(months[train_index].max()),
                        ],
                        "target_clip_bounds": list(target_bounds),
                        "target_normalization": target_normalization_report,
                        "weight_range": [
                            float(train_weight.min()),
                            float(train_weight.max()),
                        ],
                    },
                )
            if prediction.shape != (len(valid_index),):
                raise ValueError(
                    f"invalid saved prediction shape for {fold.name}/seed_{seed}"
                )
            seed_predictions.append(prediction)
            best_iterations.append(best_iteration)
            seed_reports[f"{fold.name}/seed_{seed}"] = {
                "cosine": cosine_score(target[valid_index], prediction),
                "best_iteration": best_iteration,
            }
        fold_prediction = np.mean(seed_predictions, axis=0)
        oof[valid_index] = fold_prediction
        fold_labels[valid_index] = fold.name
        fold_reports[fold.name] = asdict(
            cosine_report(target[valid_index], fold_prediction, months[valid_index])
        )
        baseline_fold_reports[fold.name] = evaluate_baselines(
            FeatureMatrix(
                baseline_matrix.sample_ids[valid_index],
                baseline_matrix.values[valid_index],
                baseline_matrix.names,
            ),
            target[valid_index],
            months[valid_index],
        )
        training_reports[fold.name] = {
            "train_rows": len(train_index),
            "valid_rows": len(valid_index),
            "train_months": [
                int(months[train_index].min()),
                int(months[train_index].max()),
            ],
            "target_clip_bounds": list(target_bounds),
            "target_normalization": target_normalization_report,
            "weight_range": [float(train_weight.min()), float(train_weight.max())],
        }
        del train_values, valid_values, preprocessor, train_target, train_weight
        gc.collect()

    covered = np.isfinite(oof)
    overall = asdict(cosine_report(target[covered], oof[covered], months[covered]))
    gate = _baseline_gate(fold_reports, baseline_fold_reports)
    metrics = {
        "run_id": run_id,
        "covered_rows": int(np.count_nonzero(covered)),
        "overall": overall,
        "folds": fold_reports,
        "seeds": seed_reports,
        "baseline_folds": baseline_fold_reports,
        "baseline_gate": gate,
        "training": training_reports,
        "feature_count": len(matrix.names),
        "best_iterations": best_iterations,
        "median_best_iteration": int(median(best_iterations)),
    }
    atomic_write_json(run_dir / "metrics.json", metrics)
    oof_table = pa.table(
        {
            "sample_id": pa.array(sample_ids[covered], type=pa.int32()),
            "month": pa.array(months[covered], type=pa.int16()),
            "target": pa.array(target[covered], type=pa.float32()),
            "prediction": pa.array(oof[covered].astype(np.float32), type=pa.float32()),
            "fold": pa.array(fold_labels[covered]),
        }
    )
    atomic_write_parquet(
        run_dir / "oof.parquet", oof_table, compression=config.cache.parquet_compression
    )
    atomic_write_json(
        run_dir / "manifest.json",
        {
            **run_payload,
            "run_id": run_id,
            "feature_names": list(matrix.names),
            "median_best_iteration": int(median(best_iterations)),
            "oof_path": "oof.parquet",
            "metrics_path": "metrics.json",
        },
    )
    return run_id, metrics


def predict_test(
    config: ProjectConfig,
    run_id: str,
    *,
    resume: bool = False,
) -> Path:
    run_dir = config.paths.artifacts_dir / "runs" / run_id
    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    blocks = tuple(run_manifest["blocks"])
    store = FeatureStore(config)
    loaded_train = store.load_matrix(Split.TRAIN, blocks)
    raw_train_names = loaded_train.names
    experiment = {
        **_experiment_payload((), None, None, None),
        **run_manifest.get("experiment", {}),
    }
    train_matrix = _add_derived_features(
        loaded_train, tuple(experiment["derived_feature_sets"])
    )
    if train_matrix is not loaded_train:
        del loaded_train
        gc.collect()
    loaded_test = store.load_matrix(Split.TEST, blocks)
    if raw_train_names != loaded_test.names:
        raise ValueError("train and test feature schemas do not match")
    test_matrix = _add_derived_features(
        loaded_test, tuple(experiment["derived_feature_sets"])
    )
    if test_matrix is not loaded_test:
        del loaded_test
        gc.collect()
    expected_names = tuple(run_manifest["feature_names"])
    selected_train = _select_exact_features(train_matrix, expected_names)
    selected_test = _select_exact_features(test_matrix, expected_names)
    if selected_train is not train_matrix:
        del train_matrix, test_matrix
        gc.collect()
    train_matrix = selected_train
    test_matrix = selected_test
    train_manifests = store.manifests(Split.TRAIN)
    expected_digests = run_manifest["features"]
    if any(
        train_manifests[name].content_digest != digest
        for name, digest in expected_digests.items()
    ):
        raise ValueError("current training feature content does not match the OOF run")
    sample_ids, months, target = load_labels(config)
    _validate_ids(train_matrix.sample_ids, sample_ids)
    rounds = int(run_manifest["median_best_iteration"])
    model_config = LightGBMConfig(**run_manifest["model"])
    train_index = _restrict_to_recent_months(
        np.arange(len(target), dtype=np.int64), months, experiment["recent_months"]
    )
    normalized_target, target_normalization_report = _normalize_target(
        target[train_index],
        months[train_index],
        experiment["target_normalization"],
    )
    train_target, target_bounds = _clip_target(
        normalized_target, experiment["target_clip_quantiles"]
    )
    train_weight = _recency_weights(
        months[train_index], experiment["recency_half_life"]
    )
    preprocessor_dir = run_dir / "preprocessors" / "full"
    if resume and (preprocessor_dir / "manifest.json").is_file():
        preprocessor = FoldPreprocessor.load(preprocessor_dir)
        if preprocessor.feature_names != train_matrix.names:
            raise ValueError("saved full preprocessor schema mismatch")
    else:
        preprocessor = FoldPreprocessor.fit(
            train_matrix.values[train_index],
            train_matrix.names,
            config.cleaning.feature_clip_quantiles,
        )
        preprocessor.save(preprocessor_dir)
    if len(train_index) == len(target):
        train_values = preprocessor.transform(train_matrix.values, copy=False)
    else:
        train_values = preprocessor.transform(
            train_matrix.values[train_index], copy=False
        )
    test_values = preprocessor.transform(test_matrix.values, copy=False)
    predictions = []
    run_seeds = tuple(int(seed) for seed in run_manifest["runtime"]["seeds"])
    for seed in run_seeds:
        model_dir = run_dir / "full_models" / f"seed_{seed}"
        model_path = model_dir / "model.txt"
        prediction_path = model_dir / "test_prediction.npy"
        if resume and model_path.is_file() and prediction_path.is_file():
            prediction = np.load(prediction_path, allow_pickle=False)
        else:
            model = LightGBMRegressor(
                model_config, seed=seed, threads=config.runtime.threads
            ).fit_full(
                train_values,
                train_target,
                train_matrix.names,
                rounds,
                train_weight,
            )
            prediction = model.predict(test_values)
            model.save(model_path)
            atomic_save_npy(prediction_path, prediction)
        if prediction.shape != (len(test_matrix.sample_ids),):
            raise ValueError(f"invalid test prediction shape for seed {seed}")
        predictions.append(prediction)
    final_prediction = np.mean(predictions, axis=0)
    if not np.isfinite(final_prediction).all() or np.linalg.norm(final_prediction) == 0:
        raise ValueError("test prediction must be finite and have non-zero norm")
    output = run_dir / "test_prediction.npy"
    atomic_save_npy(output, final_prediction)
    atomic_write_json(
        run_dir / "test_prediction.json",
        {
            "rows": len(final_prediction),
            "rounds": rounds,
            "seeds": list(run_seeds),
            "experiment": experiment,
            "train_rows": len(train_index),
            "target_clip_bounds": list(target_bounds),
            "target_normalization": target_normalization_report,
            "weight_range": [float(train_weight.min()), float(train_weight.max())],
            "prediction_norm": float(np.linalg.norm(final_prediction)),
        },
    )
    return output


def read_metrics(config: ProjectConfig, run_id: str) -> dict:
    path = config.paths.artifacts_dir / "runs" / run_id / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_ids(actual: np.ndarray, expected: np.ndarray) -> None:
    if not np.array_equal(actual, expected):
        raise ValueError("feature sample_id values do not match labels")


def _validate_model_overrides(**values: float | None) -> None:
    positive = (
        "learning_rate",
        "num_leaves",
        "min_data_in_leaf",
        "max_bin",
        "max_rounds",
        "early_stopping_rounds",
    )
    for name in positive:
        value = values[name]
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if values["num_leaves"] is not None and values["num_leaves"] < 2:
        raise ValueError("num_leaves must be at least 2")
    for name in ("feature_fraction", "bagging_fraction"):
        value = values[name]
        if value is not None and not 0 < value <= 1:
            raise ValueError(f"{name} must satisfy 0 < value <= 1")
    if values["lambda_l2"] is not None and values["lambda_l2"] < 0:
        raise ValueError("lambda_l2 must be non-negative")


def _experiment_payload(
    exclude_patterns: tuple[str, ...],
    recent_months: int | None,
    recency_half_life: float | None,
    target_clip_quantiles: tuple[float, float] | None,
    target_normalization: str = "none",
    derived_feature_sets: tuple[str, ...] = (),
) -> dict:
    if recent_months is not None and recent_months <= 0:
        raise ValueError("recent_months must be positive")
    if recency_half_life is not None and recency_half_life <= 0:
        raise ValueError("recency_half_life must be positive")
    quantiles = (0.0, 1.0) if target_clip_quantiles is None else target_clip_quantiles
    if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("target clip quantiles must satisfy 0 <= lower < upper <= 1")
    if target_normalization not in {"none", "monthly_std", "monthly_zscore"}:
        raise ValueError(f"unsupported target normalization: {target_normalization}")
    supported_derived = set(DERIVED_FEATURE_SETS)
    unknown_derived = sorted(set(derived_feature_sets) - supported_derived)
    if unknown_derived:
        raise ValueError(f"unsupported derived feature sets: {unknown_derived}")
    return {
        "exclude_patterns": list(exclude_patterns),
        "recent_months": recent_months,
        "recency_half_life": recency_half_life,
        "target_clip_quantiles": list(quantiles),
        "target_normalization": target_normalization,
        "derived_feature_sets": list(dict.fromkeys(derived_feature_sets)),
    }


def _exclude_features(
    matrix: FeatureMatrix, patterns: tuple[str, ...]
) -> FeatureMatrix:
    if not patterns:
        return matrix
    keep = [
        index
        for index, name in enumerate(matrix.names)
        if not any(fnmatchcase(name, pattern) for pattern in patterns)
    ]
    if not keep:
        raise ValueError("feature exclusion removed every feature")
    names = tuple(matrix.names[index] for index in keep)
    return FeatureMatrix(matrix.sample_ids, matrix.values[:, keep], names)


def _select_exact_features(
    matrix: FeatureMatrix, names: tuple[str, ...]
) -> FeatureMatrix:
    if matrix.names == names:
        return matrix
    positions = {name: index for index, name in enumerate(matrix.names)}
    missing = [name for name in names if name not in positions]
    if missing:
        raise ValueError(f"required features are missing: {missing[:5]}")
    indices = [positions[name] for name in names]
    return FeatureMatrix(matrix.sample_ids, matrix.values[:, indices], names)


def _add_derived_features(
    matrix: FeatureMatrix, feature_sets: tuple[str, ...]
) -> FeatureMatrix:
    if not feature_sets:
        return matrix
    unknown = sorted(set(feature_sets) - set(DERIVED_FEATURE_SETS))
    if unknown:
        raise ValueError(f"unsupported derived feature sets: {unknown}")
    positions = {name: index for index, name in enumerate(matrix.names)}
    categories = ("buy_new", "buy_cancel", "sell_new", "sell_cancel")
    derived_names: list[str] = []
    derived_values: list[np.ndarray] = []
    if "order_category_ratios" in feature_sets:
        for window in (1, 2, 5, 10, 30, 60):
            prefix = f"order__w{window}"
            count_total = matrix.values[:, positions[f"{prefix}__event_count"]]
            volume_total = np.expm1(
                matrix.values[:, positions[f"{prefix}__volume_logsum"]]
            )
            for category in categories:
                category_count = matrix.values[
                    :, positions[f"{prefix}__{category}_count"]
                ]
                category_volume = np.expm1(
                    matrix.values[:, positions[f"{prefix}__{category}_volume_logsum"]]
                )
                derived_names.extend(
                    (
                        f"{prefix}__{category}_count_share",
                        f"{prefix}__{category}_volume_share",
                    )
                )
                derived_values.extend(
                    (
                        _safe_float32_ratio(category_count, count_total),
                        _safe_float32_ratio(category_volume, volume_total),
                    )
                )
    if "order_pressure" in feature_sets:
        for window in (1, 2, 5, 10, 30, 60):
            prefix = f"order__w{window}"
            counts, volumes = _order_category_values(matrix, positions, prefix)
            count_pressure = _pressure(
                counts["buy_new"] + counts["sell_cancel"],
                counts["sell_new"] + counts["buy_cancel"],
            )
            volume_pressure = _pressure(
                volumes["buy_new"] + volumes["sell_cancel"],
                volumes["sell_new"] + volumes["buy_cancel"],
            )
            buy_cancel_count_rate = _safe_float32_ratio(
                counts["buy_cancel"], counts["buy_new"] + counts["buy_cancel"]
            )
            sell_cancel_count_rate = _safe_float32_ratio(
                counts["sell_cancel"], counts["sell_new"] + counts["sell_cancel"]
            )
            derived_names.extend(
                (
                    f"{prefix}__new_count_imbalance",
                    f"{prefix}__cancel_count_imbalance",
                    f"{prefix}__new_volume_imbalance",
                    f"{prefix}__cancel_volume_imbalance",
                    f"{prefix}__action_count_pressure",
                    f"{prefix}__buy_cancel_count_rate",
                    f"{prefix}__sell_cancel_count_rate",
                    f"{prefix}__count_volume_pressure_divergence",
                )
            )
            derived_values.extend(
                (
                    _pressure(counts["buy_new"], counts["sell_new"]),
                    _pressure(counts["sell_cancel"], counts["buy_cancel"]),
                    _pressure(volumes["buy_new"], volumes["sell_new"]),
                    _pressure(volumes["sell_cancel"], volumes["buy_cancel"]),
                    count_pressure,
                    buy_cancel_count_rate,
                    sell_cancel_count_rate,
                    count_pressure - volume_pressure,
                )
            )
    if "temporal_dynamics" in feature_sets:
        long_order_prefix = "order__w60"
        long_counts, _ = _order_category_values(matrix, positions, long_order_prefix)
        long_count_pressure = _pressure(
            long_counts["buy_new"] + long_counts["sell_cancel"],
            long_counts["sell_new"] + long_counts["buy_cancel"],
        )
        for window in (1, 2, 5, 10, 30):
            order_prefix = f"order__w{window}"
            transaction_prefix = f"transaction__w{window}"
            short_counts, _ = _order_category_values(matrix, positions, order_prefix)
            short_count_pressure = _pressure(
                short_counts["buy_new"] + short_counts["sell_cancel"],
                short_counts["sell_new"] + short_counts["buy_cancel"],
            )
            pairs = (
                (
                    "order_signed_volume_imbalance_change",
                    f"{order_prefix}__signed_volume_imbalance",
                    f"{long_order_prefix}__signed_volume_imbalance",
                ),
                (
                    "order_cancel_count_rate_change",
                    f"{order_prefix}__cancel_count_rate",
                    f"{long_order_prefix}__cancel_count_rate",
                ),
                (
                    "order_cancel_volume_rate_change",
                    f"{order_prefix}__cancel_volume_rate",
                    f"{long_order_prefix}__cancel_volume_rate",
                ),
                (
                    "transaction_count_imbalance_change",
                    f"{transaction_prefix}__count_imbalance",
                    "transaction__w60__count_imbalance",
                ),
                (
                    "transaction_volume_imbalance_change",
                    f"{transaction_prefix}__volume_imbalance",
                    "transaction__w60__volume_imbalance",
                ),
                (
                    "transaction_vwap_change",
                    f"{transaction_prefix}__vwap_bps",
                    "transaction__w60__vwap_bps",
                ),
                (
                    "transaction_price_mean_change",
                    f"{transaction_prefix}__price_bps__mean",
                    "transaction__w60__price_bps__mean",
                ),
            )
            derived_names.append(
                f"dynamics__w{window}_w60__order_action_count_pressure_change"
            )
            derived_values.append(short_count_pressure - long_count_pressure)
            for name, short_name, long_name in pairs:
                derived_names.append(f"dynamics__w{window}_w60__{name}")
                derived_values.append(
                    matrix.values[:, positions[short_name]]
                    - matrix.values[:, positions[long_name]]
                )

        for window in (5, 10, 30, 60, 180):
            prefix = f"market__w{window}"
            long_prefix = "market__w600"
            normalized_std = matrix.values[
                :, positions[f"{prefix}__mid_bps__std"]
            ] * np.sqrt(600.0 / window)
            normalized_long_std = matrix.values[
                :, positions[f"{long_prefix}__mid_bps__std"]
            ]
            normalized_rv = matrix.values[
                :, positions[f"{prefix}__realized_vol_bps"]
            ] / np.sqrt(float(window))
            normalized_long_rv = matrix.values[
                :, positions[f"{long_prefix}__realized_vol_bps"]
            ] / np.sqrt(600.0)
            derived_names.extend(
                (
                    f"dynamics__w{window}_w600__mid_std_regime_ratio",
                    f"dynamics__w{window}_w600__realized_vol_regime_ratio",
                    f"dynamics__w{window}_w600__spread_mean_change",
                    f"dynamics__w{window}_w600__mid_slope_change",
                    f"dynamics__w{window}_w600__imbalance_l2_mean_change",
                )
            )
            derived_values.extend(
                (
                    _safe_float32_ratio(normalized_std, normalized_long_std),
                    _safe_float32_ratio(normalized_rv, normalized_long_rv),
                    matrix.values[:, positions[f"{prefix}__spread_bps__mean"]]
                    - matrix.values[:, positions[f"{long_prefix}__spread_bps__mean"]],
                    matrix.values[:, positions[f"{prefix}__mid_bps__slope"]]
                    - matrix.values[:, positions[f"{long_prefix}__mid_bps__slope"]],
                    matrix.values[:, positions[f"{prefix}__imbalance_l2__mean"]]
                    - matrix.values[:, positions[f"{long_prefix}__imbalance_l2__mean"]],
                )
            )
        for window in (5, 10, 30, 60, 180, 600):
            prefix = f"market__w{window}"
            realized_vol = matrix.values[:, positions[f"{prefix}__realized_vol_bps"]]
            mid_delta = matrix.values[:, positions[f"{prefix}__mid_bps__delta"]]
            mid_range = (
                matrix.values[:, positions[f"{prefix}__mid_bps__max"]]
                - matrix.values[:, positions[f"{prefix}__mid_bps__min"]]
            )
            derived_names.extend(
                (
                    f"dynamics__w{window}__trend_efficiency",
                    f"dynamics__w{window}__range_per_realized_vol",
                )
            )
            derived_values.extend(
                (
                    _safe_float32_ratio(mid_delta, realized_vol),
                    _safe_float32_ratio(mid_range, realized_vol),
                )
            )
    appended = np.column_stack(derived_values).astype(np.float32, copy=False)
    values = np.empty(
        (len(matrix.sample_ids), matrix.values.shape[1] + appended.shape[1]),
        dtype=np.float32,
    )
    values[:, : matrix.values.shape[1]] = matrix.values
    values[:, matrix.values.shape[1] :] = appended
    return FeatureMatrix(matrix.sample_ids, values, (*matrix.names, *derived_names))


def _safe_float32_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.zeros(len(numerator), dtype=np.float32)
    np.divide(numerator, denominator, out=result, where=denominator > 0)
    return result


def _pressure(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    return _safe_float32_ratio(positive - negative, positive + negative)


def _order_category_values(
    matrix: FeatureMatrix,
    positions: dict[str, int],
    prefix: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    categories = ("buy_new", "buy_cancel", "sell_new", "sell_cancel")
    counts = {
        category: matrix.values[:, positions[f"{prefix}__{category}_count"]]
        for category in categories
    }
    volumes = {
        category: np.expm1(
            matrix.values[:, positions[f"{prefix}__{category}_volume_logsum"]]
        )
        for category in categories
    }
    return counts, volumes


def _restrict_to_recent_months(
    indices: np.ndarray, months: np.ndarray, recent_months: int | None
) -> np.ndarray:
    if recent_months is None:
        return indices
    end_month = int(months[indices].max())
    start_month = end_month - int(recent_months) + 1
    selected = indices[months[indices] >= start_month]
    if len(selected) == 0:
        raise ValueError("recent month restriction produced an empty training set")
    return selected


def _clip_target(
    target: np.ndarray, quantiles: list[float] | tuple[float, float]
) -> tuple[np.ndarray, tuple[float, float]]:
    lower_q, upper_q = quantiles
    lower, upper = np.quantile(target, (lower_q, upper_q))
    return np.clip(target, lower, upper), (float(lower), float(upper))


def _normalize_target(
    target: np.ndarray, months: np.ndarray, method: str
) -> tuple[np.ndarray, dict[str, float | str]]:
    if method == "none":
        return target, {
            "method": method,
            "mean_min": 0.0,
            "mean_max": 0.0,
            "scale_min": 1.0,
            "scale_max": 1.0,
        }
    if method not in {"monthly_std", "monthly_zscore"}:
        raise ValueError(f"unsupported target normalization: {method}")
    transformed = np.empty_like(target, dtype=np.float32)
    means: list[float] = []
    scales: list[float] = []
    for month in np.unique(months):
        mask = months == month
        values = target[mask].astype(np.float64, copy=False)
        mean = float(values.mean()) if method == "monthly_zscore" else 0.0
        scale = float(values.std())
        if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
            scale = 1.0
        transformed[mask] = ((values - mean) / scale).astype(np.float32)
        means.append(mean)
        scales.append(scale)
    return transformed, {
        "method": method,
        "mean_min": float(min(means)),
        "mean_max": float(max(means)),
        "scale_min": float(min(scales)),
        "scale_max": float(max(scales)),
    }


def _recency_weights(months: np.ndarray, half_life: float | None) -> np.ndarray:
    if half_life is None:
        return np.ones(len(months), dtype=np.float32)
    ages = months.max() - months
    weights = np.power(0.5, ages / float(half_life)).astype(np.float32)
    weights /= weights.mean()
    return weights


def _baseline_gate(
    model_folds: dict[str, dict], baseline_folds: dict[str, dict]
) -> dict:
    fold_names = tuple(model_folds)
    model_scores = np.asarray(
        [model_folds[name]["global_score"] for name in fold_names], dtype=np.float64
    )
    baseline_names = tuple(next(iter(baseline_folds.values())))
    baseline_means = {
        baseline: float(
            np.mean(
                [baseline_folds[fold][baseline]["global_score"] for fold in fold_names]
            )
        )
        for baseline in baseline_names
        if all("global_score" in baseline_folds[fold][baseline] for fold in fold_names)
    }
    best_baseline = max(baseline_means, key=baseline_means.get)
    late_folds = fold_names[-2:]
    late_comparisons = {
        fold: {
            "model": float(model_folds[fold]["global_score"]),
            "best_single_signal": float(
                max(
                    report["global_score"]
                    for report in baseline_folds[fold].values()
                    if "global_score" in report
                )
            ),
        }
        for fold in late_folds
    }
    mean_passed = float(model_scores.mean()) > baseline_means[best_baseline]
    late_passed = all(
        values["model"] > values["best_single_signal"]
        for values in late_comparisons.values()
    )
    return {
        "passed": bool(mean_passed and late_passed),
        "model_fold_mean": float(model_scores.mean()),
        "best_baseline": best_baseline,
        "best_baseline_fold_mean": baseline_means[best_baseline],
        "mean_passed": bool(mean_passed),
        "late_folds": late_comparisons,
        "late_folds_passed": bool(late_passed),
        "action_if_failed": "retain artifacts and run feature ablation before P2",
    }
