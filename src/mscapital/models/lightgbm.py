from __future__ import annotations

from pathlib import Path

import numpy as np

from mscapital.config import LightGBMConfig
from mscapital.validation.metrics import cosine_score


class LightGBMRegressor:
    def __init__(self, config: LightGBMConfig, *, seed: int, threads: int) -> None:
        self.config = config
        self.seed = seed
        self.threads = threads
        self.booster = None
        self.best_iteration = 0

    def fit(
        self,
        train_values: np.ndarray,
        train_target: np.ndarray,
        valid_values: np.ndarray,
        valid_target: np.ndarray,
        feature_names: tuple[str, ...],
        train_weight: np.ndarray | None = None,
    ) -> "LightGBMRegressor":
        lgb = _lightgbm()
        params = {
            "objective": self.config.objective,
            "metric": "None",
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "min_data_in_leaf": self.config.min_data_in_leaf,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "lambda_l2": self.config.lambda_l2,
            "max_bin": self.config.max_bin,
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "bagging_seed": self.seed,
            "data_random_seed": self.seed,
            "num_threads": self.threads,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        if self.config.alpha is not None:
            params["alpha"] = self.config.alpha
        train_set = lgb.Dataset(
            train_values,
            label=train_target,
            weight=train_weight,
            feature_name=list(feature_names),
            free_raw_data=False,
        )
        valid_set = lgb.Dataset(
            valid_values,
            label=valid_target,
            reference=train_set,
            feature_name=list(feature_names),
            free_raw_data=False,
        )
        self.booster = lgb.train(
            params,
            train_set,
            num_boost_round=self.config.max_rounds,
            valid_sets=[valid_set],
            valid_names=["valid"],
            feval=_cosine_metric,
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds, first_metric_only=True),
                lgb.log_evaluation(period=100),
            ],
        )
        self.best_iteration = int(self.booster.best_iteration or self.config.max_rounds)
        return self

    def fit_full(
        self,
        values: np.ndarray,
        target: np.ndarray,
        feature_names: tuple[str, ...],
        rounds: int,
        train_weight: np.ndarray | None = None,
    ) -> "LightGBMRegressor":
        lgb = _lightgbm()
        params = {
            "objective": self.config.objective,
            "metric": "None",
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "min_data_in_leaf": self.config.min_data_in_leaf,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "lambda_l2": self.config.lambda_l2,
            "max_bin": self.config.max_bin,
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "bagging_seed": self.seed,
            "data_random_seed": self.seed,
            "num_threads": self.threads,
            "verbosity": -1,
            "deterministic": True,
            "force_col_wise": True,
        }
        if self.config.alpha is not None:
            params["alpha"] = self.config.alpha
        dataset = lgb.Dataset(
            values,
            label=target,
            weight=train_weight,
            feature_name=list(feature_names),
        )
        self.booster = lgb.train(params, dataset, num_boost_round=int(rounds))
        self.best_iteration = int(rounds)
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model has not been fitted")
        return np.asarray(
            self.booster.predict(values, num_iteration=self.best_iteration), dtype=np.float64
        )

    def save(self, path: str | Path) -> None:
        if self.booster is None:
            raise RuntimeError("model has not been fitted")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.tmp")
        self.booster.save_model(str(temp), num_iteration=self.best_iteration)
        temp.replace(target)

    def load(self, path: str | Path) -> "LightGBMRegressor":
        lgb = _lightgbm()
        self.booster = lgb.Booster(model_file=str(path))
        self.best_iteration = int(self.booster.num_trees())
        return self

    def feature_importance(self, feature_names: tuple[str, ...]) -> list[dict[str, float | str]]:
        if self.booster is None:
            raise RuntimeError("model has not been fitted")
        gain = self.booster.feature_importance(importance_type="gain")
        split = self.booster.feature_importance(importance_type="split")
        records = [
            {"feature": name, "gain": float(gain_value), "split": int(split_value)}
            for name, gain_value, split_value in zip(feature_names, gain, split)
        ]
        return sorted(records, key=lambda item: item["gain"], reverse=True)


def _cosine_metric(prediction, dataset):
    target = dataset.get_label()
    try:
        score = cosine_score(target, prediction)
    except ValueError:
        score = 0.0
    return "cosine", score, True


def _lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError(
            "LightGBM is required for tabular training. Install the project dependencies first."
        ) from exc
    return lgb
