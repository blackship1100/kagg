from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from mscapital.config import ProjectConfig
from mscapital.contracts import Split, TableName
from mscapital.data.canonical import CanonicalStore
from mscapital.data.catalog import DataCatalog, failed_validations
from mscapital.features.store import FEATURE_BLOCKS, FeatureStore
from mscapital.training.blend import blend_runs
from mscapital.training.ensemble import ensemble_runs
from mscapital.training.postprocess import postprocess_run
from mscapital.training.submission import make_submission
from mscapital.training.tabular import (
    DERIVED_FEATURE_SETS,
    predict_test,
    read_metrics,
    run_baselines,
    train_oof,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mscapital")
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "validate-data", help="Validate required files and Feather schemas"
    )

    cache = subparsers.add_parser(
        "build-cache", help="Build aligned canonical column shards"
    )
    _add_split(cache)
    cache.add_argument(
        "--table",
        choices=["all", "label", "market", "order", "transaction"],
        default="all",
    )
    _add_build_options(cache)
    cache.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)

    features = subparsers.add_parser(
        "build-features", help="Build sample-level feature blocks"
    )
    _add_split(features)
    _add_build_options(features)

    sequences = subparsers.add_parser(
        "build-sequences",
        help="Build model-ready market and transaction sequence shards",
    )
    _add_split(sequences)
    _add_build_options(sequences)

    baselines = subparsers.add_parser(
        "run-baselines", help="Evaluate deterministic baselines"
    )
    baselines.add_argument("--max-samples", type=int)
    baselines.add_argument("--resume", action="store_true")

    train = subparsers.add_parser(
        "train-tabular", help="Train rolling LightGBM OOF models"
    )
    train.add_argument("--resume", action="store_true")
    train.add_argument(
        "--blocks", nargs="+", choices=FEATURE_BLOCKS, default=list(FEATURE_BLOCKS)
    )
    train.add_argument("--seeds", nargs="+", type=int)
    train.add_argument("--exclude-pattern", action="append", default=[])
    train.add_argument("--recent-months", type=int)
    train.add_argument("--recency-half-life", type=float)
    train.add_argument("--target-clip-quantiles", nargs=2, type=float)
    train.add_argument(
        "--objective",
        choices=("regression_l2", "huber", "regression_l1", "fair"),
        help="Override the configured LightGBM regression objective for this run",
    )
    train.add_argument(
        "--objective-alpha",
        type=float,
        help="Huber transition threshold in target units",
    )
    train.add_argument(
        "--target-normalization",
        choices=("none", "monthly_std", "monthly_zscore"),
        default="none",
        help="Fold-local target scaling; month is never passed to the model",
    )
    train.add_argument(
        "--derived-feature-set",
        action="append",
        choices=DERIVED_FEATURE_SETS,
        default=[],
        help="Add deterministic features derived from the cached matrix",
    )
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--num-leaves", type=int)
    train.add_argument("--min-data-in-leaf", type=int)
    train.add_argument("--feature-fraction", type=float)
    train.add_argument("--bagging-fraction", type=float)
    train.add_argument("--lambda-l2", type=float)
    train.add_argument("--max-bin", type=int)
    train.add_argument("--max-rounds", type=int)
    train.add_argument("--early-stopping-rounds", type=int)

    evaluate = subparsers.add_parser("evaluate-oof", help="Print saved OOF metrics")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--resume", action="store_true")

    predict = subparsers.add_parser(
        "predict-test", help="Fit full models and predict test data"
    )
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--resume", action="store_true")

    submission = subparsers.add_parser(
        "make-submission", help="Validate and write submission CSV"
    )
    submission.add_argument("--run-id", required=True)
    submission.add_argument("--output", type=Path)
    submission.add_argument("--resume", action="store_true")

    ensemble = subparsers.add_parser(
        "ensemble-runs", help="Combine compatible OOF and test predictions"
    )
    ensemble.add_argument("--run-id", nargs="+", required=True)
    ensemble.add_argument("--weight", nargs="+", type=float)

    blend = subparsers.add_parser(
        "blend-runs", help="Blend aligned OOF/test predictions across configurations"
    )
    blend.add_argument("--run-id", nargs="+", required=True)
    blend.add_argument("--weight", nargs="+", type=float)
    blend.add_argument("--no-normalize", action="store_false", dest="normalize")
    blend.set_defaults(normalize=True)

    postprocess = subparsers.add_parser(
        "postprocess-run", help="Apply deterministic signed-power calibration"
    )
    postprocess.add_argument("--run-id", required=True)
    postprocess.add_argument("--power", required=True, type=float)
    postprocess.add_argument("--center", action="store_true")

    deep = subparsers.add_parser(
        "train-deep", help="Train rolling Market/Transaction deep sequence models"
    )
    deep.add_argument("--resume", action="store_true")
    deep.add_argument("--seed", type=int, default=17)
    deep.add_argument("--fold", action="append", default=[])
    deep.add_argument("--epochs", type=int)
    deep.add_argument("--device", default="auto")
    deep.add_argument("--max-samples", type=int)

    deep_predict = subparsers.add_parser(
        "predict-deep-test", help="Predict test data with saved deep fold models"
    )
    deep_predict.add_argument("--run-id", required=True)
    deep_predict.add_argument("--resume", action="store_true")
    deep_predict.add_argument("--device", default="auto")

    blend = subparsers.add_parser(
        "blend-tabular-deep", help="Blend aligned LightGBM and deep OOF predictions"
    )
    blend.add_argument("--tabular-run-id", required=True)
    blend.add_argument("--deep-run-id", required=True)
    blend.add_argument("--deep-weight", type=float)

    smoke = subparsers.add_parser(
        "smoke-deep", help="Run a real-data cache/train/save/reload deep smoke test"
    )
    smoke.add_argument("--max-samples", type=int, default=1000)
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--device", default="auto")
    smoke.add_argument("--resume", action="store_true")
    smoke.add_argument("--max-train-batches", type=int)
    smoke.add_argument(
        "--full-model", action="store_true", help="Use the production-width model"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ProjectConfig.from_toml(args.config)

    if args.command == "validate-data":
        results = DataCatalog(config.paths.data_dir).validate()
        for result in results:
            status = "OK" if result.ok else "FAIL"
            detail = f" - {result.error}" if result.error else ""
            if result.schema is not None and not result.schema.ok:
                detail = (
                    f" - missing={result.schema.missing_columns}, "
                    f"extra={result.schema.extra_columns}, "
                    f"types={result.schema.type_mismatches}, "
                    f"order_matches={result.schema.order_matches}"
                )
            print(f"[{status}] {result.entry.key}: {result.path}{detail}")
        return 1 if failed_validations(results) else 0

    if args.command == "build-cache":
        tasks = [
            (split, table)
            for split in _splits(args.split)
            for table in _tables(args.table, split)
        ]
        if len(tasks) > 1 and not args.in_process:
            for split, table in tasks:
                command = [
                    sys.executable,
                    "-m",
                    "mscapital.cli",
                    "--config",
                    str(Path(args.config).resolve()),
                    "build-cache",
                    "--split",
                    split.value,
                    "--table",
                    table.value,
                    "--in-process",
                ]
                if args.resume:
                    command.append("--resume")
                if args.max_samples is not None:
                    command.extend(("--max-samples", str(args.max_samples)))
                subprocess.run(command, check=True)
            return 0
        store = CanonicalStore(config)
        for split, table in tasks:
            manifest = store.build(
                split,
                table,
                resume=args.resume,
                max_samples=args.max_samples,
            )
            print(
                f"[OK] {split.value}/{table.value}: "
                f"{manifest.row_count:,} rows, {len(manifest.shards)} shards"
            )
        return 0

    if args.command == "build-features":
        store = FeatureStore(config)
        for split in _splits(args.split):
            manifests = store.build(
                split,
                resume=args.resume,
                max_samples=args.max_samples,
            )
            for block, manifest in manifests.items():
                print(
                    f"[OK] {split.value}/{block}: {manifest.row_count:,} rows, "
                    f"{len(manifest.columns)} features"
                )
        if args.split == "both":
            store.validate_train_test_schema(max_samples=args.max_samples)
            print("[OK] train/test feature schemas match")
        return 0

    if args.command == "build-sequences":
        from mscapital.deep_learning.sequences import SequenceStore

        store = SequenceStore(config)
        for split in _splits(args.split):
            manifest = store.build(
                split, resume=args.resume, max_samples=args.max_samples
            )
            print(
                f"[OK] {split.value}: {manifest.sample_count:,} samples, "
                f"{len(manifest.parts)} sequence shards"
            )
        return 0

    if args.command == "run-baselines":
        path, report = run_baselines(
            config, max_samples=args.max_samples, resume=args.resume
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Saved: {path}")
        return 0

    if args.command == "train-tabular":
        run_id, metrics = train_oof(
            config,
            resume=args.resume,
            blocks=tuple(args.blocks),
            seeds=tuple(args.seeds) if args.seeds else None,
            exclude_patterns=tuple(args.exclude_pattern),
            recent_months=args.recent_months,
            recency_half_life=args.recency_half_life,
            target_clip_quantiles=(
                tuple(args.target_clip_quantiles)
                if args.target_clip_quantiles is not None
                else None
            ),
            objective=args.objective,
            objective_alpha=args.objective_alpha,
            target_normalization=args.target_normalization,
            derived_feature_sets=tuple(args.derived_feature_set),
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_in_leaf,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            lambda_l2=args.lambda_l2,
            max_bin=args.max_bin,
            max_rounds=args.max_rounds,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Run ID: {run_id}")
        return 0

    if args.command == "evaluate-oof":
        print(
            json.dumps(read_metrics(config, args.run_id), ensure_ascii=False, indent=2)
        )
        return 0

    if args.command == "predict-test":
        path = predict_test(config, args.run_id, resume=args.resume)
        print(f"Saved: {path}")
        return 0

    if args.command == "make-submission":
        path = make_submission(
            config, args.run_id, output=args.output, resume=args.resume
        )
        print(f"Saved: {path}")
        return 0

    if args.command == "ensemble-runs":
        ensemble_id, metrics = ensemble_runs(
            config,
            tuple(args.run_id),
            weights=tuple(args.weight) if args.weight is not None else None,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Run ID: {ensemble_id}")
        return 0

    if args.command == "blend-runs":
        blend_id, metrics = blend_runs(
            config,
            tuple(args.run_id),
            weights=tuple(args.weight) if args.weight is not None else None,
            normalize=args.normalize,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Run ID: {blend_id}")
        return 0

    if args.command == "postprocess-run":
        processed_id, metrics = postprocess_run(
            config, args.run_id, power=args.power, center=args.center
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Run ID: {processed_id}")
        return 0

    if args.command == "train-deep":
        from mscapital.deep_learning.training import train_deep_oof

        run_id, metrics = train_deep_oof(
            config,
            resume=args.resume,
            seed=args.seed,
            fold_names=tuple(args.fold) if args.fold else None,
            epochs=args.epochs,
            device=args.device,
            max_samples=args.max_samples,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Run ID: {run_id}")
        return 0

    if args.command == "predict-deep-test":
        from mscapital.deep_learning.training import predict_deep_test

        path = predict_deep_test(
            config, args.run_id, resume=args.resume, device=args.device
        )
        print(f"Saved: {path}")
        return 0

    if args.command == "blend-tabular-deep":
        from mscapital.deep_learning.blend import blend_tabular_deep

        run_id, metrics = blend_tabular_deep(
            config,
            args.tabular_run_id,
            args.deep_run_id,
            deep_weight=args.deep_weight,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Run ID: {run_id}")
        return 0

    if args.command == "smoke-deep":
        from mscapital.deep_learning.training import smoke_deep

        output_dir, report = smoke_deep(
            config,
            max_samples=args.max_samples,
            seed=args.seed,
            device=args.device,
            resume=args.resume,
            max_train_batches=args.max_train_batches,
            full_model=args.full_model,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"Saved: {output_dir}")
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


def _add_split(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")


def _add_build_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int)


def _splits(value: str) -> tuple[Split, ...]:
    if value == "both":
        return Split.TRAIN, Split.TEST
    return (Split(value),)


def _tables(value: str, split: Split) -> tuple[TableName, ...]:
    if value != "all":
        table = TableName(value)
        if split is Split.TEST and table is TableName.LABEL:
            return ()
        return (table,)
    if split is Split.TRAIN:
        return TableName.LABEL, TableName.MARKET, TableName.ORDER, TableName.TRANSACTION
    return TableName.MARKET, TableName.ORDER, TableName.TRANSACTION


if __name__ == "__main__":
    raise SystemExit(main())
