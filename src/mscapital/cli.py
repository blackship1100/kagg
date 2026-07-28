from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from mscapital.config import ProjectConfig
from mscapital.data.catalog import DataCatalog, failed_validations
from mscapital.data.canonical import CanonicalStore
from mscapital.contracts import Split, TableName
from mscapital.features.store import FEATURE_BLOCKS, FeatureStore
from mscapital.training.ensemble import ensemble_runs
from mscapital.training.submission import make_submission
from mscapital.training.tabular import (
    predict_test,
    read_metrics,
    run_baselines,
    train_oof,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mscapital")
    parser.add_argument("--config", type=Path, default=Path("configs/base.toml"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-data", help="Validate required files and Feather schemas")

    cache = subparsers.add_parser("build-cache", help="Build aligned canonical column shards")
    _add_split(cache)
    cache.add_argument(
        "--table",
        choices=["all", "label", "market", "order", "transaction"],
        default="all",
    )
    _add_build_options(cache)
    cache.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)

    features = subparsers.add_parser("build-features", help="Build sample-level feature blocks")
    _add_split(features)
    _add_build_options(features)

    baselines = subparsers.add_parser("run-baselines", help="Evaluate deterministic baselines")
    baselines.add_argument("--max-samples", type=int)
    baselines.add_argument("--resume", action="store_true")

    train = subparsers.add_parser("train-tabular", help="Train rolling LightGBM OOF models")
    train.add_argument("--resume", action="store_true")
    train.add_argument("--blocks", nargs="+", choices=FEATURE_BLOCKS, default=list(FEATURE_BLOCKS))
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
        choices=("order_category_ratios",),
        default=[],
        help="Add deterministic features derived from the cached matrix",
    )

    evaluate = subparsers.add_parser("evaluate-oof", help="Print saved OOF metrics")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--resume", action="store_true")

    predict = subparsers.add_parser("predict-test", help="Fit full models and predict test data")
    predict.add_argument("--run-id", required=True)
    predict.add_argument("--resume", action="store_true")

    submission = subparsers.add_parser("make-submission", help="Validate and write submission CSV")
    submission.add_argument("--run-id", required=True)
    submission.add_argument("--output", type=Path)
    submission.add_argument("--resume", action="store_true")

    ensemble = subparsers.add_parser(
        "ensemble-runs", help="Combine compatible OOF and test predictions"
    )
    ensemble.add_argument("--run-id", nargs="+", required=True)
    ensemble.add_argument("--weight", nargs="+", type=float)
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
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        print(f"Run ID: {run_id}")
        return 0

    if args.command == "evaluate-oof":
        print(json.dumps(read_metrics(config, args.run_id), ensure_ascii=False, indent=2))
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
