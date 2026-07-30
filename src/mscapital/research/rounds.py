from __future__ import annotations

import ctypes
import json
import platform
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mscapital.artifacts import atomic_write_json
from mscapital.config import ProjectConfig
from mscapital.training.tabular import train_oof


@dataclass(frozen=True)
class ResearchRound:
    """One pre-registered, OOF-only LightGBM experiment."""

    number: int
    code: str
    phase: str
    hypothesis: str
    arguments: dict[str, Any]


def research_rounds() -> tuple[ResearchRound, ...]:
    """Return the fixed 20-round CPU research queue.

    Every candidate uses the same rolling folds and seed.  The queue deliberately
    excludes prior negative ideas so results remain interpretable rather than a
    repeated search over the public leaderboard.
    """

    all_blocks = ("market", "order", "transaction", "cross")
    category_ratio = {"derived_feature_sets": ("order_category_ratios",)}
    return (
        ResearchRound(
            1,
            "reference_all_blocks",
            "reference",
            "Establish a reproducible seed-97 reference for all later deltas.",
            {"blocks": all_blocks, **category_ratio},
        ),
        ResearchRound(
            2,
            "market_only",
            "block_signal",
            "Measure price and book features without order-flow information.",
            {"blocks": ("market",)},
        ),
        ResearchRound(
            3,
            "transaction_only",
            "block_signal",
            "Measure the standalone signal from executed trades.",
            {"blocks": ("transaction",)},
        ),
        ResearchRound(
            4,
            "market_transaction",
            "block_signal",
            "Test the direct price-and-trade combination without resting orders.",
            {"blocks": ("market", "transaction")},
        ),
        ResearchRound(
            5,
            "market_order",
            "block_signal",
            "Test whether resting liquidity improves price features without trades.",
            {"blocks": ("market", "order")},
        ),
        ResearchRound(
            6,
            "order_transaction",
            "block_signal",
            "Test whether order and trade flow carry signal without the full book block.",
            {"blocks": ("order", "transaction"), **category_ratio},
        ),
        ResearchRound(
            7,
            "no_cross",
            "block_ablation",
            "Measure whether hand-built cross features add information beyond source blocks.",
            {"blocks": ("market", "order", "transaction"), **category_ratio},
        ),
        ResearchRound(
            8,
            "no_market",
            "block_ablation",
            "Measure order-flow and transaction information without market aggregates.",
            {"blocks": ("order", "transaction", "cross"), **category_ratio},
        ),
        ResearchRound(
            9,
            "no_order",
            "block_ablation",
            "Measure price, trades, and cross features without resting-order aggregates.",
            {"blocks": ("market", "transaction", "cross")},
        ),
        ResearchRound(
            10,
            "no_transaction",
            "block_ablation",
            "Measure price, orders, and cross features without executed-trade aggregates.",
            {"blocks": ("market", "order", "cross"), **category_ratio},
        ),
        ResearchRound(
            11,
            "compact_63_700",
            "capacity",
            "Test a smaller, lower-variance tree ensemble.",
            {
                "blocks": all_blocks,
                **category_ratio,
                "num_leaves": 63,
                "min_data_in_leaf": 700,
                "feature_fraction": 0.90,
                "bagging_fraction": 0.90,
                "lambda_l2": 1.0,
            },
        ),
        ResearchRound(
            12,
            "medium_95_750",
            "capacity",
            "Test medium tree capacity with moderate feature and row sampling.",
            {
                "blocks": all_blocks,
                **category_ratio,
                "num_leaves": 95,
                "min_data_in_leaf": 750,
                "feature_fraction": 0.90,
                "bagging_fraction": 0.90,
                "lambda_l2": 1.5,
            },
        ),
        ResearchRound(
            13,
            "standard_127_750",
            "capacity",
            "Test baseline capacity with additional leaf support and regularization.",
            {
                "blocks": all_blocks,
                **category_ratio,
                "num_leaves": 127,
                "min_data_in_leaf": 750,
                "feature_fraction": 0.85,
                "bagging_fraction": 0.85,
                "lambda_l2": 1.5,
            },
        ),
        ResearchRound(
            14,
            "wide_159_750",
            "capacity",
            "Test whether controlled extra tree capacity finds complementary nonlinear signal.",
            {
                "blocks": all_blocks,
                **category_ratio,
                "num_leaves": 159,
                "min_data_in_leaf": 750,
                "feature_fraction": 0.80,
                "bagging_fraction": 0.80,
                "lambda_l2": 2.0,
            },
        ),
        ResearchRound(
            15,
            "compact_63_1500",
            "capacity",
            "Test strong regularization for a potentially lower-correlation blend member.",
            {
                "blocks": all_blocks,
                **category_ratio,
                "num_leaves": 63,
                "min_data_in_leaf": 1500,
                "feature_fraction": 0.90,
                "bagging_fraction": 0.90,
                "lambda_l2": 2.0,
            },
        ),
        ResearchRound(
            16,
            "medium_95_1200",
            "capacity",
            "Test an intermediate high-regularization model.",
            {
                "blocks": all_blocks,
                **category_ratio,
                "num_leaves": 95,
                "min_data_in_leaf": 1200,
                "feature_fraction": 0.90,
                "bagging_fraction": 0.90,
                "lambda_l2": 3.0,
            },
        ),
        ResearchRound(
            17,
            "standard_127_1000_l3",
            "capacity",
            "Test baseline capacity under high leaf support and L2 regularization.",
            {
                "blocks": all_blocks,
                **category_ratio,
                "num_leaves": 127,
                "min_data_in_leaf": 1000,
                "feature_fraction": 0.90,
                "bagging_fraction": 0.90,
                "lambda_l2": 3.0,
            },
        ),
        ResearchRound(
            18,
            "medium_long_horizon",
            "horizon_ablation",
            "Test a less reactive representation by removing the shortest event windows.",
            {
                "blocks": all_blocks,
                **category_ratio,
                "exclude_patterns": (
                    "market__w5__*",
                    "market__w10__*",
                    "order__w1__*",
                    "order__w2__*",
                    "transaction__w1__*",
                    "transaction__w2__*",
                    "cross__w1__*",
                    "cross__w2__*",
                ),
            },
        ),
        ResearchRound(
            19,
            "short_horizon",
            "horizon_ablation",
            "Test short-horizon information by removing the 180 and 600 second market windows.",
            {
                "blocks": all_blocks,
                **category_ratio,
                "exclude_patterns": ("market__w180__*", "market__w600__*"),
            },
        ),
        ResearchRound(
            20,
            "flow_first",
            "horizon_ablation",
            "Test a flow-first model by removing direct market mid-price return summaries.",
            {
                "blocks": all_blocks,
                **category_ratio,
                "exclude_patterns": ("market__w*__mid_bps*",),
            },
        ),
    )


def run_research(
    config: ProjectConfig,
    *,
    rounds: tuple[int, ...] | None = None,
    resume: bool = False,
    min_free_gb: float = 0.0,
    threads: int | None = None,
) -> dict[str, Any]:
    """Run selected pre-registered rounds and atomically update the research ledger."""

    if min_free_gb < 0:
        raise ValueError("min_free_gb must be non-negative")
    if threads is not None and threads <= 0:
        raise ValueError("threads must be positive")

    plan = research_rounds()
    by_number = {item.number: item for item in plan}
    requested = tuple(item.number for item in plan) if rounds is None else rounds
    unknown = sorted(set(requested) - set(by_number))
    if unknown:
        raise ValueError(f"unknown research rounds: {unknown}")

    run_config = config
    if threads is not None:
        run_config = replace(config, runtime=replace(config.runtime, threads=threads))

    ledger_path = config.paths.artifacts_dir / "research" / "research_20.json"
    ledger = _load_ledger(ledger_path, plan)
    ledger["runtime"] = {
        "min_free_gb": min_free_gb,
        "threads": run_config.runtime.threads,
    }
    _write_ledger(ledger_path, ledger)

    for number in requested:
        item = by_number[number]
        record = ledger["rounds"][str(number)]
        if resume and record.get("status") == "completed":
            print(f"[SKIP] round {number:02d}: {item.code}", flush=True)
            continue
        _check_free_memory(min_free_gb)
        print(
            f"[START] round {number:02d}/20 ({item.phase}): {item.code}",
            flush=True,
        )
        record["status"] = "running"
        record["started_at"] = _timestamp()
        record.pop("error", None)
        _write_ledger(ledger_path, ledger)
        try:
            run_id, metrics = train_oof(
                run_config,
                resume=resume,
                seeds=(97,),
                **item.arguments,
            )
        except Exception as error:
            record["status"] = "failed"
            record["completed_at"] = _timestamp()
            record["error"] = f"{type(error).__name__}: {error}"
            _write_ledger(ledger_path, ledger)
            raise
        record.update(
            {
                "status": "completed",
                "completed_at": _timestamp(),
                "run_id": run_id,
                "overall_cosine": metrics["overall"]["global_score"],
                "fold_cosines": {
                    name: report["global_score"]
                    for name, report in metrics["folds"].items()
                },
                "baseline_gate": metrics.get("baseline_gate"),
            }
        )
        _write_ledger(ledger_path, ledger)
        print(
            f"[DONE] round {number:02d}: {run_id}, "
            f"OOF={record['overall_cosine']:.6f}",
            flush=True,
        )

    return ledger


def _load_ledger(path: Path, plan: tuple[ResearchRound, ...]) -> dict[str, Any]:
    if path.is_file():
        ledger = json.loads(path.read_text(encoding="utf-8"))
        if ledger.get("plan_version") != 1:
            raise ValueError("research ledger plan version mismatch")
        return ledger
    return {
        "plan_version": 1,
        "created_at": _timestamp(),
        "rounds": {
            str(item.number): {
                **asdict(item),
                "status": "pending",
            }
            for item in plan
        },
    }


def _write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    atomic_write_json(path, ledger)


def _check_free_memory(min_free_gb: float) -> None:
    if min_free_gb == 0 or platform.system() != "Windows":
        return
    available = _windows_available_memory_gb()
    if available < min_free_gb:
        raise RuntimeError(
            f"available memory {available:.1f} GB is below required {min_free_gb:.1f} GB"
        )


def _windows_available_memory_gb() -> float:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.ullAvailPhys / (1024**3)


def _timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
