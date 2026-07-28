from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mscapital.artifacts import (
    atomic_save_npy,
    atomic_write_json,
    atomic_write_parquet,
    fingerprint,
)
from mscapital.config import ProjectConfig
from mscapital.contracts import Split, TableName
from mscapital.data.canonical import CanonicalStore
from mscapital.deep_learning.dataset import SequenceDataset, ShardBatchSampler
from mscapital.deep_learning.model import (
    MSCapitalSequenceModel,
    SequenceModelConfig,
    competition_loss,
)
from mscapital.deep_learning.sequences import SequenceManifest, SequenceStore
from mscapital.training.tabular import load_labels
from mscapital.validation.metrics import cosine_report
from mscapital.validation.splits import folds_from_config

DEEP_TRAINER_VERSION = 1


def model_config_from_manifest(
    config: ProjectConfig,
    manifest: SequenceManifest,
    *,
    smoke: bool = False,
) -> SequenceModelConfig:
    hidden = 32 if smoke else config.deep_learning.hidden_size
    layers = 1 if smoke else config.deep_learning.attention_layers
    heads = min(config.deep_learning.attention_heads, hidden)
    while hidden % heads != 0:
        heads -= 1
    return SequenceModelConfig(
        market_features=len(manifest.market_features),
        transaction_features=len(manifest.transaction_features),
        transaction_grid_features=len(manifest.transaction_grid_features),
        market_steps=manifest.market_steps,
        event_steps=manifest.event_steps,
        grid_steps=manifest.grid_steps,
        hidden_size=hidden,
        attention_layers=layers,
        attention_heads=heads,
        dropout=config.deep_learning.dropout,
        gradient_checkpointing=(
            config.deep_learning.gradient_checkpointing and not smoke
        ),
    )


def train_deep_oof(
    config: ProjectConfig,
    *,
    resume: bool = False,
    seed: int = 17,
    fold_names: tuple[str, ...] | None = None,
    epochs: int | None = None,
    device: str = "auto",
    max_samples: int | None = None,
) -> tuple[str, dict]:
    sequence_dir, sequence_manifest = SequenceStore(config).manifest(
        Split.TRAIN, max_samples=max_samples
    )
    del sequence_dir
    sample_ids, months, target = load_labels(config, max_samples=max_samples)
    if len(sample_ids) != sequence_manifest.sample_count:
        raise ValueError("label and sequence cache row counts do not match")
    dataset = SequenceDataset(
        config,
        Split.TRAIN,
        max_samples=max_samples,
        target=target,
        months=months,
    )
    model_config = model_config_from_manifest(config, sequence_manifest)
    selected_folds = [
        fold
        for fold in folds_from_config(config.folds)
        if fold_names is None or fold.name in fold_names
    ]
    if not selected_folds:
        raise ValueError("no validation folds were selected")
    if fold_names is not None and {fold.name for fold in selected_folds} != set(
        fold_names
    ):
        raise ValueError("one or more requested folds do not exist")
    run_epochs = config.deep_learning.epochs if epochs is None else epochs
    if run_epochs <= 0:
        raise ValueError("epochs must be positive")
    run_payload = {
        "kind": "deep_market_transaction_fusion",
        "trainer_version": DEEP_TRAINER_VERSION,
        "sequence_digest": sequence_manifest.content_digest,
        "sequence_scope": sequence_manifest.scope,
        "model": model_config.to_dict(),
        "training": {
            **asdict(config.deep_learning),
            "epochs": run_epochs,
            "seed": seed,
        },
        "folds": [
            asdict(fold)
            for fold in config.folds
            if fold.name in {f.name for f in selected_folds}
        ],
    }
    run_id = f"deep-{fingerprint(run_payload)}"
    run_dir = config.paths.artifacts_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "config.json", run_payload)

    resolved_device = resolve_device(device)
    oof = np.full(len(target), np.nan, dtype=np.float32)
    fold_labels = np.full(len(target), "", dtype="<U32")
    fold_reports: dict[str, dict] = {}
    model_records: list[dict] = []
    for fold in selected_folds:
        train_index, valid_index = fold.split(months)
        model_dir = run_dir / "models" / fold.name / f"seed_{seed}"
        checkpoint_path = model_dir / "best.pt"
        prediction_path = model_dir / "valid_prediction.npy"
        metadata_path = model_dir / "metadata.json"
        if resume and all(
            path.is_file() for path in (checkpoint_path, prediction_path, metadata_path)
        ):
            prediction = np.load(prediction_path, allow_pickle=False)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            model, prediction, metadata = _fit_with_oom_retry(
                config,
                dataset,
                train_index,
                valid_index,
                model_config,
                seed=seed,
                epochs=run_epochs,
                device=resolved_device,
            )
            _atomic_torch_save(
                checkpoint_path,
                {
                    "model_state": model.state_dict(),
                    "model_config": model_config.to_dict(),
                    "metadata": metadata,
                },
            )
            atomic_save_npy(prediction_path, prediction.astype(np.float32))
            atomic_write_json(metadata_path, metadata)
            del model
        if prediction.shape != (len(valid_index),):
            raise ValueError(f"invalid prediction shape for {fold.name}")
        oof[valid_index] = prediction
        fold_labels[valid_index] = fold.name
        report = _report(target[valid_index], prediction, months[valid_index])
        fold_reports[fold.name] = report
        model_records.append(
            {
                "fold": fold.name,
                "seed": seed,
                "checkpoint": checkpoint_path.relative_to(run_dir).as_posix(),
                "metadata": metadata_path.relative_to(run_dir).as_posix(),
            }
        )

    covered = np.isfinite(oof)
    metrics = {
        "run_id": run_id,
        "overall": _report(target[covered], oof[covered], months[covered]),
        "folds": fold_reports,
        "fold_mean": float(
            np.mean([report["global_score"] for report in fold_reports.values()])
        ),
        "covered_rows": int(np.count_nonzero(covered)),
        "parameter_count": MSCapitalSequenceModel(model_config).parameter_count,
        "device": str(resolved_device),
    }
    oof_table = pa.table(
        {
            "sample_id": pa.array(sample_ids[covered], type=pa.int32()),
            "month": pa.array(months[covered], type=pa.int16()),
            "target": pa.array(target[covered], type=pa.float32()),
            "prediction": pa.array(oof[covered], type=pa.float32()),
            "fold": pa.array(fold_labels[covered]),
        }
    )
    atomic_write_parquet(
        run_dir / "oof.parquet",
        oof_table,
        compression=config.cache.parquet_compression,
    )
    atomic_write_json(run_dir / "metrics.json", metrics)
    atomic_write_json(
        run_dir / "manifest.json",
        {
            **run_payload,
            "run_id": run_id,
            "models": model_records,
            "oof_path": "oof.parquet",
            "metrics_path": "metrics.json",
            "test_prediction_path": None,
        },
    )
    dataset.close()
    return run_id, metrics


def predict_deep_test(
    config: ProjectConfig,
    run_id: str,
    *,
    resume: bool = False,
    device: str = "auto",
) -> Path:
    run_dir = config.paths.artifacts_dir / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != "deep_market_transaction_fusion":
        raise ValueError(f"run is not a deep sequence model: {run_id}")
    output = run_dir / "test_prediction.npy"
    if resume and output.is_file():
        values = np.load(output, allow_pickle=False)
        if (
            values.shape == (config.dataset.test_sample_count,)
            and np.isfinite(values).all()
        ):
            return output
    SequenceStore(config).manifest(Split.TEST)
    dataset = SequenceDataset(config, Split.TEST)
    model_config = SequenceModelConfig(**manifest["model"])
    resolved_device = resolve_device(device)
    predictions = []
    for record in manifest["models"]:
        model = MSCapitalSequenceModel(model_config).to(resolved_device)
        checkpoint_data = torch.load(
            run_dir / record["checkpoint"],
            map_location=resolved_device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint_data["model_state"])
        predictions.append(
            _predict_dataset(
                model,
                dataset,
                np.arange(len(dataset), dtype=np.int64),
                batch_size=config.deep_learning.physical_batch_size,
                config=config,
                seed=int(record["seed"]),
                device=resolved_device,
            )
        )
        del model
    prediction = np.mean(predictions, axis=0).astype(np.float32)
    if not np.isfinite(prediction).all() or np.linalg.norm(prediction) == 0:
        raise ValueError("deep test prediction must be finite and have non-zero norm")
    atomic_save_npy(output, prediction)
    manifest["test_prediction_path"] = output.name
    atomic_write_json(manifest_path, manifest)
    dataset.close()
    return output


def smoke_deep(
    config: ProjectConfig,
    *,
    max_samples: int = 1000,
    seed: int = 17,
    device: str = "auto",
    resume: bool = True,
    max_train_batches: int | None = None,
    full_model: bool = False,
) -> tuple[Path, dict]:
    if max_samples < 8:
        raise ValueError("deep smoke test requires at least 8 samples")
    canonical = CanonicalStore(config)
    for table in (TableName.MARKET, TableName.TRANSACTION):
        canonical.build(Split.TRAIN, table, resume=resume, max_samples=max_samples)
    store = SequenceStore(config, canonical)
    sequence_manifest = store.build(Split.TRAIN, resume=resume, max_samples=max_samples)
    _, months, target = load_labels(config, max_samples=max_samples)
    dataset = SequenceDataset(
        config,
        Split.TRAIN,
        max_samples=max_samples,
        target=target,
        months=months,
    )
    rng = np.random.default_rng(seed)
    indices = np.arange(max_samples, dtype=np.int64)
    rng.shuffle(indices)
    valid_rows = max(1, round(max_samples * 0.2))
    valid_index = np.sort(indices[:valid_rows])
    train_index = np.sort(indices[valid_rows:])
    model_config = model_config_from_manifest(
        config, sequence_manifest, smoke=not full_model
    )
    smoke_config = replace(
        config,
        deep_learning=replace(
            config.deep_learning,
            physical_batch_size=min(config.deep_learning.physical_batch_size, 16),
            effective_batch_size=min(config.deep_learning.effective_batch_size, 64),
            num_workers=0,
            epochs=1,
            early_stopping_patience=1,
            gradient_checkpointing=False,
        ),
    )
    resolved_device = resolve_device(device)
    model, prediction, metadata = _fit_with_oom_retry(
        smoke_config,
        dataset,
        train_index,
        valid_index,
        model_config,
        seed=seed,
        epochs=1,
        device=resolved_device,
        max_train_batches=max_train_batches,
    )
    identity = fingerprint(
        {
            "kind": "real_data_deep_smoke",
            "sequence": sequence_manifest.content_digest,
            "model": model_config.to_dict(),
            "seed": seed,
            "max_train_batches": max_train_batches,
            "full_model": full_model,
        }
    )
    output_dir = config.paths.artifacts_dir / "smoke" / f"deep-{identity}"
    checkpoint_path = output_dir / "model.pt"
    _atomic_torch_save(
        checkpoint_path,
        {
            "model_state": model.state_dict(),
            "model_config": model_config.to_dict(),
            "metadata": metadata,
        },
    )
    reloaded = MSCapitalSequenceModel(model_config).to(resolved_device)
    checkpoint_data = torch.load(
        checkpoint_path, map_location=resolved_device, weights_only=True
    )
    reloaded.load_state_dict(checkpoint_data["model_state"])
    reloaded_prediction = _predict_dataset(
        reloaded,
        dataset,
        valid_index,
        batch_size=smoke_config.deep_learning.physical_batch_size,
        config=smoke_config,
        seed=seed,
        device=resolved_device,
    )
    if not np.allclose(prediction, reloaded_prediction, rtol=1e-5, atol=1e-6):
        raise ValueError("reloaded smoke checkpoint changed predictions")
    report = {
        "kind": "real_data_deep_smoke",
        "samples": max_samples,
        "train_rows": len(train_index),
        "valid_rows": len(valid_index),
        "random_split_for_smoke_only": True,
        "valid_cosine": _safe_cosine(target[valid_index], prediction),
        "valid_mse": float(np.mean(np.square(target[valid_index] - prediction))),
        "parameter_count": model.parameter_count,
        "full_model": full_model,
        "device": str(resolved_device),
        "sequence_digest": sequence_manifest.content_digest,
        "checkpoint": str(checkpoint_path),
        "training": metadata,
    }
    atomic_save_npy(output_dir / "valid_prediction.npy", prediction)
    atomic_write_json(output_dir / "report.json", report)
    dataset.close()
    return output_dir, report


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def _fit_with_oom_retry(
    config: ProjectConfig,
    dataset: SequenceDataset,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    model_config: SequenceModelConfig,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    max_train_batches: int | None = None,
) -> tuple[MSCapitalSequenceModel, np.ndarray, dict]:
    batch_size = config.deep_learning.physical_batch_size
    while batch_size >= config.deep_learning.minimum_batch_size:
        try:
            return _fit_once(
                config,
                dataset,
                train_index,
                valid_index,
                model_config,
                seed=seed,
                epochs=epochs,
                device=device,
                batch_size=batch_size,
                max_train_batches=max_train_batches,
            )
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if (
                not _is_oom(exc)
                or batch_size // 2 < config.deep_learning.minimum_batch_size
            ):
                raise
            batch_size //= 2
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[OOM] retrying fold with physical batch size {batch_size}")
    raise RuntimeError("no viable physical batch size remained")


def _fit_once(
    config: ProjectConfig,
    dataset: SequenceDataset,
    train_index: np.ndarray,
    valid_index: np.ndarray,
    model_config: SequenceModelConfig,
    *,
    seed: int,
    epochs: int,
    device: torch.device,
    batch_size: int,
    max_train_batches: int | None,
) -> tuple[MSCapitalSequenceModel, np.ndarray, dict]:
    _seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = MSCapitalSequenceModel(model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.deep_learning.learning_rate,
        weight_decay=config.deep_learning.weight_decay,
    )
    train_sampler = ShardBatchSampler(
        dataset, train_index, batch_size, shuffle=True, seed=seed
    )
    valid_sampler = ShardBatchSampler(
        dataset, valid_index, batch_size, shuffle=False, seed=seed
    )
    train_loader = _loader(dataset, train_sampler, config, device)
    valid_loader = _loader(dataset, valid_sampler, config, device)
    accumulation = max(
        1, math.ceil(config.deep_learning.effective_batch_size / batch_size)
    )
    batches_per_epoch = len(train_loader)
    if max_train_batches is not None:
        batches_per_epoch = min(batches_per_epoch, max_train_batches)
    total_steps = max(1, math.ceil(batches_per_epoch / accumulation) * epochs)
    warmup_steps = round(total_steps * config.deep_learning.warmup_ratio)
    scheduler = LambdaLR(
        optimizer,
        _warmup_cosine_lambda(total_steps, warmup_steps),
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_score = -math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    best_prediction: np.ndarray | None = None
    history = []
    stale_epochs = 0
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch_number, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_number >= max_train_batches:
                break
            moved = _move_batch(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=amp_enabled,
            ):
                prediction = _forward(model, moved)
                loss, _ = competition_loss(
                    prediction,
                    moved["target"],
                    config.deep_learning.cosine_loss_weight,
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            should_step = (
                batch_number + 1
            ) % accumulation == 0 or batch_number + 1 == batches_per_epoch
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.deep_learning.gradient_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            losses.append(float(loss.detach().cpu()))
        valid_prediction = _predict_loader(model, valid_loader, device)
        valid_target = np.asarray(dataset.target)[valid_index]
        score = _safe_cosine(valid_target, valid_prediction)
        valid_mse = float(np.mean(np.square(valid_target - valid_prediction)))
        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "valid_mse": valid_mse,
            "valid_cosine": score,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        print(
            f"[epoch {epoch + 1}/{epochs}] loss={epoch_record['train_loss']:.8f} "
            f"valid_mse={valid_mse:.8f} cosine={score:.6f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_prediction = valid_prediction.copy()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.deep_learning.early_stopping_patience:
                break
    if best_state is None or best_prediction is None:
        raise RuntimeError("deep training did not produce a checkpoint")
    model.load_state_dict(best_state)
    peak_vram_gb = (
        float(torch.cuda.max_memory_allocated(device) / 1024**3)
        if device.type == "cuda"
        else 0.0
    )
    if peak_vram_gb > config.deep_learning.max_vram_gb:
        raise RuntimeError(
            f"peak VRAM {peak_vram_gb:.2f} GB exceeds configured limit "
            f"{config.deep_learning.max_vram_gb:.2f} GB"
        )
    metadata = {
        "best_epoch": best_epoch,
        "best_valid_cosine": best_score,
        "physical_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": batch_size * accumulation,
        "peak_vram_gb": peak_vram_gb,
        "device": str(device),
        "amp": amp_enabled,
        "train_rows": len(train_index),
        "valid_rows": len(valid_index),
        "history": history,
    }
    return model, best_prediction, metadata


def _predict_dataset(
    model: MSCapitalSequenceModel,
    dataset: SequenceDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    config: ProjectConfig,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    sampler = ShardBatchSampler(dataset, indices, batch_size, shuffle=False, seed=seed)
    return _predict_loader(model, _loader(dataset, sampler, config, device), device)


def _predict_loader(
    model: MSCapitalSequenceModel,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    predictions = []
    with torch.inference_mode():
        for batch in loader:
            moved = _move_batch(batch, device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                prediction = _forward(model, moved)
            predictions.append(prediction.float().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


def _forward(
    model: MSCapitalSequenceModel, batch: dict[str, torch.Tensor]
) -> torch.Tensor:
    return model(
        batch["market_values"],
        batch["market_mask"],
        batch["transaction_values"],
        batch["transaction_side"],
        batch["transaction_mask"],
        batch["transaction_grid"],
    )


def _move_batch(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda")
        for name, value in batch.items()
    }


def _loader(
    dataset: SequenceDataset,
    sampler: ShardBatchSampler,
    config: ProjectConfig,
    device: torch.device,
) -> DataLoader:
    workers = config.deep_learning.num_workers
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )


def _warmup_cosine_lambda(total_steps: int, warmup_steps: int):
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-3)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return schedule


def _report(target: np.ndarray, prediction: np.ndarray, months: np.ndarray) -> dict:
    return asdict(cosine_report(target, prediction, months))


def _safe_cosine(target: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(target, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    denominator = np.linalg.norm(truth) * np.linalg.norm(predicted)
    if denominator == 0:
        return 0.0
    return float(np.dot(truth, predicted) / denominator)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _is_oom(error: BaseException) -> bool:
    return (
        isinstance(error, torch.OutOfMemoryError)
        or "out of memory" in str(error).lower()
    )


def _atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
