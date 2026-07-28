from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def atomic_write_text(path: str | Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temp.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def atomic_save_npy(path: str | Path, values: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temp.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, target)


def atomic_write_parquet(
    path: str | Path,
    table: pa.Table,
    *,
    compression: str = "zstd",
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    pq.write_table(table, temp, compression=compression, use_dictionary=False)
    os.replace(temp, target)


def validate_npy(path: str | Path, shape: tuple[int, ...], dtype: np.dtype) -> bool:
    target = Path(path)
    checksum_path = target.with_suffix(target.suffix + ".sha256")
    if not target.is_file() or not checksum_path.is_file():
        return False
    try:
        values = np.load(target, mmap_mode="r", allow_pickle=False)
        metadata_ok = values.shape == shape and values.dtype == dtype
        del values
        if not metadata_ok:
            return False
        expected = checksum_path.read_text(encoding="ascii").strip()
        return bool(expected) and sha256_file(target) == expected
    except (OSError, ValueError):
        return False

