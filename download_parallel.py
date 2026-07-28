from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
from kagglehub.clients import build_kaggle_client
from kagglesdk.competitions.types.competition_api_service import ApiDownloadDataFileRequest


COMPETITION = "ms-capital-real-financial-market-forecasting"
LIST_URL = f"https://www.kaggle.com/api/v1/competitions/data/list/{COMPETITION}"
MIB = 1024 * 1024
_thread_local = threading.local()


@dataclass(frozen=True)
class FileSpec:
    name: str
    size: int
    output: Path
    parts_dir: Path


class SignedUrlProvider:
    def __init__(self, file_name: str, refresh_seconds: int = 480) -> None:
        self.file_name = file_name
        self.refresh_seconds = refresh_seconds
        self._url = ""
        self._fetched_at = 0.0
        self._lock = threading.Lock()

    def get(self, force: bool = False) -> str:
        with self._lock:
            stale = time.monotonic() - self._fetched_at > self.refresh_seconds
            if force or not self._url or stale:
                request = ApiDownloadDataFileRequest()
                request.competition_name = COMPETITION
                request.file_name = self.file_name
                with build_kaggle_client() as client:
                    response = client.competitions.competition_api_client.download_data_file(request)
                    self._url = response.url
                    response.close()
                self._fetched_at = time.monotonic()
            return self._url


def get_http_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.trust_env = False
        session.headers.update({"User-Agent": "mscapital-parallel-downloader/1.0"})
        _thread_local.session = session
    return session


def get_manifest(token: str, output_dir: Path) -> list[FileSpec]:
    response = requests.get(
        LIST_URL,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    files = response.json()["files"]
    specs = []
    for item in files:
        name = item["name"]
        safe_name = name.replace("/", "__").replace("\\", "__")
        specs.append(
            FileSpec(
                name=name,
                size=int(item["totalBytes"]),
                output=output_dir / Path(name),
                parts_dir=output_dir / ".parts" / safe_name,
            )
        )
    return specs


def expected_part_size(spec: FileSpec, part_index: int, chunk_size: int) -> int:
    start = part_index * chunk_size
    return min(chunk_size, spec.size - start)


def download_part(
    spec: FileSpec,
    part_index: int,
    chunk_size: int,
    provider: SignedUrlProvider,
    retries: int = 8,
) -> int:
    start = part_index * chunk_size
    expected = expected_part_size(spec, part_index, chunk_size)
    end = start + expected - 1
    part_path = spec.parts_dir / f"{part_index:06d}.part"
    if part_path.exists() and part_path.stat().st_size == expected:
        return 0

    spec.parts_dir.mkdir(parents=True, exist_ok=True)
    temp_path = part_path.with_suffix(".tmp")
    for attempt in range(retries):
        try:
            url = provider.get(force=attempt > 0 and attempt % 3 == 0)
            with get_http_session().get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(60, 300),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"range request returned HTTP {response.status_code}")
                with temp_path.open("wb") as output:
                    for block in response.iter_content(256 * 1024):
                        if block:
                            output.write(block)
            actual = temp_path.stat().st_size
            if actual != expected:
                raise RuntimeError(f"part size {actual} != {expected}")
            os.replace(temp_path, part_path)
            return expected
        except Exception as exc:  # noqa: BLE001 - network retries need broad handling
            if temp_path.exists():
                temp_path.unlink()
            if attempt == retries - 1:
                raise RuntimeError(f"failed {spec.name} part {part_index}: {exc}") from exc
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def completed_part_bytes(spec: FileSpec, chunk_size: int) -> int:
    if not spec.parts_dir.exists():
        return 0
    total = 0
    part_count = (spec.size + chunk_size - 1) // chunk_size
    for index in range(part_count):
        path = spec.parts_dir / f"{index:06d}.part"
        expected = expected_part_size(spec, index, chunk_size)
        if path.exists() and path.stat().st_size == expected:
            total += expected
    return total


def write_status(path: Path, payload: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temp, path)


def assemble(spec: FileSpec, chunk_size: int) -> None:
    spec.output.parent.mkdir(parents=True, exist_ok=True)
    assembling = spec.output.with_suffix(spec.output.suffix + ".assembling")
    part_count = (spec.size + chunk_size - 1) // chunk_size
    with assembling.open("wb") as output:
        for index in range(part_count):
            part_path = spec.parts_dir / f"{index:06d}.part"
            expected = expected_part_size(spec, index, chunk_size)
            if not part_path.exists() or part_path.stat().st_size != expected:
                raise RuntimeError(f"missing or invalid part: {part_path}")
            with part_path.open("rb") as part:
                while block := part.read(4 * MIB):
                    output.write(block)
    if assembling.stat().st_size != spec.size:
        raise RuntimeError(f"assembled size mismatch for {spec.name}")
    os.replace(assembling, spec.output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--chunk-mib", type=int, default=1)
    args = parser.parse_args()

    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN is required")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir.parent / "download_status.json"
    chunk_size = args.chunk_mib * MIB
    specs = get_manifest(token, output_dir)

    pending_specs: list[FileSpec] = []
    for spec in specs:
        if spec.output.exists() and spec.output.stat().st_size == spec.size:
            print(f"complete: {spec.name} ({spec.size:,} bytes)", flush=True)
            continue
        if spec.output.exists():
            legacy = spec.output.with_suffix(spec.output.suffix + ".legacy-partial")
            os.replace(spec.output, legacy)
            print(f"moved incomplete file to {legacy}", flush=True)
        pending_specs.append(spec)

    if not pending_specs:
        write_status(status_path, {"state": "complete", "files": len(specs)})
        print("all files already complete", flush=True)
        return

    providers = {spec.name: SignedUrlProvider(spec.name) for spec in pending_specs}
    task_rows: list[tuple[FileSpec, int]] = []
    max_parts = max((spec.size + chunk_size - 1) // chunk_size for spec in pending_specs)
    for index in range(max_parts):
        for spec in pending_specs:
            if index * chunk_size < spec.size:
                task_rows.append((spec, index))

    total_bytes = sum(spec.size for spec in specs)
    complete_files_bytes = sum(
        spec.size for spec in specs if spec.output.exists() and spec.output.stat().st_size == spec.size
    )
    resumed_bytes = sum(completed_part_bytes(spec, chunk_size) for spec in pending_specs)
    downloaded_bytes = complete_files_bytes + resumed_bytes
    started_at = time.monotonic()
    print(
        f"starting {len(task_rows):,} chunks with {args.workers} workers; "
        f"already have {downloaded_bytes / MIB:.1f} MiB",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_part, spec, index, chunk_size, providers[spec.name]): (spec, index)
            for spec, index in task_rows
        }
        last_report = 0.0
        for future in as_completed(futures):
            downloaded_bytes += future.result()
            now = time.monotonic()
            if now - last_report >= 10:
                elapsed = max(now - started_at, 0.001)
                newly_downloaded = max(downloaded_bytes - complete_files_bytes - resumed_bytes, 0)
                speed = newly_downloaded / elapsed
                remaining = max(total_bytes - downloaded_bytes, 0)
                eta_seconds = remaining / speed if speed > 0 else None
                payload = {
                    "state": "downloading",
                    "downloaded_bytes": downloaded_bytes,
                    "total_bytes": total_bytes,
                    "percent": round(downloaded_bytes / total_bytes * 100, 3),
                    "speed_mib_s": round(speed / MIB, 3),
                    "eta_seconds": round(eta_seconds) if eta_seconds is not None else None,
                    "workers": args.workers,
                }
                write_status(status_path, payload)
                print(json.dumps(payload), flush=True)
                last_report = now

    print("all chunks downloaded; assembling files", flush=True)
    for spec in pending_specs:
        assemble(spec, chunk_size)
        print(f"assembled: {spec.name}", flush=True)

    invalid = [spec.name for spec in specs if not spec.output.exists() or spec.output.stat().st_size != spec.size]
    if invalid:
        raise RuntimeError(f"final validation failed: {invalid}")
    write_status(
        status_path,
        {"state": "complete", "files": len(specs), "total_bytes": total_bytes, "workers": args.workers},
    )
    print(f"download complete: {len(specs)} files, {total_bytes:,} bytes", flush=True)


if __name__ == "__main__":
    main()
