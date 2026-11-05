"""Load test over real HTTP: does micro-batching actually help, and at what latency cost?

Starts the service as a subprocess for each configuration, then drives it with a closed-loop client:
each simulated camera sends its next image as soon as the previous answer arrives. The client runs
on the same machine and shares its CPU, so absolute numbers are conservative.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

logger = logging.getLogger(__name__)
REPORTS = Path("reports")
CONCURRENCY_LEVELS = (1, 4, 16, 32)
CONFIGURATIONS = (
    {"name": "no-batching", "max_batch_size": 1, "max_wait_ms": 0.0},
    {"name": "micro-batching", "max_batch_size": 16, "max_wait_ms": 5.0},
)
PORT = 8765


def _sample_images(data_root: Path, category: str, count: int = 32) -> list[bytes]:
    paths = sorted((data_root / category / "Data" / "Images" / "Normal").glob("*"))[:count]
    if not paths:
        raise FileNotFoundError(f"No images found for {category} under {data_root}")
    return [path.read_bytes() for path in paths]


async def _camera(
    client: httpx.AsyncClient,
    url: str,
    images: list[bytes],
    deadline: float,
    latencies: list[float],
    errors: list[int],
    seed: int,
) -> None:
    position = seed
    while time.perf_counter() < deadline:
        data = images[position % len(images)]
        position += 1
        started = time.perf_counter()
        response = await client.post(url, files={"file": ("image.jpg", data, "image/jpeg")})
        if response.status_code == 200:
            latencies.append(time.perf_counter() - started)
        else:
            errors.append(response.status_code)


async def _drive(
    url: str, images: list[bytes], concurrency: int, duration: float
) -> tuple[list[float], list[int], float]:
    latencies: list[float] = []
    errors: list[int] = []
    timeout = httpx.Timeout(120.0)
    limits = httpx.Limits(max_connections=concurrency + 2)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        started = time.perf_counter()
        deadline = started + duration
        await asyncio.gather(
            *(
                _camera(client, url, images, deadline, latencies, errors, seed)
                for seed in range(concurrency)
            )
        )
        elapsed = time.perf_counter() - started
    return latencies, errors, elapsed


def _wait_ready(base: str, process: subprocess.Popen, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited early with code {process.returncode}")
        try:
            if httpx.get(f"{base}/readyz", timeout=2).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(0.5)
    raise TimeoutError("Service did not become ready.")


def run(bundle: Path, data_root: Path, category: str, duration: float) -> None:
    images = _sample_images(data_root, category)
    base = f"http://127.0.0.1:{PORT}"
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as temp:
        for configuration in CONFIGURATIONS:
            for concurrency in CONCURRENCY_LEVELS:
                environment = {
                    **os.environ,
                    "PATCHSENTRY_BUNDLE_DIR": str(bundle),
                    "PATCHSENTRY_CATEGORIES": json.dumps([category]),
                    "PATCHSENTRY_MAX_BATCH_SIZE": str(configuration["max_batch_size"]),
                    "PATCHSENTRY_MAX_WAIT_MS": str(configuration["max_wait_ms"]),
                    "PATCHSENTRY_DATABASE_URL": f"sqlite:///{temp}/{time.time_ns()}.db",
                    "PATCHSENTRY_LOG_LEVEL": "WARNING",
                }
                process = subprocess.Popen(
                    [sys.executable, "-m", "patchsentry.serving", "--port", str(PORT)],
                    env=environment,
                )
                try:
                    _wait_ready(base, process)
                    url = f"{base}/v1/inspect/{category}"
                    # Warm up so that first-call costs do not land in the measurement.
                    asyncio.run(_drive(url, images, concurrency, 5.0))
                    before = httpx.get(f"{base}/metrics", timeout=10).text
                    latencies, errors, elapsed = asyncio.run(
                        _drive(url, images, concurrency, duration)
                    )
                    after = httpx.get(f"{base}/metrics", timeout=10).text
                finally:
                    process.terminate()
                    process.wait(timeout=30)
                values = np.asarray(latencies) * 1000
                sum_delta = _counter(after, "sum") - _counter(before, "sum")
                count_delta = _counter(after, "count") - _counter(before, "count")
                row = {
                    "configuration": configuration["name"],
                    "max_batch_size": configuration["max_batch_size"],
                    "max_wait_ms": configuration["max_wait_ms"],
                    "concurrent_clients": concurrency,
                    "requests": int(values.size),
                    "errors": len(errors),
                    "requests_per_second": values.size / elapsed,
                    "latency_p50_ms": float(np.quantile(values, 0.5)),
                    "latency_p95_ms": float(np.quantile(values, 0.95)),
                    "latency_p99_ms": float(np.quantile(values, 0.99)),
                    "mean_batch_size": sum_delta / count_delta if count_delta else None,
                }
                rows.append(row)
                logger.info("%s", row)
                REPORTS.mkdir(exist_ok=True)
                (REPORTS / "loadtest.json").write_text(
                    json.dumps(
                        {"category": category, "duration_s": duration, "rows": rows}, indent=2
                    )
                    + "\n"
                )


def _counter(text: str, suffix: str) -> float:
    match = re.search(rf"patchsentry_batch_size_{suffix} ([\d.e+]+)", text)
    return float(match.group(1)) if match else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/bundle"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/visa"))
    parser.add_argument("--category", default="pcb1")
    parser.add_argument("--duration", type=float, default=30.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.bundle, args.data_root, args.category, args.duration)


if __name__ == "__main__":
    main()
