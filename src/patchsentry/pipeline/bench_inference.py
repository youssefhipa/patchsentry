"""Latency and throughput benchmarks, run on real images.

Two parts, in separate processes because FAISS and PyTorch cannot share one on macOS:
- ``--part backbone``: PyTorch against ONNX Runtime (float, quantised, CoreML) by batch size.
- ``--part index``: FAISS flat against HNSW against IVF: speed, recall and effect on AUROC.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from patchsentry import metrics
from patchsentry.data.preprocessing import normalize_batch
from patchsentry.data.visa import load_category
from patchsentry.models.patchcore_index import FAISS_INDEX_KINDS, PatchCoreIndex

logger = logging.getLogger(__name__)
REPORTS = Path("reports")
BATCH_SIZES = (1, 8, 16, 32)
MIN_IMAGES = 160
WARMUP_RUNS = 3


def _machine() -> dict[str, str]:
    brand = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
    ).stdout.strip()
    return {"processor": brand or platform.processor(), "system": platform.platform()}


def _measure(run_batch: Callable[[np.ndarray], Any], images: np.ndarray, batch: int) -> dict:
    data = normalize_batch(np.resize(images, (batch, *images.shape[1:])))
    for _ in range(WARMUP_RUNS):
        run_batch(data)
    timings = []
    for _ in range(max(10, MIN_IMAGES // batch)):
        started = time.perf_counter()
        run_batch(data)
        timings.append(time.perf_counter() - started)
    values = np.asarray(timings)
    return {
        "batch_size": batch,
        "images_per_second": batch / float(np.median(values)),
        "batch_latency_p50_ms": float(np.median(values) * 1000),
        "batch_latency_p95_ms": float(np.quantile(values, 0.95) * 1000),
        "runs": values.size,
    }


def bench_backbone(bundle: Path, images: np.ndarray) -> list[dict[str, Any]]:
    import torch

    from patchsentry.pipeline.common import build_extractor, config_from_manifest

    rows: list[dict[str, Any]] = []
    manifest = json.loads((bundle / "manifest.json").read_text())
    config = config_from_manifest(manifest)

    def add(name: str, run_batch: Callable[[np.ndarray], Any]) -> None:
        for batch in BATCH_SIZES:
            row = {"runtime": name, **_measure(run_batch, images, batch)}
            rows.append(row)
            logger.info("%s", row)

    def torch_runner(device: str) -> Callable[[np.ndarray], Any]:
        extractor = build_extractor(config, device)

        @torch.inference_mode()
        def run(data: np.ndarray) -> Any:
            output = extractor(torch.from_numpy(data).to(device))
            if device == "mps":
                torch.mps.synchronize()  # MPS is asynchronous; without this, timings are too low.
            return output

        return run

    def onnx_runner(path: Path, providers: list[str]) -> Callable[[np.ndarray], Any]:
        session = ort.InferenceSession(str(path), providers=providers)
        name = session.get_inputs()[0].name
        return lambda data: session.run(None, {name: data})

    add("pytorch-cpu", torch_runner("cpu"))
    if torch.backends.mps.is_available():
        add("pytorch-mps", torch_runner("mps"))
    add("onnx-cpu-float", onnx_runner(bundle / manifest["model"], ["CPUExecutionProvider"]))
    add(
        "onnx-cpu-quantized",
        onnx_runner(bundle / manifest["quantized_model"], ["CPUExecutionProvider"]),
    )
    if "CoreMLExecutionProvider" in ort.get_available_providers():
        try:
            add(
                "onnx-coreml",
                onnx_runner(
                    bundle / manifest["model"], ["CoreMLExecutionProvider", "CPUExecutionProvider"]
                ),
            )
        except Exception:
            logger.exception("CoreML provider failed; leaving it out of the report")
    return rows


def _features(bundle: Path, images: np.ndarray, batch: int = 16) -> np.ndarray:
    manifest = json.loads((bundle / "manifest.json").read_text())
    session = ort.InferenceSession(
        str(bundle / manifest["model"]), providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    outputs = [
        session.run(None, {name: normalize_batch(images[start : start + batch])})[0]
        for start in range(0, len(images), batch)
    ]
    return np.concatenate(outputs).transpose(0, 2, 3, 1)


def bench_index(bundle: Path, data_root: Path, categories: list[str]) -> list[dict[str, Any]]:
    import faiss

    manifest = json.loads((bundle / "manifest.json").read_text())
    config = manifest["config"]
    rows: list[dict[str, Any]] = []
    for category in categories:
        data = load_category(
            data_root,
            category,
            config["image_size"],
            config["calibration_fraction"],
            config["seed"],
        )
        features = _features(bundle, data.test)
        directory = bundle / "categories" / category
        reference: np.ndarray | None = None
        for kind in FAISS_INDEX_KINDS:
            started = time.perf_counter()
            index = PatchCoreIndex.load(directory, kind)
            build_seconds = time.perf_counter() - started
            timings = []
            for image_features in features[:60]:
                started = time.perf_counter()
                index.score(image_features[None])
                timings.append(time.perf_counter() - started)
            scores = np.concatenate(
                [index.score(features[i : i + 16])[0] for i in range(0, len(features), 16)]
            )
            flat = features.reshape(-1, features.shape[-1])
            _, nearest = index.index.search(np.ascontiguousarray(flat), 1)
            if reference is None:
                reference = nearest
            size_mb = faiss.serialize_index(index.index).size / 1e6
            row = {
                "category": category,
                "index": kind,
                "bank_size": index.bank.shape[0],
                "build_seconds": build_seconds,
                "search_p50_ms_per_image": float(np.median(timings) * 1000),
                "search_p95_ms_per_image": float(np.quantile(timings, 0.95) * 1000),
                "recall_at_1": float((nearest == reference).mean()),
                "image_auroc": metrics.auroc(data.test_labels, scores),
                "index_size_mb": size_mb,
            }
            rows.append(row)
            logger.info("%s", row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["backbone", "index"], required=True)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/bundle"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/visa"))
    parser.add_argument("--categories", nargs="*", default=["candle", "pcb1", "macaroni2"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    REPORTS.mkdir(exist_ok=True)
    if args.part == "backbone":
        config = manifest["config"]
        data = load_category(
            args.data_root,
            args.categories[0],
            config["image_size"],
            config["calibration_fraction"],
            config["seed"],
        )
        rows = bench_backbone(args.bundle, data.test[:32])
    else:
        rows = bench_index(args.bundle, args.data_root, args.categories)
    payload = {"machine": _machine(), "rows": rows}
    path = REPORTS / f"bench_{args.part}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
