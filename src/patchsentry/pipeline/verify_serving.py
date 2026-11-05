"""Run the whole test set through the exact code the service uses (ONNX Runtime and FAISS).

Training measures metrics with PyTorch features and GPU search; the service uses neither. This check
closes that gap: it reports the metrics of the deployed path, not of a training-time approximation.
It must not import PyTorch.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from patchsentry import metrics
from patchsentry.data.visa import CATEGORIES, load_category
from patchsentry.models.patchcore_index import upsample_maps
from patchsentry.serving.engine import InferenceEngine

logger = logging.getLogger(__name__)
REPORTS = Path("reports")


def _verify(engine: InferenceEngine, data: Any, category: str, batch_size: int) -> dict[str, float]:
    scores: list[float] = []
    maps: list[np.ndarray] = []
    decisions: list[bool] = []
    for start in range(0, len(data.test), batch_size):
        batch = data.test[start : start + batch_size]
        results = engine.infer_batch(batch, [category] * len(batch))
        scores.extend(result.score for result in results)
        maps.extend(result.patch_map for result in results)
        decisions.extend(result.is_anomalous for result in results)
    decision_array = np.array(decisions, dtype=np.bool_)
    pixel_maps = upsample_maps(np.stack(maps), engine.image_size, engine.gaussian_sigma)
    labels = data.test_labels
    return {
        "image_auroc": metrics.auroc(labels, np.asarray(scores)),
        "aupro": metrics.aupro(data.test_masks, pixel_maps),
        "false_alarm_rate": float(decision_array[~labels].mean()),
        "recall": float(decision_array[labels].mean()),
    }


def run(bundle: Path, data_root: Path, categories: tuple[str, ...], batch_size: int) -> None:
    manifest = json.loads((bundle / "manifest.json").read_text())
    config = manifest["config"]
    rows: list[dict[str, Any]] = []
    for use_quantized in (False, True):
        engine = InferenceEngine(bundle, categories=list(categories), use_quantized=use_quantized)
        for category in categories:
            data = load_category(
                data_root,
                category,
                engine.image_size,
                config["calibration_fraction"],
                config["seed"],
            )
            row = {
                "category": category,
                "model": "quantized" if use_quantized else "float",
                **_verify(engine, data, category, batch_size),
            }
            rows.append(row)
            logger.info("%s", row)
    REPORTS.mkdir(exist_ok=True)
    payload = {"alpha": manifest["default_alpha"], "rows": rows}
    (REPORTS / "serving_verification.json").write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/bundle"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/visa"))
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.bundle, args.data_root, tuple(args.categories), args.batch_size)


if __name__ == "__main__":
    main()
