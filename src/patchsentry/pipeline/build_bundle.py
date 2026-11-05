"""Build the deployable bundle and evaluate conformal calibration and the ONNX models on the way.

The bundle holds one shared ONNX backbone (float and quantised), plus for each category a memory
bank, a neighbour table and calibration scores. Everything the service needs, and no PyTorch.

Reports:
- ``reports/conformal.json``: nominal against observed false alarm rates, and recall at those rates.
- ``reports/onnx_parity.json``: ONNX float and quantised models compared with PyTorch.
- ``reports/drift.json``: whether the p-value monitor detects simulated image degradation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from patchsentry import metrics
from patchsentry.conformal import (
    conformal_p_values,
    conformal_threshold,
    false_alarm_rate_quantile,
    uniformity_ks,
)
from patchsentry.data.preprocessing import normalize_batch
from patchsentry.data.visa import CATEGORIES, load_category
from patchsentry.models.features import extract_features
from patchsentry.models.patchcore_index import upsample_maps
from patchsentry.pipeline.common import (
    DEV_CATEGORIES,
    Config,
    build_extractor,
    compute_features,
    evaluate,
    fit_patchcore,
    pick_device,
    score_in_batches,
    setup_logging,
    write_report,
)
from patchsentry.pipeline.export import export_onnx, quantize_onnx
from patchsentry.serving.engine import MANIFEST_NAME

logger = logging.getLogger(__name__)

N_RESAMPLES = 2000
N_PARITY_IMAGES = 16


def _onnx_features(session: ort.InferenceSession, images: np.ndarray, batch: int) -> np.ndarray:
    outputs = []
    name = session.get_inputs()[0].name
    for start in range(0, len(images), batch):
        output = session.run(None, {name: normalize_batch(images[start : start + batch])})[0]
        outputs.append(output.transpose(0, 2, 3, 1))
    return np.concatenate(outputs)


def _conformal_rates(
    calibration: np.ndarray,
    test_normal: np.ndarray,
    test_anomalous: np.ndarray,
    train_in_bank: np.ndarray,
    alphas: tuple[float, ...],
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pooled = np.concatenate([calibration, test_normal])
    generator = np.random.default_rng(seed)
    for alpha in alphas:
        threshold = conformal_threshold(calibration, alpha)
        # Resampling makes calibration and test exchangeable by construction, so this estimates
        # what the theorem promises, free of any distribution shift between the official splits.
        resampled = []
        for _ in range(N_RESAMPLES):
            order = generator.permutation(pooled.size)
            cal, held = pooled[order[: calibration.size]], pooled[order[calibration.size :]]
            resampled.append(float((held > conformal_threshold(cal, alpha)).mean()))
        # What many people do: take the quantile of scores on the training images themselves.
        naive = float(np.quantile(train_in_bank, 1.0 - alpha))
        rows.append(
            {
                "alpha": alpha,
                "threshold": threshold,
                "n_calibration": int(calibration.size),
                "test_false_alarm_rate": float((test_normal > threshold).mean()),
                "test_recall": float((test_anomalous > threshold).mean()),
                "resampled_mean_false_alarm_rate": float(np.mean(resampled)),
                "resampled_p95_false_alarm_rate": float(np.quantile(resampled, 0.95)),
                "theory_p95_false_alarm_rate": false_alarm_rate_quantile(
                    calibration.size, alpha, 0.95
                ),
                "naive_train_threshold": naive,
                "naive_test_false_alarm_rate": float((test_normal > naive).mean()),
            }
        )
    return rows


def _degrade(images: np.ndarray, kind: str) -> np.ndarray:
    """Simulate plausible line problems on normal images."""
    if kind == "darker_30pct":
        return (images.astype(np.float32) * 0.7).clip(0, 255).astype(np.uint8)
    if kind == "defocus_blur":
        return np.stack([cv2.GaussianBlur(image, (0, 0), sigmaX=2.0) for image in images])
    raise ValueError(kind)


def run(
    config: Config, categories: tuple[str, ...], output: Path, quantized_for_all: bool = True
) -> None:
    device = pick_device()
    extractor = build_extractor(config, device)
    output.mkdir(parents=True, exist_ok=True)

    float_path = export_onnx(extractor, output / "backbone.onnx", config.image_size)
    extractor = extractor.to(device)

    # Calibrate quantisation on normal images from all categories: it is one shared graph.
    samples = []
    for category in categories:
        data = load_category(
            config.data_root, category, config.image_size, config.calibration_fraction, config.seed
        )
        samples.append(
            data.train[np.random.default_rng(config.seed).permutation(len(data.train))[:8]]
        )
    quant_path = quantize_onnx(float_path, output / "backbone.quant.onnx", np.concatenate(samples))
    float_session = ort.InferenceSession(str(float_path), providers=["CPUExecutionProvider"])
    quant_session = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])

    manifest: dict[str, Any] = {
        "version": f"{config.backbone}-{config.image_size}-coreset{config.coreset_fraction}",
        "model": float_path.name,
        "quantized_model": quant_path.name,
        "image_size": config.image_size,
        "gaussian_sigma": config.gaussian_sigma,
        "default_alpha": str(config.default_alpha),
        "config": config.to_params(),
        "categories": {},
    }
    conformal_rows: list[dict[str, Any]] = []
    parity_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []

    for category in categories:
        data = load_category(
            config.data_root, category, config.image_size, config.calibration_fraction, config.seed
        )
        features = compute_features(extractor, data, config, device)
        index, _ = fit_patchcore(features, config, device)
        index.save(output / "categories" / category)

        calibration_scores, _ = score_in_batches(index, features.calibration, config.batch_size)
        test_scores, test_maps = score_in_batches(index, features.test, config.batch_size)
        normal = test_scores[~data.test_labels]
        anomalous = test_scores[data.test_labels]

        # Score a sample of the images whose patches built the bank, to demonstrate the leak.
        subset = data.train[:: max(1, len(data.train) // 200)]
        train_features = extract_features(extractor, subset, config.batch_size, device)
        train_scores, _ = score_in_batches(index, train_features, config.batch_size)
        rows = _conformal_rates(
            calibration_scores, normal, anomalous, train_scores, config.alphas, config.seed
        )
        conformal_rows.extend({"category": category, **row} for row in rows)
        manifest["categories"][category] = {
            "calibration_scores": [float(s) for s in calibration_scores],
            "thresholds": {
                str(alpha): conformal_threshold(calibration_scores, alpha)
                for alpha in config.alphas
            },
        }

        # ONNX against PyTorch: first raw feature agreement, then the effect on the end metrics.
        probe = data.test[:N_PARITY_IMAGES]
        reference = extract_features(extractor, probe, config.batch_size, device)
        onnx_float = _onnx_features(float_session, probe, 8)
        quant_test = _onnx_features(quant_session, data.test, 8)
        quant_scores, quant_maps = score_in_batches(index, quant_test, config.batch_size)
        quant_calibration = _onnx_features(quant_session, data.calibration, 8)
        quant_cal_scores, _ = score_in_batches(index, quant_calibration, config.batch_size)
        manifest["categories"][category]["quantized_calibration_scores"] = [
            float(s) for s in quant_cal_scores
        ]
        manifest["categories"][category]["quantized_thresholds"] = {
            str(alpha): conformal_threshold(quant_cal_scores, alpha) for alpha in config.alphas
        }
        float_metrics = evaluate(
            data.test_labels,
            data.test_masks,
            test_scores,
            upsample_maps(test_maps, config.image_size, config.gaussian_sigma),
        )
        quant_metrics = evaluate(
            data.test_labels,
            data.test_masks,
            quant_scores,
            upsample_maps(quant_maps, config.image_size, config.gaussian_sigma),
        )
        parity_rows.append(
            {
                "category": category,
                "float_max_abs_feature_diff": float(np.abs(reference - onnx_float).max()),
                "float_image_auroc": float_metrics["image_auroc"],
                "float_aupro": float_metrics["aupro"],
                "quant_image_auroc": quant_metrics["image_auroc"],
                "quant_aupro": quant_metrics["aupro"],
            }
        )

        # Monitoring: normal images should look uniform; degraded normal images should not.
        p_clean = conformal_p_values(normal, calibration_scores)
        statistic, p_value = uniformity_ks(p_clean)
        drift_row: dict[str, Any] = {
            "category": category,
            "n_images": int(normal.size),
            "clean_ks": statistic,
            "clean_ks_p": p_value,
        }
        normal_images = data.test[~data.test_labels]
        for kind in ("darker_30pct", "defocus_blur"):
            degraded = extract_features(
                extractor, _degrade(normal_images, kind), config.batch_size, device
            )
            scores, _ = score_in_batches(index, degraded, config.batch_size)
            statistic, p_value = uniformity_ks(conformal_p_values(scores, calibration_scores))
            drift_row[f"{kind}_ks"] = statistic
            drift_row[f"{kind}_ks_p"] = p_value
        drift_rows.append(drift_row)

        logger.info(
            "%s: bank=%d auroc float=%.4f quant=%.4f onnx_diff=%.2e",
            category,
            index.bank.shape[0],
            metrics.auroc(data.test_labels, test_scores),
            parity_rows[-1]["quant_image_auroc"],
            parity_rows[-1]["float_max_abs_feature_diff"],
        )
        (output / MANIFEST_NAME).write_text(json.dumps(manifest))
        _write_reports(config, conformal_rows, parity_rows, drift_rows)


def _write_reports(
    config: Config,
    conformal_rows: list[dict[str, Any]],
    parity_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
) -> None:
    summary = []
    for alpha in config.alphas:
        selected = [row for row in conformal_rows if row["alpha"] == alpha]
        summary.append(
            {
                "alpha": alpha,
                "mean_test_false_alarm_rate": float(
                    np.mean([r["test_false_alarm_rate"] for r in selected])
                ),
                "mean_resampled_false_alarm_rate": float(
                    np.mean([r["resampled_mean_false_alarm_rate"] for r in selected])
                ),
                "mean_test_recall": float(np.mean([r["test_recall"] for r in selected])),
                "mean_naive_test_false_alarm_rate": float(
                    np.mean([r["naive_test_false_alarm_rate"] for r in selected])
                ),
            }
        )
    write_report(
        "conformal.json",
        {
            "dev_categories": list(DEV_CATEGORIES),
            "n_resamples": N_RESAMPLES,
            "summary": summary,
            "rows": conformal_rows,
        },
    )
    write_report("onnx_parity.json", {"rows": parity_rows})
    write_report("drift.json", {"rows": drift_rows})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bundle"))
    args = parser.parse_args()
    setup_logging()
    run(Config.load(args.config), tuple(args.categories), args.output)


if __name__ == "__main__":
    main()
