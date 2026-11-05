"""Compare PatchCore with the baselines on every VisA category, with the selected configuration."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import mlflow
import numpy as np

from patchsentry.data.visa import CATEGORIES, CategoryData, load_category
from patchsentry.models.baselines import (
    ImageKnn,
    Padim,
    autoencoder_error_maps,
    train_autoencoder,
)
from patchsentry.models.features import PatchFeatureExtractor, iter_features
from patchsentry.models.patchcore_index import upsample_maps
from patchsentry.pipeline.common import (
    DEV_CATEGORIES,
    METRIC_KEYS,
    Config,
    build_extractor,
    compute_features,
    evaluate,
    evaluate_patchcore,
    fit_patchcore,
    mean_metrics,
    pick_device,
    setup_logging,
    setup_mlflow,
    write_report,
)

logger = logging.getLogger(__name__)

PATCHCORE_SEEDS = (0, 1, 2)


def _run_image_knn(
    extractor: PatchFeatureExtractor,
    data: CategoryData,
    test: np.ndarray,
    config: Config,
    device: str,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    model = ImageKnn().fit(iter_features(extractor, data.train, config.batch_size, device))
    fit_seconds = time.perf_counter() - started
    return evaluate(data.test_labels, data.test_masks, model.score(test), None), fit_seconds


def _run_padim(
    extractor: PatchFeatureExtractor,
    data: CategoryData,
    test: np.ndarray,
    config: Config,
    device: str,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    model = Padim(seed=config.seed).fit(
        iter_features(extractor, data.train, config.batch_size, device)
    )
    fit_seconds = time.perf_counter() - started
    scores, maps = [], []
    for start in range(0, len(test), config.batch_size):
        batch_scores, batch_maps = model.score(test[start : start + config.batch_size])
        scores.append(batch_scores)
        maps.append(batch_maps)
    pixel_maps = upsample_maps(np.concatenate(maps), config.image_size, config.gaussian_sigma)
    result = evaluate(data.test_labels, data.test_masks, np.concatenate(scores), pixel_maps)
    return result, fit_seconds


def _run_autoencoder(data: CategoryData, config: Config, device: str) -> tuple[dict, float]:
    started = time.perf_counter()
    model, _ = train_autoencoder(
        data.train, data.calibration, batch_size=config.batch_size, device=device, seed=config.seed
    )
    fit_seconds = time.perf_counter() - started
    errors = autoencoder_error_maps(model, data.test, config.batch_size, device)
    sigma = config.gaussian_sigma
    maps = np.stack([cv2.GaussianBlur(e, (0, 0), sigmaX=sigma, sigmaY=sigma) for e in errors])
    scores = maps.reshape(len(maps), -1).max(axis=1)
    return evaluate(data.test_labels, data.test_masks, scores, maps), fit_seconds


def run(config: Config, categories: tuple[str, ...]) -> None:
    device = pick_device()
    setup_mlflow("model-comparison")
    extractor = build_extractor(config, device)
    rows: list[dict[str, Any]] = []

    def record(model: str, category: str, result: dict, fit_seconds: float, **extra: Any) -> None:
        row = {
            "model": model,
            "category": category,
            "held_out": category not in DEV_CATEGORIES,
            **result,
            "fit_seconds": fit_seconds,
            **extra,
        }
        rows.append(row)
        with mlflow.start_run(run_name=f"{model}/{category}"):
            mlflow.log_params({**config.to_params(), "model": model, "category": category})
            mlflow.log_metrics(
                {
                    k: float(v)
                    for k, v in row.items()
                    if isinstance(v, int | float) and not isinstance(v, bool)
                }
            )
        logger.info(
            "%-12s %-11s image_auroc=%.4f aupro=%s fit=%.0fs",
            model,
            category,
            result["image_auroc"],
            "n/a" if result["aupro"] is None else f"{result['aupro']:.4f}",
            fit_seconds,
        )

    for category in categories:
        data = load_category(
            config.data_root, category, config.image_size, config.calibration_fraction, config.seed
        )
        features = compute_features(extractor, data, config, device)
        for seed in PATCHCORE_SEEDS:
            index, fit_seconds = fit_patchcore(features, config, device, seed=seed)
            result, scoring_ms = evaluate_patchcore(index, features, data, config)
            record(
                f"patchcore_seed{seed}",
                category,
                result,
                fit_seconds,
                scoring_ms_per_image=scoring_ms,
                bank_size=index.bank.shape[0],
            )

        result, fit_seconds = _run_image_knn(extractor, data, features.test, config, device)
        record("image_knn", category, result, fit_seconds)
        result, fit_seconds = _run_padim(extractor, data, features.test, config, device)
        record("padim", category, result, fit_seconds)
        result, fit_seconds = _run_autoencoder(data, config, device)
        record("autoencoder", category, result, fit_seconds)

        write_report("benchmark.json", _summarise(rows, config))


def _summarise(rows: list[dict[str, Any]], config: Config) -> dict[str, Any]:
    """Aggregate per model over all categories and over the held-out categories alone."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family = "patchcore" if row["model"].startswith("patchcore") else row["model"]
        groups.setdefault(family, []).append(row)
    summary: dict[str, Any] = {}
    for family, members in groups.items():
        entry: dict[str, Any] = {
            "all": mean_metrics(members, METRIC_KEYS),
            "held_out": mean_metrics([m for m in members if m["held_out"]], METRIC_KEYS),
            "mean_fit_seconds": float(np.mean([m["fit_seconds"] for m in members])),
        }
        if family == "patchcore":
            # Spread across coreset seeds of the mean over categories.
            by_seed = [
                [m for m in members if m["model"] == f"patchcore_seed{seed}"]
                for seed in PATCHCORE_SEEDS
            ]
            entry["seed_std_image_auroc"] = float(
                np.std([np.mean([m["image_auroc"] for m in seed_rows]) for seed_rows in by_seed])
            )
        summary[family] = entry
    return {
        "config": config.to_params(),
        "dev_categories": list(DEV_CATEGORIES),
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    args = parser.parse_args()
    setup_logging()
    run(Config.load(args.config), tuple(args.categories))


if __name__ == "__main__":
    main()
