"""One-at-a-time hyperparameter sweep, on the development categories only.

Most anomaly detection papers tune on the test set of the category they report, which inflates the
result. Here, settings are chosen on four categories and the headline number comes from the other
eight, which the sweep never sees.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import mlflow

from patchsentry.data.visa import load_category
from patchsentry.pipeline.common import (
    DEV_CATEGORIES,
    METRIC_KEYS,
    Config,
    build_extractor,
    compute_features,
    evaluate_patchcore,
    fit_patchcore,
    mean_metrics,
    pick_device,
    setup_logging,
    setup_mlflow,
    write_report,
)

logger = logging.getLogger(__name__)

VARIATIONS: list[dict[str, Any]] = [
    {},
    {"backbone": "resnet18"},
    {"backbone": "resnet50"},
    {"image_size": 320},
    {"target_dim": 512},
    {"neighbourhood": 5},
    {"coreset_fraction": 0.001},
    {"coreset_fraction": 0.05},
    {"reweight_neighbours": 1},
]

# Second stage: combinations of the best single changes, still on the development categories.
COMBINATIONS: list[dict[str, Any]] = [
    {"backbone": "resnet50", "image_size": 320},
    {"image_size": 320, "target_dim": 512},
]

# Rerun after a bug fix: with one neighbour the weight was always 0, so every score was 0.
RERUNS: list[dict[str, Any]] = [{"reweight_neighbours": 1}]

STAGES: dict[str, tuple[list[dict[str, Any]], str]] = {
    "single": (VARIATIONS, "sweep.json"),
    "combinations": (COMBINATIONS, "sweep_combinations.json"),
    "rerun": (RERUNS, "sweep_rerun.json"),
}


def _objective(means: dict[str, float | None]) -> float:
    """Equal weight on detection and localisation."""
    return ((means["image_auroc"] or 0.0) + (means["aupro"] or 0.0)) / 2.0


def run(
    base: Config,
    categories: tuple[str, ...],
    variations: list[dict[str, Any]],
    report_name: str = "sweep.json",
) -> None:
    device = pick_device()
    setup_mlflow("patchcore-sweep")
    rows: list[dict[str, Any]] = []
    for variation in variations:
        config = base.with_overrides(**variation)
        name = ",".join(f"{k}={v}" for k, v in variation.items()) or "baseline"
        extractor = build_extractor(config, device)
        per_category: list[dict[str, Any]] = []
        for category in categories:
            started = time.perf_counter()
            data = load_category(
                config.data_root,
                category,
                config.image_size,
                config.calibration_fraction,
                config.seed,
            )
            features = compute_features(extractor, data, config, device)
            index, fit_seconds = fit_patchcore(features, config, device)
            result, scoring_ms = evaluate_patchcore(index, features, data, config)
            row = {
                "category": category,
                **result,
                "fit_seconds": fit_seconds,
                "scoring_ms_per_image": scoring_ms,
                "bank_size": index.bank.shape[0],
            }
            per_category.append(row)
            with mlflow.start_run(run_name=f"{name}/{category}"):
                mlflow.log_params({**config.to_params(), "category": category, "variation": name})
                mlflow.log_metrics(
                    {k: float(v) for k, v in row.items() if k != "category" and v is not None}
                )
            logger.info(
                "%s %s: image_auroc=%.4f aupro=%.4f bank=%d fit=%.0fs score=%.0fms total=%.0fs",
                name,
                category,
                row["image_auroc"],
                row["aupro"],
                row["bank_size"],
                fit_seconds,
                scoring_ms,
                time.perf_counter() - started,
            )
        means = mean_metrics(per_category, (*METRIC_KEYS, "scoring_ms_per_image", "fit_seconds"))
        rows.append(
            {
                "variation": name,
                "overrides": variation,
                "objective": _objective(means),
                "means": means,
                "per_category": per_category,
            }
        )
        write_report(
            report_name,
            {"dev_categories": list(categories), "base": base.to_params(), "results": rows},
        )
    best = max(rows, key=lambda row: row["objective"])
    logger.info("Best variation: %s (objective %.4f)", best["variation"], best["objective"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--categories", nargs="*", default=list(DEV_CATEGORIES))
    parser.add_argument(
        "--stage", choices=list(STAGES), default="single", help="Which grid to run."
    )
    args = parser.parse_args()
    setup_logging()
    variations, report_name = STAGES[args.stage]
    run(Config.load(args.config), tuple(args.categories), variations, report_name)


if __name__ == "__main__":
    main()
