"""Draw the README figures, using only the JSON reports and the model bundle. No PyTorch."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from patchsentry.data.visa import load_category
from patchsentry.models.patchcore_index import upsample_maps
from patchsentry.serving.engine import InferenceEngine

logger = logging.getLogger(__name__)
REPORTS = Path("reports")
FIGURES = REPORTS / "figures"

# First three slots of a palette validated for colour-vision deficiency across all pairs.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "xtick.bottom": False,
            "ytick.left": False,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 8,
        }
    )


def _load(name: str) -> dict[str, Any] | None:
    path = REPORTS / name
    if not path.exists():
        logger.warning("Skipping a figure: %s does not exist", path)
        return None
    report: dict[str, Any] = json.loads(path.read_text())
    return report


def _save(figure: Any, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(figure)
    logger.info("Wrote %s", FIGURES / name)


def model_comparison() -> None:
    report = _load("benchmark.json")
    if report is None:
        return
    rows = report["rows"]
    categories = sorted({row["category"] for row in rows})

    def mean_auroc(family: str, category: str) -> float:
        values = [
            row["image_auroc"]
            for row in rows
            if row["category"] == category and row["model"].startswith(family)
        ]
        return float(np.mean(values))

    order = sorted(categories, key=lambda category: mean_auroc("patchcore", category))
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(order))
    for colour, (family, label) in zip(
        SERIES,
        (("patchcore", "PatchCore"), ("padim", "PaDiM"), ("autoencoder", "Autoencoder")),
        strict=True,
    ):
        values = [mean_auroc(family, category) for category in order]
        axis.scatter(
            values, positions, s=64, color=colour, label=label, edgecolor=SURFACE, linewidth=1.5
        )
    axis.set_yticks(positions, order)
    axis.set_xlabel("Image-level AUROC (higher is better)")
    axis.set_title("Detection quality per VisA category", pad=26)
    axis.grid(axis="y", visible=False)
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    _save(figure, "model_comparison.png")


def conformal() -> None:
    report = _load("conformal.json")
    if report is None:
        return
    summary = report["summary"]
    alphas = [row["alpha"] for row in summary]
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    limit = max(max(row["mean_naive_test_false_alarm_rate"] for row in summary), max(alphas)) * 1.1
    axis.plot([0, limit], [0, limit], color=AXIS, linewidth=1.5, zorder=1)
    axis.annotate(
        "observed = target",
        (limit * 0.8, limit * 0.8),
        xytext=(-8, 8),
        textcoords="offset points",
        color=INK_SECONDARY,
        ha="right",
        rotation=0,
    )
    series = (
        ("mean_resampled_false_alarm_rate", "Conformal, resampled splits", "o"),
        ("mean_test_false_alarm_rate", "Conformal, official test split", "s"),
        ("mean_naive_test_false_alarm_rate", "Naive: quantile of training scores", "^"),
    )
    for colour, (key, label, marker) in zip(SERIES, series, strict=True):
        axis.plot(alphas, [row[key] for row in summary], marker=marker, color=colour, label=label)
    axis.set_xlabel("Target false alarm rate")
    axis.set_ylabel("Observed false alarm rate")
    axis.set_title("Calibrated thresholds keep their promise", pad=62)
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    _save(figure, "conformal_calibration.png")


def backbone_throughput() -> None:
    report = _load("bench_backbone.json")
    if report is None:
        return
    rows = sorted(
        (row for row in report["rows"] if row["batch_size"] == 16),
        key=lambda row: row["images_per_second"],
    )
    figure, axis = plt.subplots(figsize=(7, 3.2))
    names = [row["runtime"] for row in rows]
    values = [row["images_per_second"] for row in rows]
    axis.barh(names, values, color=SERIES[0], height=0.55)
    for position, value in enumerate(values):
        axis.annotate(
            f"{value:.0f}",
            (value, position),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            color=INK,
        )
    axis.set_xlabel("Images per second, feature extraction only, batch size 16")
    axis.set_title("Backbone throughput by runtime")
    axis.grid(axis="y", visible=False)
    _save(figure, "backbone_throughput.png")


def loadtest() -> None:
    report = _load("loadtest.json")
    if report is None:
        return
    figure, (left, right) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    names = (("micro-batching", "Micro-batching"), ("no-batching", "No batching"))
    for colour, (name, label) in zip(SERIES, names, strict=False):
        rows = [row for row in report["rows"] if row["configuration"] == name]
        clients = [row["concurrent_clients"] for row in rows]
        left.plot(clients, [row["requests_per_second"] for row in rows], "o-", color=colour)
        right.plot(
            clients, [row["latency_p95_ms"] for row in rows], "o-", color=colour, label=label
        )
    left.set_title("Throughput (requests per second)")
    right.set_title("p95 latency (ms)")
    for axis in (left, right):
        axis.set_xlabel("Concurrent clients")
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 4, 16, 32], ["1", "4", "16", "32"])
        axis.set_ylim(bottom=0)
    right.legend(loc="upper left")
    figure.tight_layout()
    _save(figure, "loadtest.png")


def _overlay(image: np.ndarray, pixel_map: np.ndarray, threshold: float) -> np.ndarray:
    level = np.clip(pixel_map / max(2.0 * threshold, 1e-6), 0.0, 1.0)
    coloured = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.addWeighted(image, 0.5, cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB), 0.5, 0)


def gallery(bundle: Path, data_root: Path, categories: list[str]) -> None:
    """Show real detections, plus each category's lowest-scoring defect, so failures are visible."""
    manifest_path = bundle / "manifest.json"
    if not manifest_path.exists():
        logger.warning("Skipping the gallery: no bundle at %s", bundle)
        return
    config = json.loads(manifest_path.read_text())["config"]
    engine = InferenceEngine(bundle, categories=categories)
    panels = []
    for category in categories:
        data = load_category(
            data_root,
            category,
            engine.image_size,
            config["calibration_fraction"],
            config["seed"],
        )
        images = data.test[data.test_labels]
        masks = data.test_masks[data.test_labels]
        results = []
        for start in range(0, len(images), 16):
            batch = images[start : start + 16]
            results.extend(engine.infer_batch(batch, [category] * len(batch)))
        scores = np.array([result.score for result in results])
        # Median-scoring detection (typical, not cherry-picked) and the hardest defect (lowest).
        detected = np.flatnonzero([result.is_anomalous for result in results])
        typical = detected[np.argsort(scores[detected])[len(detected) // 2]] if detected.size else 0
        for kind, position in (("typical", int(typical)), ("hardest", int(np.argmin(scores)))):
            result = results[position]
            pixel_map = upsample_maps(
                result.patch_map[None], engine.image_size, engine.gaussian_sigma
            )[0]
            panels.append((category, kind, images[position], masks[position], pixel_map, result))

    figure, axes = plt.subplots(len(panels), 3, figsize=(7.5, 2.5 * len(panels)))
    axes = np.atleast_2d(axes)
    for row, (category, kind, image, mask, pixel_map, result) in zip(axes, panels, strict=True):
        decision = "ALARM" if result.is_anomalous else "missed"
        row[0].imshow(image)
        row[0].set_title(f"{category}, {kind} defect", fontsize=10)
        row[1].imshow(mask, cmap="gray")
        row[1].set_title("Ground truth", fontsize=10)
        row[2].imshow(_overlay(image, pixel_map, result.threshold))
        row[2].set_title(
            f"Score {result.score:.1f} vs {result.threshold:.1f}: {decision}", fontsize=10
        )
        for axis in row:
            axis.axis("off")
    figure.tight_layout()
    _save(figure, "gallery.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=Path("artifacts/bundle"))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/visa"))
    parser.add_argument("--categories", nargs="*", default=["pcb1", "candle", "cashew"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _style()
    model_comparison()
    conformal()
    backbone_throughput()
    loadtest()
    gallery(args.bundle, args.data_root, args.categories)


if __name__ == "__main__":
    main()
