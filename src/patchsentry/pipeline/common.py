"""Shared pipeline code: configuration, PatchCore fitting, evaluation and report writing."""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
import yaml

from patchsentry import metrics
from patchsentry.data.visa import CategoryData
from patchsentry.models.coreset import greedy_k_center
from patchsentry.models.features import PatchFeatureExtractor, iter_features, sample_patches
from patchsentry.models.patchcore_index import PatchCoreIndex, upsample_maps

logger = logging.getLogger(__name__)

REPORTS = Path("reports")
MLFLOW_URI = "sqlite:///artifacts/mlflow.db"

# Tuning happens on these only. They span the dataset's structure types (single and multiple
# instances, simple and complex surfaces). The other eight categories are the unbiased estimate.
DEV_CATEGORIES = ("candle", "cashew", "macaroni1", "pcb1")


@dataclass(frozen=True)
class Config:
    data_root: Path = Path("data/raw/visa")
    seed: int = 0
    calibration_fraction: float = 0.25
    image_size: int = 224
    backbone: str = "wide_resnet50_2"
    target_dim: int = 1024
    neighbourhood: int = 3
    coreset_fraction: float = 0.01
    max_pool_patches: int = 200_000
    projection_dim: int = 128
    reweight_neighbours: int = 9
    gaussian_sigma: float = 4.0
    batch_size: int = 32
    alphas: tuple[float, ...] = (0.01, 0.05, 0.1)
    default_alpha: float = 0.05
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = yaml.safe_load(path.read_text())
        raw["data_root"] = Path(raw["data_root"])
        raw["alphas"] = tuple(raw["alphas"])
        return cls(**raw)

    def with_overrides(self, **overrides: Any) -> Config:
        return replace(self, **overrides)

    def to_params(self) -> dict[str, Any]:
        params = asdict(self)
        params.pop("extra")
        params["data_root"] = str(self.data_root)
        return params


@dataclass
class CategoryFeatures:
    """Features of one category, computed once and reused across models and memory-bank sizes."""

    pool: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    n_total_train_patches: int


def config_from_manifest(manifest: dict[str, Any]) -> Config:
    """Rebuild the configuration stored in a model bundle."""
    raw = dict(manifest["config"])
    raw["data_root"] = Path(raw["data_root"])
    raw["alphas"] = tuple(raw["alphas"])
    return Config(**raw)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("alembic").setLevel(logging.WARNING)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def setup_mlflow(experiment: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment)


def environment_info() -> dict[str, str]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        commit = ""
    return {
        "machine": platform.machine(),
        "system": platform.platform(),
        "processor": subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
        or platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "git_commit": commit,
    }


def write_report(name: str, payload: dict[str, Any]) -> Path:
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / name
    payload = {"environment": environment_info(), **payload}
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n")
    logger.info("Wrote %s", path)
    return path


def build_extractor(config: Config, device: str) -> PatchFeatureExtractor:
    extractor = PatchFeatureExtractor(
        backbone=config.backbone,
        target_dim=config.target_dim,
        neighbourhood=config.neighbourhood,
    )
    return extractor.to(device).eval()


def compute_features(
    extractor: PatchFeatureExtractor, data: CategoryData, config: Config, device: str
) -> CategoryFeatures:
    grid = config.image_size // 8
    pool = sample_patches(
        extractor,
        data.train,
        config.max_pool_patches,
        config.batch_size,
        device,
        seed=config.seed,
    )
    calibration = np.concatenate(
        list(iter_features(extractor, data.calibration, config.batch_size, device))
    )
    test = np.concatenate(list(iter_features(extractor, data.test, config.batch_size, device)))
    return CategoryFeatures(pool, calibration, test, len(data.train) * grid * grid)


def fit_patchcore(
    features: CategoryFeatures, config: Config, device: str, seed: int | None = None
) -> tuple[PatchCoreIndex, float]:
    """Select the coreset and build the index. Returns the index and the fit time in seconds."""
    started = time.perf_counter()
    n_select = max(1, int(config.coreset_fraction * features.n_total_train_patches))
    n_select = min(n_select, features.pool.shape[0])
    chosen = greedy_k_center(
        features.pool,
        n_select,
        projection_dim=config.projection_dim,
        seed=config.seed if seed is None else seed,
        device=device,
    )
    index = PatchCoreIndex(
        features.pool[chosen], index_kind="torch", reweight_neighbours=config.reweight_neighbours
    )
    return index, time.perf_counter() - started


def score_in_batches(
    index: PatchCoreIndex, features: np.ndarray, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    scores, maps = [], []
    for start in range(0, len(features), batch_size):
        batch_scores, batch_maps = index.score(features[start : start + batch_size])
        scores.append(batch_scores)
        maps.append(batch_maps)
    return np.concatenate(scores), np.concatenate(maps)


def evaluate(
    labels: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    pixel_maps: np.ndarray | None,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "image_auroc": metrics.auroc(labels, scores),
        "image_f1_max": metrics.f1_max(labels, scores),
        "pixel_auroc": None,
        "aupro": None,
    }
    if pixel_maps is not None:
        result["pixel_auroc"] = metrics.auroc(masks, pixel_maps)
        result["aupro"] = metrics.aupro(masks, pixel_maps)
    return result


def evaluate_patchcore(
    index: PatchCoreIndex, features: CategoryFeatures, data: CategoryData, config: Config
) -> tuple[dict[str, float | None], float]:
    """Return the metrics and the mean scoring time per image, in milliseconds."""
    started = time.perf_counter()
    scores, maps = score_in_batches(index, features.test, config.batch_size)
    scoring_ms = (time.perf_counter() - started) * 1000 / len(features.test)
    pixel_maps = upsample_maps(maps, config.image_size, config.gaussian_sigma)
    return evaluate(data.test_labels, data.test_masks, scores, pixel_maps), scoring_ms


def mean_metrics(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float | None]:
    means: dict[str, float | None] = {}
    for key in keys:
        values = [row[key] for row in rows if row.get(key) is not None]
        means[key] = float(np.mean(values)) if values else None
    return means


METRIC_KEYS = ("image_auroc", "image_f1_max", "pixel_auroc", "aupro")
