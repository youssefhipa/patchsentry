"""Parse the VisA dataset using its official one-class split (``split_csv/1cls.csv``)."""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np

from patchsentry.data.download import SPLIT_CSV
from patchsentry.data.preprocessing import load_mask, load_rgb

CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
)
_REQUIRED_COLUMNS = {"object", "split", "label", "image", "mask"}
_T = TypeVar("_T")


@dataclass(frozen=True)
class VisaSample:
    category: str
    split: str
    is_anomaly: bool
    image_path: Path
    mask_path: Path | None


@dataclass
class CategoryData:
    """All images of one category, resized and held in memory as uint8 NHWC arrays."""

    category: str
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    test_masks: np.ndarray


def read_split(root: Path) -> list[VisaSample]:
    """Read and validate the split file. Raises instead of silently dropping malformed rows."""
    csv_path = root / SPLIT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `python -m patchsentry.data.download`.")

    samples: list[VisaSample] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = _REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} lacks columns {sorted(missing)}")
        for row in reader:
            if row["label"] not in {"normal", "anomaly"}:
                raise ValueError(f"Unknown label {row['label']!r} in {csv_path}")
            if row["split"] not in {"train", "test"}:
                raise ValueError(f"Unknown split {row['split']!r} in {csv_path}")
            is_anomaly = row["label"] == "anomaly"
            if is_anomaly and not row["mask"]:
                raise ValueError(f"Anomalous sample without a mask: {row['image']}")
            samples.append(
                VisaSample(
                    category=row["object"],
                    split=row["split"],
                    is_anomaly=is_anomaly,
                    image_path=root / row["image"],
                    mask_path=root / row["mask"] if row["mask"] else None,
                )
            )
    return samples


def split_calibration(n: int, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split ``n`` indices into (train, calibration).

    The calibration images must not enter the memory bank. An image in the bank has a distance of
    near zero to itself, so its score would be too small to represent unseen data and the conformal
    guarantee would not hold.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    permutation = np.random.default_rng(seed).permutation(n)
    n_calibration = round(n * fraction)
    return np.sort(permutation[n_calibration:]), np.sort(permutation[:n_calibration])


def _parallel_map(function: Callable[[Path], _T], paths: Sequence[Path]) -> list[_T]:
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(function, paths))


def load_category(
    root: Path,
    category: str,
    image_size: int,
    calibration_fraction: float = 0.25,
    seed: int = 0,
) -> CategoryData:
    samples = [sample for sample in read_split(root) if sample.category == category]
    if not samples:
        raise ValueError(f"Category {category!r} not found. Known categories: {CATEGORIES}")

    normal_train = [s for s in samples if s.split == "train" and not s.is_anomaly]
    test = [s for s in samples if s.split == "test"]
    if not normal_train or not test:
        raise ValueError(f"Category {category!r} needs both train and test samples.")

    train_images = np.stack(
        _parallel_map(lambda p: load_rgb(p, image_size), [s.image_path for s in normal_train])
    )
    test_images = np.stack(
        _parallel_map(lambda p: load_rgb(p, image_size), [s.image_path for s in test])
    )
    empty_mask = np.zeros((image_size, image_size), dtype=bool)
    masks = [
        load_mask(s.mask_path, image_size) if s.mask_path is not None else empty_mask for s in test
    ]

    train_idx, calibration_idx = split_calibration(len(train_images), calibration_fraction, seed)
    return CategoryData(
        category=category,
        train=train_images[train_idx],
        calibration=train_images[calibration_idx],
        test=test_images,
        test_labels=np.array([s.is_anomaly for s in test], dtype=bool),
        test_masks=np.stack(masks),
    )
