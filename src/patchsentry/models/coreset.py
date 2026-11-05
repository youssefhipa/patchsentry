"""Greedy k-center coreset selection for the PatchCore memory bank."""

from __future__ import annotations

import numpy as np
import torch


def greedy_k_center(
    features: np.ndarray,
    n_select: int,
    projection_dim: int | None = 128,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    """Select ``n_select`` indices so that every point lies close to some selected point.

    Greedy k-center repeatedly adds the point furthest from the current selection, which is a
    2-approximation of the optimal covering radius (Gonzalez, 1985). Unlike random sampling, it
    keeps rare-but-normal appearance, and those patches are the ones whose absence causes false
    alarms.

    Each step is O(N * d), so the features are first projected with a random Gaussian matrix. By the
    Johnson-Lindenstrauss lemma, this approximately preserves pairwise distances.
    """
    if not 0 < n_select <= features.shape[0]:
        raise ValueError(f"n_select must be in [1, {features.shape[0]}], got {n_select}")

    generator = np.random.default_rng(seed)
    points = np.asarray(features, dtype=np.float32)
    if projection_dim is not None and projection_dim < points.shape[1]:
        projection = generator.normal(size=(points.shape[1], projection_dim)).astype(np.float32)
        points = (points @ projection) / np.sqrt(projection_dim, dtype=np.float32)

    data = torch.from_numpy(points).to(device)
    selected = np.empty(n_select, dtype=np.int64)
    current = int(generator.integers(points.shape[0]))
    selected[0] = current
    min_squared = torch.sum((data - data[current]) ** 2, dim=1)
    for position in range(1, n_select):
        current = int(torch.argmax(min_squared).item())
        selected[position] = current
        min_squared = torch.minimum(min_squared, torch.sum((data - data[current]) ** 2, dim=1))
    return selected
