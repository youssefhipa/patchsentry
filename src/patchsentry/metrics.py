"""Evaluation metrics, implemented in NumPy and cross-checked against scikit-learn in the tests."""

from __future__ import annotations

import cv2
import numpy as np


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic, with average ranks for ties.

    AUROC equals the probability that a random positive outranks a random negative, so it needs
    only one sort and no threshold.
    """
    labels = np.asarray(labels, dtype=bool).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs at least one positive and one negative sample.")

    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average_ranks = ends - (counts - 1) / 2.0  # 1-based average rank of each tied group
    rank_sum = average_ranks[inverse][labels].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    """Best F1 score over all thresholds (an oracle value: it uses test labels to choose)."""
    labels = np.asarray(labels, dtype=bool).ravel()
    order = np.argsort(-np.asarray(scores, dtype=np.float64).ravel(), kind="stable")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    predicted_positives = np.arange(1, labels.size + 1)
    denominator = predicted_positives + labels.sum()
    return float((2.0 * true_positives / denominator).max())


def aupro(
    masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3, n_thresholds: int = 200
) -> float:
    """Normalised area under the per-region-overlap curve, up to ``fpr_limit``.

    Pixel AUROC is dominated by large defects because every pixel counts equally. PRO averages over
    connected ground-truth regions instead, so a tiny scratch weighs as much as a large stain.
    Thresholds are quantiles of the normal-pixel scores, which spaces them evenly in FPR.

    Args:
        masks: boolean array (N, H, W).
        maps: float array (N, H, W) of anomaly scores.
        fpr_limit: integrate over FPR in [0, fpr_limit], then divide by fpr_limit.
        n_thresholds: number of operating points on the curve.
    """
    masks = np.asarray(masks, dtype=bool)
    if masks.shape != maps.shape:
        raise ValueError(f"Shape mismatch: masks {masks.shape}, maps {maps.shape}")
    if not 0.0 < fpr_limit <= 1.0:
        raise ValueError(f"fpr_limit must be in (0, 1], got {fpr_limit}")

    # Give each connected region a globally unique id, with 0 meaning background.
    region_ids = np.zeros(masks.shape, dtype=np.int32)
    next_id = 1
    for index, mask in enumerate(masks):
        if mask.any():
            count, components = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
            region_ids[index] = np.where(components > 0, components + next_id - 1, 0)
            next_id += count - 1
    if next_id == 1:
        raise ValueError("AUPRO needs at least one anomalous region.")

    flat_ids = region_ids.ravel()
    flat_scores = np.asarray(maps, dtype=np.float32).ravel()
    normal_scores = flat_scores[flat_ids == 0]
    region_sizes = np.bincount(flat_ids, minlength=next_id)[1:]
    anomalous = flat_ids > 0
    anomalous_ids = flat_ids[anomalous]
    anomalous_scores = flat_scores[anomalous]

    thresholds = np.quantile(normal_scores, np.linspace(1.0 - fpr_limit, 1.0, n_thresholds))
    fprs = np.empty(n_thresholds + 1)
    pros = np.empty(n_thresholds + 1)
    for position, threshold in enumerate(thresholds):
        fprs[position] = (normal_scores >= threshold).mean()
        hits = anomalous_scores >= threshold
        overlap = np.bincount(anomalous_ids[hits], minlength=next_id)[1:]
        pros[position] = (overlap / region_sizes).mean()
    # Anchor the curve at the origin so that the integral covers the full range.
    fprs[-1] = 0.0
    pros[-1] = 0.0

    order = np.argsort(fprs, kind="stable")
    fprs, pros = fprs[order], pros[order]
    keep = fprs <= fpr_limit
    # Interpolate an exact end point at fpr_limit if the curve stops short of it.
    x = np.append(fprs[keep], fpr_limit)
    y = np.append(pros[keep], np.interp(fpr_limit, fprs, pros))
    return float(np.trapezoid(y, x) / fpr_limit)
