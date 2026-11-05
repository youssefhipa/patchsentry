from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from patchsentry.metrics import aupro, auroc, f1_max


@pytest.mark.parametrize("seed", range(5))
def test_auroc_matches_sklearn_with_ties(seed: int) -> None:
    generator = np.random.default_rng(seed)
    labels = generator.random(500) < 0.3
    scores = np.round(generator.normal(size=500) + labels, 1)  # rounding creates ties
    assert auroc(labels, scores) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)


def test_auroc_rejects_a_single_class() -> None:
    with pytest.raises(ValueError, match="positive and one negative"):
        auroc(np.zeros(5), np.arange(5))


def test_f1_max_matches_exhaustive_search() -> None:
    generator = np.random.default_rng(0)
    labels = generator.random(200) < 0.4
    scores = generator.normal(size=200) + labels
    expected = max(f1_score(labels, scores >= threshold) for threshold in scores)
    assert f1_max(labels, scores) == pytest.approx(expected)


def _two_region_masks() -> np.ndarray:
    masks = np.zeros((2, 64, 64), dtype=bool)
    masks[0, 5:35, 5:35] = True  # large region
    masks[1, 50:53, 50:53] = True  # tiny region
    return masks


def test_aupro_is_one_for_a_perfect_map() -> None:
    masks = _two_region_masks()
    noise = np.random.default_rng(0).random(masks.shape) * 0.1
    assert aupro(masks, masks + noise, n_thresholds=50) == pytest.approx(1.0, abs=1e-3)


def test_aupro_weighs_regions_equally_unlike_pixel_auroc() -> None:
    """Finding only the large defect gets high pixel AUROC, but PRO counts the missed small one."""
    masks = _two_region_masks()
    maps = np.random.default_rng(0).random(masks.shape) * 0.1
    maps[0, 5:35, 5:35] += 1.0
    assert auroc(masks, maps) > 0.97
    assert aupro(masks, maps, n_thresholds=50) < 0.7


@pytest.mark.parametrize("seed", range(3))
def test_aupro_of_a_random_map_matches_theory(seed: int) -> None:
    """A random map has PRO == FPR in expectation, so the normalised area up to 0.3 is 0.15.

    Many regions are needed: with only a few small ones, the average PRO is very noisy.
    """
    generator = np.random.default_rng(seed)
    masks = np.zeros((40, 64, 64), dtype=bool)
    for mask in masks:
        row, column = generator.integers(0, 50, size=2)
        mask[row : row + 8, column : column + 8] = True
    maps = generator.random(masks.shape)
    assert aupro(masks, maps, fpr_limit=0.3, n_thresholds=100) == pytest.approx(0.15, abs=0.02)


def test_aupro_validates_input() -> None:
    with pytest.raises(ValueError, match="at least one anomalous region"):
        aupro(np.zeros((1, 8, 8), dtype=bool), np.zeros((1, 8, 8)))
    with pytest.raises(ValueError, match="Shape mismatch"):
        aupro(np.zeros((1, 8, 8), dtype=bool), np.zeros((1, 4, 4)))
