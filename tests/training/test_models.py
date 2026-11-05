from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from patchsentry.models.baselines import (
    ImageKnn,
    Padim,
    autoencoder_error_maps,
    train_autoencoder,
)
from patchsentry.models.coreset import greedy_k_center
from patchsentry.models.features import (
    PatchFeatureExtractor,
    channel_bins,
    extract_features,
    sample_patches,
)
from patchsentry.models.patchcore_index import PatchCoreIndex, upsample_maps
from patchsentry.models.torch_index import TorchExactIndex
from tests.synthetic import IMAGE_SIZE, textured_images, with_defect


def _covering_radius(points: np.ndarray, selected: np.ndarray) -> float:
    distances = np.linalg.norm(points[:, None] - points[selected][None], axis=2)
    return float(distances.min(axis=1).max())


def test_coreset_covers_rare_points_that_random_sampling_misses() -> None:
    generator = np.random.default_rng(0)
    dense = generator.normal(size=(2000, 8))
    rare = generator.normal(size=(5, 8)) + 25.0  # rare but legitimate appearance
    points = np.vstack([dense, rare]).astype(np.float32)
    selected = greedy_k_center(points, 50, projection_dim=None)
    random_selected = generator.choice(len(points), 50, replace=False)
    assert (selected >= 2000).any(), "greedy k-center must pick from the rare cluster"
    assert _covering_radius(points, selected) < 0.5 * _covering_radius(points, random_selected)


def test_coreset_returns_unique_indices_and_is_deterministic() -> None:
    points = np.random.default_rng(1).normal(size=(500, 16)).astype(np.float32)
    first = greedy_k_center(points, 40, projection_dim=8, seed=5)
    second = greedy_k_center(points, 40, projection_dim=8, seed=5)
    assert len(set(first.tolist())) == 40
    assert np.array_equal(first, second)


def test_projection_approximately_preserves_the_coreset_quality() -> None:
    points = np.random.default_rng(2).normal(size=(1000, 256)).astype(np.float32)
    exact = _covering_radius(points, greedy_k_center(points, 50, projection_dim=None))
    projected = _covering_radius(points, greedy_k_center(points, 50, projection_dim=64))
    assert projected < 1.15 * exact


@pytest.mark.parametrize("n", [0, 11])
def test_coreset_rejects_an_invalid_size(n: int) -> None:
    with pytest.raises(ValueError, match="n_select"):
        greedy_k_center(np.zeros((10, 4), dtype=np.float32), n)


def test_channel_bins_are_row_stochastic_and_cover_all_channels() -> None:
    weights = channel_bins(1536, 1024).squeeze()
    assert torch.allclose(weights.sum(dim=1), torch.ones(1024))
    assert (weights.sum(dim=0) > 0).all()


def test_extractor_output_grid_and_dimension() -> None:
    extractor = PatchFeatureExtractor("resnet18", target_dim=128, pretrained=False)
    output = extractor(torch.zeros(2, 3, 64, 64))
    assert output.shape == (2, 128, 8, 8)
    with pytest.raises(ValueError, match="odd"):
        PatchFeatureExtractor("resnet18", neighbourhood=4, pretrained=False)
    with pytest.raises(ValueError, match="Unknown backbone"):
        PatchFeatureExtractor("vgg", pretrained=False)


def test_extractor_does_not_project_when_already_small() -> None:
    extractor = PatchFeatureExtractor("resnet18", target_dim=1024, pretrained=False)
    assert extractor.output_dim == 384  # layer2 (128) + layer3 (256)


def test_sample_patches_respects_budget() -> None:
    extractor = PatchFeatureExtractor("resnet18", target_dim=64, pretrained=False)
    sampled = sample_patches(extractor, textured_images(4), max_patches=40, batch_size=2)
    assert sampled.shape == (40, 64)


def test_patchcore_reweighting_formula() -> None:
    """Checks the score by hand on a bank small enough to compute directly."""
    bank = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    index = PatchCoreIndex(bank, "torch", reweight_neighbours=2)
    features = np.array([[[[0.0, 2.0]]]], dtype=np.float32)  # one patch; m* = bank[0]
    scores, maps = index.score(features)
    d0, d1 = 2.0, np.sqrt(5.0)  # distances to m* and to its other neighbour, bank[1]
    weight = 1.0 - np.exp(d0) / (np.exp(d0) + np.exp(d1))
    assert maps[0, 0, 0] == pytest.approx(d0)
    assert scores[0] == pytest.approx(weight * d0, rel=1e-5)


def test_patchcore_scoring_survives_distances_that_would_overflow_exp() -> None:
    bank = np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32)
    index = PatchCoreIndex(bank, "torch")
    features = np.full((1, 2, 2, 8), 500.0, dtype=np.float32)
    scores, _ = index.score(features)
    assert np.isfinite(scores).all()


def test_patchcore_shape_validation() -> None:
    index = PatchCoreIndex(np.zeros((5, 4), dtype=np.float32), "torch", reweight_neighbours=2)
    with pytest.raises(ValueError, match="features"):
        index.score(np.zeros((1, 2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="non-empty"):
        PatchCoreIndex(np.zeros((0, 4), dtype=np.float32), "torch")


def test_patchcore_save_and_load_round_trip(tmp_path: Path) -> None:
    bank = np.random.default_rng(0).normal(size=(100, 8)).astype(np.float32)
    features = np.random.default_rng(1).normal(size=(2, 3, 3, 8)).astype(np.float32)
    original = PatchCoreIndex(bank, "torch", reweight_neighbours=4)
    original.save(tmp_path)
    scores_before, maps_before = original.score(features)
    scores_after, maps_after = PatchCoreIndex.load(tmp_path, "torch").score(features)
    assert np.array_equal(scores_before, scores_after)
    assert np.array_equal(maps_before, maps_after)


def test_upsample_maps_keeps_the_location_of_the_peak() -> None:
    patch_map = np.zeros((1, 8, 8), dtype=np.float32)
    patch_map[0, 2, 6] = 1.0
    pixels = upsample_maps(patch_map, 64, sigma=2.0)
    assert pixels.shape == (1, 64, 64)
    row, column = np.unravel_index(pixels[0].argmax(), (64, 64))
    assert abs(row - 20) <= 6 and abs(column - 52) <= 6


def test_image_knn_ranks_a_shifted_image_higher() -> None:
    generator = np.random.default_rng(0)
    train = generator.normal(size=(50, 4, 4, 8)).astype(np.float32)
    model = ImageKnn(k=3).fit([train])
    normal = generator.normal(size=(5, 4, 4, 8)).astype(np.float32)
    assert model.score(normal + 5.0).min() > model.score(normal).max()


def test_padim_mahalanobis_matches_direct_computation() -> None:
    generator = np.random.default_rng(0)
    train = generator.normal(size=(40, 2, 2, 3)).astype(np.float32)
    model = Padim(n_channels=3, regularisation=0.01).fit([train[:20], train[20:]])

    # Online accumulation over two batches must equal the direct estimate at one position.
    position = train[:, 1, 0, model.channels]
    assert model.mean[1, 0] == pytest.approx(position.mean(axis=0), abs=1e-5)
    covariance = np.cov(position, rowvar=False) + 0.01 * np.eye(3)
    query = generator.normal(size=(1, 2, 2, 3)).astype(np.float32)
    delta = query[0, 1, 0, model.channels] - position.mean(axis=0)
    expected = np.sqrt(delta @ np.linalg.inv(covariance) @ delta)
    assert model.score(query)[1][0, 1, 0] == pytest.approx(expected, rel=1e-3)


def test_autoencoder_trains_and_highlights_defects() -> None:
    train = textured_images(16)
    model, losses = train_autoencoder(train, train[:4], epochs=25, batch_size=8)
    assert losses[-1] < losses[0]
    normal_error = autoencoder_error_maps(model, textured_images(2, seed=9))
    defect_error = autoencoder_error_maps(model, with_defect(textured_images(2, seed=9)))
    assert normal_error.shape == (2, IMAGE_SIZE, IMAGE_SIZE)
    assert defect_error[:, 20:44, 20:44].mean() > 2 * normal_error[:, 20:44, 20:44].mean()


def test_patchcore_end_to_end_separates_a_synthetic_defect_and_localises_it() -> None:
    torch.manual_seed(0)
    extractor = PatchFeatureExtractor("resnet18", target_dim=128, pretrained=False)
    train = extract_features(extractor, textured_images(30), batch_size=8)
    pool = train.reshape(-1, 128)
    index = PatchCoreIndex(pool[greedy_k_center(pool, 200, projection_dim=32)], "torch")

    normal_scores, _ = index.score(extract_features(extractor, textured_images(8, seed=3)))
    defect_scores, defect_maps = index.score(
        extract_features(extractor, with_defect(textured_images(8, seed=3)))
    )
    assert defect_scores.min() > normal_scores.max()
    inside = defect_maps[:, 3:5, 3:5].mean()
    corners = defect_maps[:, 0, 0].mean()
    assert inside > corners


def test_torch_index_matches_brute_force() -> None:
    generator = np.random.default_rng(0)
    bank = generator.normal(size=(300, 16)).astype(np.float32)
    queries = generator.normal(size=(50, 16)).astype(np.float32)
    squared, indices = TorchExactIndex(bank, device="cpu", chunk_size=7).search(queries, 3)
    expected = ((queries[:, None] - bank[None]) ** 2).sum(axis=2)
    assert np.array_equal(indices, np.argsort(expected, axis=1)[:, :3])
    assert squared == pytest.approx(np.sort(expected, axis=1)[:, :3], rel=1e-3)
