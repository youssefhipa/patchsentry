"""Baselines that PatchCore has to beat, from simplest to most similar.

- ``ImageKnn``: one global feature per image. Shows what the patch-level bank adds.
- ``Padim``: a Gaussian per spatial position. Parametric and compact, but assumes alignment.
- ``ConvAutoencoder``: reconstruction error with no pretrained features. Shows what transfer adds.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


class ImageKnn:
    """Mean distance to the k nearest training images in global-average-pooled feature space."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.vectors: np.ndarray | None = None

    def fit(self, batches: Iterable[np.ndarray]) -> ImageKnn:
        self.vectors = np.concatenate([batch.mean(axis=(1, 2)) for batch in batches], axis=0)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.vectors is None:
            raise RuntimeError("Call fit() first.")
        vectors = features.mean(axis=(1, 2))
        # A few hundred vectors per category, so brute force in NumPy is fast enough.
        distances = np.linalg.norm(vectors[:, None, :] - self.vectors[None, :, :], axis=2)
        k = min(self.k, self.vectors.shape[0])
        return np.asarray(np.partition(distances, k - 1, axis=1)[:, :k].mean(axis=1))


class Padim:
    """Per-position multivariate Gaussian scored by Mahalanobis distance (Defard et al., 2021).

    Mean and covariance are accumulated online, so memory does not grow with the training set. The
    channels are subsampled at random because a full covariance per position costs O(HW * d^2).
    """

    def __init__(self, n_channels: int = 100, regularisation: float = 0.01, seed: int = 0) -> None:
        self.n_channels = n_channels
        self.regularisation = regularisation
        self.seed = seed
        self.channels: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.precision: np.ndarray | None = None

    def fit(self, batches: Iterable[np.ndarray]) -> Padim:
        count = 0
        sum_x: np.ndarray | None = None
        sum_xx: np.ndarray | None = None
        grid = (0, 0)
        for batch in batches:
            n, height, width, dimension = batch.shape
            if self.channels is None:
                generator = np.random.default_rng(self.seed)
                size = min(self.n_channels, dimension)
                self.channels = np.sort(generator.choice(dimension, size, replace=False))
            grid = (height, width)
            selected = batch[..., self.channels].astype(np.float64)
            # (positions, images, channels), so matmul gives one covariance per position.
            by_position = selected.reshape(n, height * width, -1).transpose(1, 0, 2)
            if sum_x is None or sum_xx is None:
                sum_x = np.zeros((height * width, by_position.shape[2]))
                sum_xx = np.zeros((height * width, by_position.shape[2], by_position.shape[2]))
            sum_x += by_position.sum(axis=1)
            sum_xx += by_position.transpose(0, 2, 1) @ by_position
            count += n
        if count < 2 or sum_x is None or sum_xx is None:
            raise ValueError("PaDiM needs at least two training images.")

        mean = sum_x / count
        covariance = (sum_xx - count * mean[:, :, None] * mean[:, None, :]) / (count - 1)
        covariance += self.regularisation * np.eye(mean.shape[1])
        self.mean = mean.astype(np.float32).reshape(*grid, -1)
        self.precision = np.linalg.inv(covariance).astype(np.float32)
        return self

    def score(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.mean is None or self.precision is None or self.channels is None:
            raise RuntimeError("Call fit() first.")
        n, height, width, _ = features.shape
        delta = features[..., self.channels] - self.mean[None]
        delta = delta.reshape(n, height * width, -1)
        squared = np.einsum("npi,pij,npj->np", delta, self.precision, delta, optimize=True)
        maps = np.sqrt(np.maximum(squared, 0.0)).reshape(n, height, width).astype(np.float32)
        return maps.reshape(n, -1).max(axis=1), maps


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with an 8x-downsampled bottleneck, trained from scratch."""

    def __init__(self, base_channels: int = 32, latent_channels: int = 64) -> None:
        super().__init__()

        def down(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 4, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(0.2, inplace=True),
            )

        def up(c_in: int, c_out: int) -> nn.Sequential:
            return nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        b = base_channels
        self.encoder = nn.Sequential(
            down(3, b), down(b, b * 2), down(b * 2, b * 4), nn.Conv2d(b * 4, latent_channels, 1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, b * 4, 1),
            up(b * 4, b * 2),
            up(b * 2, b),
            nn.ConvTranspose2d(b, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        decoded: torch.Tensor = self.decoder(self.encoder(images))
        return decoded


def _to_tensor(images: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(images.transpose(0, 3, 1, 2)).float() / 255.0


def train_autoencoder(
    train_images: np.ndarray,
    validation_images: np.ndarray,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
) -> tuple[ConvAutoencoder, list[float]]:
    """Train on normal images with MSE loss. Returns the model and per-epoch validation losses."""
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    validation_losses: list[float] = []
    for epoch in range(epochs):
        model.train()
        order = generator.permutation(len(train_images))
        for start in range(0, len(order), batch_size):
            batch = _to_tensor(train_images[order[start : start + batch_size]]).to(device)
            loss = torch.mean((model(batch) - batch) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        scheduler.step()
        errors = autoencoder_error_maps(model, validation_images, batch_size, device)
        validation_losses.append(float(errors.mean()))
        logger.debug("epoch %d validation loss %.5f", epoch, validation_losses[-1])
    return model, validation_losses


@torch.inference_mode()
def autoencoder_error_maps(
    model: ConvAutoencoder, images: np.ndarray, batch_size: int = 32, device: str = "cpu"
) -> np.ndarray:
    """Per-pixel squared reconstruction error (N, H, W)."""
    model.eval()
    maps = []
    for start in range(0, len(images), batch_size):
        batch = _to_tensor(images[start : start + batch_size]).to(device)
        maps.append(torch.mean((model(batch) - batch) ** 2, dim=1).cpu().numpy())
    return np.concatenate(maps, axis=0)
