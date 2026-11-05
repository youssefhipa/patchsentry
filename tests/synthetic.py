"""Synthetic images for tests. NumPy only, so that both test groups can import this."""

from __future__ import annotations

import numpy as np

IMAGE_SIZE = 64


def textured_images(n: int, seed: int = 0) -> np.ndarray:
    """A low-noise repeating texture: the "normal" class in the end-to-end tests."""
    generator = np.random.default_rng(seed)
    xx, yy = np.meshgrid(np.arange(IMAGE_SIZE), np.arange(IMAGE_SIZE))
    base = (127 + 90 * np.sin(xx / 3.0) * np.cos(yy / 3.0))[..., None].repeat(3, axis=2)
    noise = generator.normal(0, 3, size=(n, IMAGE_SIZE, IMAGE_SIZE, 3))
    return np.clip(base[None] + noise, 0, 255).astype(np.uint8)


def with_defect(images: np.ndarray) -> np.ndarray:
    defective = images.copy()
    defective[:, 20:44, 20:44] = (255, 0, 255)
    return defective
