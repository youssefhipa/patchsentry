"""Image preprocessing shared by training and serving.

This module depends only on NumPy and Pillow. Training and the PyTorch-free service call the same
functions, which removes a common production bug: training/serving skew from two preprocessing
implementations that differ slightly (for example in their resize filter).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def resize_rgb(image: Image.Image, image_size: int) -> np.ndarray:
    """Convert to RGB and resize to a square uint8 HWC array.

    There is no centre crop: a defect at the edge of a part is still a defect, and cropping would
    silently make it undetectable. The cost is a mild aspect-ratio distortion, which is harmless
    because training and inference distort identically.
    """
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def load_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        return resize_rgb(image, image_size)


def load_mask(path: Path, image_size: int) -> np.ndarray:
    """Load a segmentation mask as a boolean HW array. Any non-zero pixel counts as anomalous."""
    with Image.open(path) as mask:
        # Nearest-neighbour keeps the mask binary; interpolating would invent partial labels.
        resized = mask.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)
        return np.asarray(resized) > 0


def normalize_batch(images: np.ndarray) -> np.ndarray:
    """Convert uint8 NHWC images into ImageNet-normalised float32 NCHW."""
    if images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"Expected an NHWC RGB batch, got shape {images.shape}")
    chw = images.transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    return (chw - IMAGENET_MEAN) / IMAGENET_STD
