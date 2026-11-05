"""Locally aware patch features from a pretrained CNN, written to export cleanly to ONNX."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from patchsentry.data.preprocessing import normalize_batch

_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "wide_resnet50_2": (models.wide_resnet50_2, models.Wide_ResNet50_2_Weights.IMAGENET1K_V1),
}


def channel_bins(in_channels: int, out_channels: int) -> torch.Tensor:
    """Weights (out, in, 1, 1) of a 1x1 convolution that averages channels into equal-width bins.

    This matches PatchCore's adaptive pooling over the channel axis. Writing it as a fixed
    convolution means the graph needs no dynamic adaptive pooling, which ONNX exports poorly.
    """
    weights = torch.zeros(out_channels, in_channels, 1, 1)
    for out_index in range(out_channels):
        start = out_index * in_channels // out_channels
        end = max(start + 1, -((-(out_index + 1) * in_channels) // out_channels))
        weights[out_index, start:end] = 1.0 / (end - start)
    return weights


class PatchFeatureExtractor(nn.Module):
    """Map images (N, 3, S, S) to patch features (N, D, S/8, S/8).

    Uses layers 2 and 3 of a ResNet: layer 1 is too generic and layer 4 is biased towards ImageNet
    classes and too coarse to localise. Average pooling over a neighbourhood gives each patch some
    context and tolerance to small misalignments without losing resolution.
    """

    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        target_dim: int = 1024,
        neighbourhood: int = 3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; expected one of {list(_BACKBONES)}")
        if neighbourhood % 2 == 0:
            raise ValueError(f"neighbourhood must be odd to keep the grid, got {neighbourhood}")
        factory, weights = _BACKBONES[backbone]
        net = factory(weights=weights if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.pool = nn.AvgPool2d(neighbourhood, stride=1, padding=neighbourhood // 2)

        with torch.no_grad():
            probe = self._concatenated(torch.zeros(1, 3, 64, 64))
        in_channels = probe.shape[1]
        self.output_dim = min(in_channels, target_dim)
        self.projection: nn.Module
        if in_channels > target_dim:
            conv = nn.Conv2d(in_channels, target_dim, kernel_size=1, bias=False)
            conv.weight.data.copy_(channel_bins(in_channels, target_dim))
            self.projection = conv
        else:
            self.projection = nn.Identity()
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _concatenated(self, images: torch.Tensor) -> torch.Tensor:
        mid = self.layer2(self.stem(images))
        deep = self.layer3(mid)
        upsampled = F.interpolate(self.pool(deep), scale_factor=2.0, mode="bilinear")
        return torch.cat([self.pool(mid), upsampled], dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        projected: torch.Tensor = self.projection(self._concatenated(images))
        return projected


@torch.inference_mode()
def iter_features(
    extractor: PatchFeatureExtractor,
    images: np.ndarray,
    batch_size: int = 32,
    device: str = "cpu",
) -> Iterator[np.ndarray]:
    """Yield feature batches (B, H, W, D) as float32 arrays for uint8 NHWC images."""
    extractor.to(device)
    for start in range(0, len(images), batch_size):
        batch = torch.from_numpy(normalize_batch(images[start : start + batch_size])).to(device)
        yield extractor(batch).permute(0, 2, 3, 1).cpu().numpy()


def extract_features(
    extractor: PatchFeatureExtractor,
    images: np.ndarray,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    return np.concatenate(list(iter_features(extractor, images, batch_size, device)), axis=0)


def sample_patches(
    extractor: PatchFeatureExtractor,
    images: np.ndarray,
    max_patches: int,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 0,
) -> np.ndarray:
    """Extract a uniform random subset of patches (M, D), allocating only what is kept.

    Holding every patch of 900 images would cost several GB, so patches are subsampled per image
    while extracting, rather than after.
    """
    generator = np.random.default_rng(seed)
    per_image: int | None = None
    kept: list[np.ndarray] = []
    for batch in iter_features(extractor, images, batch_size, device):
        n, height, width, dimension = batch.shape
        total = height * width
        if per_image is None:
            per_image = min(total, max(1, int(np.ceil(max_patches / len(images)))))
        flat = batch.reshape(n, total, dimension)
        for row in flat:
            kept.append(row[generator.choice(total, per_image, replace=False)].copy())
    return np.concatenate(kept, axis=0)
