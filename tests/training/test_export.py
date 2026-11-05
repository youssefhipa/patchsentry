from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch

from patchsentry.data.preprocessing import normalize_batch
from patchsentry.models.features import PatchFeatureExtractor
from patchsentry.pipeline.export import export_onnx
from tests.synthetic import IMAGE_SIZE, textured_images


@pytest.mark.parametrize("batch", [1, 5])
def test_onnx_matches_pytorch_for_any_batch_size(tmp_path: Path, batch: int) -> None:
    extractor = PatchFeatureExtractor("resnet18", target_dim=128, pretrained=False)
    path = export_onnx(extractor, tmp_path / "model.onnx", IMAGE_SIZE)
    images = normalize_batch(textured_images(batch))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"images": images})[0]
    with torch.inference_mode():
        expected = extractor(torch.from_numpy(images)).numpy()
    assert np.allclose(output, expected, atol=1e-4)
