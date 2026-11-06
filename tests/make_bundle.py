"""Build a tiny real bundle for the serving tests. Runs as a separate process, because it needs
PyTorch and the serving tests hold FAISS (see ``models/torch_index.py``)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from patchsentry.conformal import conformal_threshold
from patchsentry.models.coreset import greedy_k_center
from patchsentry.models.features import PatchFeatureExtractor, extract_features
from patchsentry.models.patchcore_index import PatchCoreIndex
from patchsentry.pipeline.export import export_onnx, quantize_onnx
from tests.synthetic import IMAGE_SIZE, textured_images


def main(directory: Path) -> None:
    # The network is untrained, so its weights must be seeded: otherwise every test session gets a
    # different bundle, and the margin between normal and defective images changes from run to run.
    torch.manual_seed(0)
    extractor = PatchFeatureExtractor("resnet18", target_dim=256, pretrained=False)
    export_onnx(extractor, directory / "backbone.onnx", IMAGE_SIZE)
    quantize_onnx(
        directory / "backbone.onnx", directory / "backbone.quant.onnx", textured_images(8, seed=7)
    )
    features = extract_features(extractor, textured_images(30, seed=1), batch_size=8)
    pool = features.reshape(-1, features.shape[-1])
    bank = pool[greedy_k_center(pool, 200, projection_dim=32)]
    index = PatchCoreIndex(bank, index_kind="torch", reweight_neighbours=3)
    index.save(directory / "categories" / "widget")

    calibration = extract_features(extractor, textured_images(40, seed=2), batch_size=8)
    scores, _ = index.score(calibration)
    thresholds = {str(a): conformal_threshold(scores, a) for a in (0.05, 0.1)}
    manifest = {
        "version": "test",
        "model": "backbone.onnx",
        "quantized_model": "backbone.quant.onnx",
        "image_size": IMAGE_SIZE,
        "gaussian_sigma": 2.0,
        "default_alpha": "0.05",
        "categories": {
            "widget": {
                "calibration_scores": [float(s) for s in scores],
                "thresholds": thresholds,
                "quantized_calibration_scores": [float(s) for s in scores],
                "quantized_thresholds": thresholds,
            }
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
