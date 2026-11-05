"""FAISS index behaviour, and the ONNX graphs as the service loads them."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from patchsentry.data.preprocessing import normalize_batch
from patchsentry.models.patchcore_index import FAISS_INDEX_KINDS, PatchCoreIndex
from patchsentry.serving.engine import InferenceEngine
from tests.synthetic import textured_images


@pytest.mark.parametrize("kind", FAISS_INDEX_KINDS)
def test_approximate_indexes_nearly_match_exact_search(kind: str) -> None:
    generator = np.random.default_rng(0)
    centres = generator.normal(size=(20, 32)) * 4
    bank = centres[generator.integers(20, size=3000)] + generator.normal(size=(3000, 32))
    bank = bank.astype(np.float32)
    features = (bank[:400] + 0.1 * generator.normal(size=(400, 32))).astype(np.float32)
    features = features.reshape(1, 20, 20, 32)
    reference = PatchCoreIndex(bank, "flat").score(features)[1]
    approximate = PatchCoreIndex(bank, kind).score(features)[1]
    assert np.mean(np.abs(reference - approximate) < 1e-3) > 0.9


def test_unknown_index_kind_raises() -> None:
    with pytest.raises(ValueError, match="Unknown index kind"):
        PatchCoreIndex(np.zeros((5, 4), dtype=np.float32), "bogus")


def test_quantized_graph_keeps_shape_stays_close_and_is_smaller(bundle_dir: Path) -> None:
    images = normalize_batch(textured_images(4, seed=50))
    baseline = ort.InferenceSession(
        str(bundle_dir / "backbone.onnx"), providers=["CPUExecutionProvider"]
    ).run(None, {"images": images})[0]
    quantized = ort.InferenceSession(
        str(bundle_dir / "backbone.quant.onnx"), providers=["CPUExecutionProvider"]
    ).run(None, {"images": images})[0]
    assert baseline.shape == quantized.shape
    assert np.corrcoef(baseline.ravel(), quantized.ravel())[0, 1] > 0.9
    float_size = (bundle_dir / "backbone.onnx").stat().st_size
    assert (bundle_dir / "backbone.quant.onnx").stat().st_size < float_size


def test_engine_mixed_batch_equals_separate_calls(bundle_dir: Path) -> None:
    engine = InferenceEngine(bundle_dir)
    images = textured_images(3, seed=60)
    together = engine.infer_batch(images, ["widget"] * 3)
    alone = [engine.infer_batch(images[i : i + 1], ["widget"])[0] for i in range(3)]
    for first, second in zip(together, alone, strict=True):
        assert first.score == pytest.approx(second.score, rel=1e-4)
    with pytest.raises(ValueError, match="categories"):
        engine.infer_batch(images, ["widget"])


def test_engine_rejects_categories_missing_from_the_bundle(bundle_dir: Path) -> None:
    with pytest.raises(ValueError, match="not in the bundle"):
        InferenceEngine(bundle_dir, categories=["missing"])


def test_engine_rejects_unavailable_providers(bundle_dir: Path) -> None:
    with pytest.raises(ValueError, match="providers"):
        InferenceEngine(bundle_dir, providers=["NoSuchExecutionProvider"])


def test_serving_package_does_not_import_torch() -> None:
    """The container image contains no PyTorch, so serving must not depend on it."""
    code = (
        "import sys; import patchsentry.serving.app; "
        "import patchsentry.models.patchcore_index as p; p.build_index; "
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
