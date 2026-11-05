"""Inference engine: ONNX Runtime for features, FAISS for scoring, conformal rules for decisions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from patchsentry.conformal import conformal_p_values
from patchsentry.data.preprocessing import normalize_batch
from patchsentry.models.patchcore_index import PatchCoreIndex, upsample_maps

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class CategoryModel:
    index: PatchCoreIndex
    calibration_scores: np.ndarray
    thresholds: dict[str, float]


@dataclass(frozen=True)
class InferenceResult:
    score: float
    p_value: float
    threshold: float
    is_anomalous: bool
    patch_map: np.ndarray


class InferenceEngine:
    """Loads a model bundle and runs batched inference.

    All categories share one backbone graph, so a micro-batch can mix categories: features are
    extracted once for the whole batch, then each image is scored against its own memory bank.
    """

    def __init__(
        self,
        bundle_dir: Path,
        categories: list[str] | None = None,
        use_quantized: bool = False,
        providers: list[str] | None = None,
        index_kind: str = "flat",
        intra_op_threads: int = 0,
    ) -> None:
        manifest_path = bundle_dir / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"No model bundle at {bundle_dir}: {MANIFEST_NAME} is missing.")
        self.manifest = json.loads(manifest_path.read_text())
        self.image_size = int(self.manifest["image_size"])
        self.default_alpha = str(self.manifest["default_alpha"])
        self.gaussian_sigma = float(self.manifest["gaussian_sigma"])

        model_key = "quantized_model" if use_quantized else "model"
        model_path = bundle_dir / self.manifest[model_key]
        options = ort.SessionOptions()
        options.intra_op_num_threads = intra_op_threads
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        chosen = [p for p in (providers or ["CPUExecutionProvider"]) if p in available]
        if not chosen:
            raise ValueError(f"None of the providers {providers} are available: {available}")
        self.session = ort.InferenceSession(str(model_path), options, providers=chosen)
        self.input_name = self.session.get_inputs()[0].name

        names = categories or sorted(self.manifest["categories"])
        self.categories: dict[str, CategoryModel] = {}
        for name in names:
            entry = self.manifest["categories"].get(name)
            if entry is None:
                raise ValueError(f"Category {name!r} is not in the bundle.")
            # Quantisation shifts the scores slightly, so a graph is only ever paired with the
            # thresholds calibrated on its own output. Float thresholds would void the bound.
            prefix = "quantized_" if use_quantized else ""
            self.categories[name] = CategoryModel(
                index=PatchCoreIndex.load(bundle_dir / "categories" / name, index_kind),
                calibration_scores=np.asarray(
                    entry[f"{prefix}calibration_scores"], dtype=np.float64
                ),
                thresholds={str(k): float(v) for k, v in entry[f"{prefix}thresholds"].items()},
            )
        logger.info(
            "Engine ready: %d categories, model=%s, providers=%s, index=%s",
            len(self.categories),
            model_path.name,
            self.session.get_providers(),
            index_kind,
        )

    def extract(self, images: np.ndarray) -> np.ndarray:
        """Return features (N, H, W, D) for uint8 NHWC images."""
        features = self.session.run(None, {self.input_name: normalize_batch(images)})[0]
        return np.asarray(features).transpose(0, 2, 3, 1)

    def infer_batch(self, images: np.ndarray, categories: list[str]) -> list[InferenceResult]:
        """Score a batch whose images may belong to different categories."""
        if len(images) != len(categories):
            raise ValueError(f"{len(images)} images but {len(categories)} categories.")
        features = self.extract(images)

        positions: dict[str, list[int]] = {}
        for position, category in enumerate(categories):
            positions.setdefault(category, []).append(position)

        results: list[InferenceResult | None] = [None] * len(images)
        for category, members in positions.items():
            model = self.categories[category]
            threshold = model.thresholds[self.default_alpha]
            scores, maps = model.index.score(features[members])
            p_values = conformal_p_values(scores, model.calibration_scores)
            for row, position in enumerate(members):
                score = float(scores[row])
                results[position] = InferenceResult(
                    score=score,
                    p_value=float(p_values[row]),
                    threshold=threshold,
                    is_anomalous=score > threshold,
                    patch_map=maps[row],
                )
        return [result for result in results if result is not None]

    def pixel_map(self, patch_map: np.ndarray) -> np.ndarray:
        """Upsample one patch map to the model's input resolution."""
        return np.asarray(upsample_maps(patch_map[None], self.image_size, self.gaussian_sigma)[0])
