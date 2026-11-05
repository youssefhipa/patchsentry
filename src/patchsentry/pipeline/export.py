"""Export the feature extractor to ONNX and quantise it."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import onnx
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from patchsentry.data.preprocessing import normalize_batch
from patchsentry.models.features import PatchFeatureExtractor

logger = logging.getLogger(__name__)

INPUT_NAME = "images"
OUTPUT_NAME = "features"


def export_onnx(extractor: PatchFeatureExtractor, path: Path, image_size: int) -> Path:
    """Export with a dynamic batch axis, as one self-contained file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    extractor = extractor.to("cpu").eval()
    dummy = torch.zeros(2, 3, image_size, image_size)
    torch.onnx.export(
        extractor,
        (dummy,),
        str(path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    onnx.checker.check_model(str(path))
    logger.info("Exported %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return path


class _ImageReader(CalibrationDataReader):  # type: ignore[misc]
    """Feeds real, normalised images to the quantisation calibrator, one batch at a time."""

    def __init__(self, images: np.ndarray, batch_size: int = 8) -> None:
        self._batches = iter(
            [
                {INPUT_NAME: normalize_batch(images[start : start + batch_size])}
                for start in range(0, len(images), batch_size)
            ]
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._batches, None)


def quantize_onnx(model_path: Path, output_path: Path, calibration_images: np.ndarray) -> Path:
    """Static 8-bit quantisation in QDQ format, with weights quantised per channel.

    Static, because dynamic quantisation only helps MatMul and LSTM-heavy graphs and a ResNet is
    nearly all Conv. Activation ranges are calibrated with percentiles on normal images, which
    keeps one extreme pixel from wasting the 8-bit range.
    """
    with tempfile.TemporaryDirectory() as temp:
        prepared = Path(temp) / "prepared.onnx"
        quant_pre_process(str(model_path), str(prepared), skip_symbolic_shape=True)
        quantize_static(
            str(prepared),
            str(output_path),
            _ImageReader(calibration_images),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=CalibrationMethod.Percentile,
            extra_options={"CalibPercentile": 99.99},
        )
    logger.info("Quantised %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)
    return output_path
