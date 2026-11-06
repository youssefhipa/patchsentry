"""End-to-end API tests against a real ONNX model and FAISS index, with no mocks."""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from patchsentry.serving.app import create_app
from patchsentry.serving.settings import Settings
from tests.synthetic import textured_images, with_defect


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _settings(bundle_dir: Path, tmp_path: Path, **overrides) -> Settings:
    values = {
        "bundle_dir": bundle_dir,
        "database_url": f"sqlite:///{tmp_path}/audit.db",
        "max_wait_ms": 1.0,
        "_env_file": None,
    }
    return Settings(**{**values, **overrides})


@pytest.fixture
def client(bundle_dir: Path, tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(_settings(bundle_dir, tmp_path))) as test_client:
        yield test_client


def _inspect(client: TestClient, data: bytes, category: str = "widget", **params):
    files = {"file": ("image.png", data, "image/png")}
    return client.post(f"/v1/inspect/{category}", files=files, params=params)


def test_normal_image_passes_and_defect_raises_an_alarm(client: TestClient) -> None:
    normal = _inspect(client, _png(textured_images(1, seed=20)[0])).json()
    defect = _inspect(client, _png(with_defect(textured_images(1, seed=20))[0])).json()
    assert normal["is_anomalous"] is False
    assert defect["is_anomalous"] is True
    assert defect["anomaly_score"] > normal["anomaly_score"]
    assert defect["p_value"] < normal["p_value"]
    assert defect["target_false_alarm_rate"] == 0.05


def test_decision_is_recorded_in_the_audit_log(client: TestClient) -> None:
    data = _png(textured_images(1, seed=21)[0])
    body = _inspect(client, data).json()
    record = client.get(f"/v1/inspections/{body['inspection_id']}").json()
    assert record["score"] == pytest.approx(body["anomaly_score"])
    assert record["category"] == "widget"
    assert record["model_version"] == "test"
    assert len(record["image_sha256"]) == 64
    assert client.get("/v1/inspections/does-not-exist").status_code == 404


def test_heatmap_is_only_returned_on_request_and_is_a_valid_png(client: TestClient) -> None:
    data = _png(with_defect(textured_images(1, seed=22))[0])
    assert _inspect(client, data).json()["heatmap_png_base64"] is None
    encoded = _inspect(client, data, include_heatmap="true").json()["heatmap_png_base64"]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as heatmap:
        assert heatmap.format == "PNG"


@pytest.mark.parametrize(
    ("data", "status"),
    [(b"", 400), (b"this is not an image", 400), (b"\x89PNG\r\n\x1a\ntruncated", 400)],
)
def test_invalid_uploads_get_client_errors(client: TestClient, data: bytes, status: int) -> None:
    assert _inspect(client, data).status_code == status


def test_unsupported_format_is_415(client: TestClient) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, format="GIF")
    assert _inspect(client, buffer.getvalue()).status_code == 415


def test_unknown_category_is_404(client: TestClient) -> None:
    assert _inspect(client, _png(textured_images(1)[0]), category="nope").status_code == 404


def test_missing_file_is_422(client: TestClient) -> None:
    assert client.post("/v1/inspect/widget").status_code == 422


def test_oversized_upload_is_413(bundle_dir: Path, tmp_path: Path) -> None:
    settings = _settings(bundle_dir, tmp_path, max_upload_bytes=100)
    with TestClient(create_app(settings)) as client:
        assert _inspect(client, _png(textured_images(1)[0])).status_code == 413


def test_decompression_bomb_is_rejected_before_decoding(bundle_dir: Path, tmp_path: Path) -> None:
    settings = _settings(bundle_dir, tmp_path, max_image_pixels=1000)
    bomb = _png(np.zeros((500, 500, 3), dtype=np.uint8))  # compresses to very few bytes
    assert len(bomb) < 5000
    with TestClient(create_app(settings)) as client:
        response = _inspect(client, bomb)
    assert response.status_code == 413
    assert "pixels" in response.json()["detail"]


def test_categories_and_operational_endpoints(client: TestClient) -> None:
    [category] = client.get("/v1/categories").json()
    assert category["name"] == "widget"
    assert category["memory_bank_size"] == 200
    assert category["calibration_size"] == 40
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"


def test_request_id_is_echoed_or_generated(client: TestClient) -> None:
    assert client.get("/healthz", headers={"x-request-id": "abc"}).headers["x-request-id"] == "abc"
    assert client.get("/healthz").headers["x-request-id"]


def test_metrics_count_decisions_and_rejections(client: TestClient) -> None:
    _inspect(client, _png(textured_images(1, seed=23)[0]))
    _inspect(client, b"junk")
    text = client.get("/metrics").text
    assert 'patchsentry_inspections_total{category="widget"' in text
    assert 'patchsentry_rejected_total{reason="invalid_image"} 1.0' in text
    assert "patchsentry_batch_size_bucket" in text


def test_drift_monitor_flags_a_stream_of_shifted_images(bundle_dir: Path, tmp_path: Path) -> None:
    with TestClient(create_app(_settings(bundle_dir, tmp_path, drift_window=30))) as client:
        assert client.get("/v1/drift/widget").json()["ks_statistic"] is None, "too few samples"
        for image in with_defect(textured_images(30, seed=24)):
            _inspect(client, _png(image))
        drift = client.get("/v1/drift/widget").json()
    assert drift["window_size"] == 30
    assert drift["drift_suspected"] is True
    assert drift["alarm_rate"] == 1.0


def test_concurrent_requests_get_the_same_scores_as_sequential_ones(
    bundle_dir: Path, tmp_path: Path
) -> None:
    """Batching must never change a result or cross-wire responses between callers."""
    # Defects of different sizes give each image a clearly different score, so that a response
    # delivered to the wrong caller cannot pass as float noise.
    arrays = textured_images(8, seed=30)
    for index, size in enumerate(range(4, 36, 4)):
        arrays[index, 10 : 10 + size, 10 : 10 + size] = (255, 0, 255)
    images = [_png(array) for array in arrays]

    with TestClient(create_app(_settings(bundle_dir, tmp_path, max_batch_size=1))) as client:
        sequential = np.array([_inspect(client, data).json()["anomaly_score"] for data in images])

    settings = _settings(bundle_dir, tmp_path, max_batch_size=8, max_wait_ms=100.0)
    with TestClient(create_app(settings)) as client, ThreadPoolExecutor(8) as pool:
        responses = list(pool.map(lambda data: _inspect(client, data).json(), images))
        batch_text = client.get("/metrics").text
    batched = np.array([response["anomaly_score"] for response in responses])

    # Batch shape changes float32 rounding in the convolution and in FAISS. Measured worst case:
    # about 1e-4 on Apple M3 and 1.1e-3 on the x86 CI runner. Allow 5e-3 for that noise...
    noise = 5e-3
    assert np.abs(batched - sequential).max() < noise
    # ...and require every pair of images to differ by more, so the check above can catch a swap.
    gaps = np.abs(sequential[:, None] - sequential[None, :])[~np.eye(len(sequential), dtype=bool)]
    assert gaps.min() > 2 * noise, f"test images are not distinct enough: min gap {gaps.min()}"
    assert 'patchsentry_batch_size_bucket{le="1.0"} 8.0' not in batch_text, "nothing was batched"


def test_quantized_model_serves_requests(bundle_dir: Path, tmp_path: Path) -> None:
    with TestClient(create_app(_settings(bundle_dir, tmp_path, use_quantized=True))) as client:
        response = _inspect(client, _png(textured_images(1, seed=40)[0]))
    assert response.status_code == 200


def test_missing_bundle_fails_at_startup(tmp_path: Path) -> None:
    settings = _settings(tmp_path / "nowhere", tmp_path)
    with (
        pytest.raises(FileNotFoundError, match="No model bundle"),
        TestClient(create_app(settings)),
    ):
        pass
