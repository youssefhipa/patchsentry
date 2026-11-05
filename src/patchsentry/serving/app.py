"""FastAPI application: validation, micro-batched inference, audit log, drift monitor, metrics."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.concurrency import run_in_threadpool

from patchsentry import __version__
from patchsentry.conformal import uniformity_ks
from patchsentry.data.preprocessing import resize_rgb
from patchsentry.serving.audit import AuditLog
from patchsentry.serving.batcher import MicroBatcher, QueueFullError
from patchsentry.serving.engine import InferenceEngine, InferenceResult
from patchsentry.serving.schemas import (
    CategoryInfo,
    DriftResponse,
    ErrorResponse,
    InspectionResponse,
)
from patchsentry.serving.settings import Settings

logger = logging.getLogger("patchsentry.api")

ALLOWED_FORMATS = {"JPEG", "PNG", "BMP", "TIFF"}
DRIFT_P_VALUE = 0.001
MIN_DRIFT_SAMPLES = 20
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


@dataclass
class _Metrics:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        self.requests = Counter(
            "patchsentry_inspections_total",
            "Finished inspections by category and decision.",
            ["category", "decision"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "patchsentry_inspection_latency_seconds",
            "End-to-end inspection latency.",
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "patchsentry_batch_size",
            "Images per micro-batch.",
            buckets=(1, 2, 4, 8, 16, 32, 64),
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            "patchsentry_queue_depth", "Requests waiting to be batched.", registry=self.registry
        )
        self.rejected = Counter(
            "patchsentry_rejected_total",
            "Rejected requests by reason.",
            ["reason"],
            registry=self.registry,
        )


def decode_image(data: bytes, image_size: int, max_pixels: int) -> np.ndarray:
    """Decode untrusted bytes into a resized RGB array. Raises HTTPException if invalid.

    The header is checked before any pixel is decoded, so a small file that expands to a huge image
    (a decompression bomb) is rejected without allocating the memory.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise HTTPException(415, f"Unsupported image format {image.format}.")
            width, height = image.size
            if width * height > max_pixels:
                raise HTTPException(413, f"Image has {width * height} pixels; limit {max_pixels}.")
            return resize_rgb(image, image_size)
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise HTTPException(400, "The file is not a readable image.") from error


def render_heatmap(image: np.ndarray, pixel_map: np.ndarray, threshold: float) -> str:
    """Overlay the anomaly map on the image and return a base64 PNG.

    Colours are scaled so that the top of the scale is twice the alarm threshold. Scaling per image
    would make every normal image look alarming, because its largest value would always be red.
    """
    scale = max(2.0 * threshold, 1e-6)
    level = np.clip(pixel_map / scale, 0.0, 1.0)
    coloured = cv2.applyColorMap((level * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.55, coloured, 0.45, 0)
    success, encoded = cv2.imencode(".png", overlay)
    if not success:
        raise RuntimeError("PNG encoding failed.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()
    metrics = _Metrics()
    logging.basicConfig(
        level=config.log_level,
        format=(
            '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        ),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = InferenceEngine(
            config.bundle_dir,
            categories=config.categories,
            use_quantized=config.use_quantized,
            providers=config.providers,
            index_kind=config.index_kind,
            intra_op_threads=config.intra_op_threads,
        )

        def run_batch(items: list[tuple[np.ndarray, str]]) -> list[InferenceResult]:
            images = np.stack([item[0] for item in items])
            return engine.infer_batch(images, [item[1] for item in items])

        batcher: MicroBatcher[tuple[np.ndarray, str], InferenceResult] = MicroBatcher(
            run_batch,
            max_batch_size=config.max_batch_size,
            max_wait_ms=config.max_wait_ms,
            max_queue_size=config.max_queue_size,
            on_batch=metrics.batch_size.observe,
        )
        await batcher.start()
        app.state.engine = engine
        app.state.batcher = batcher
        app.state.audit = AuditLog(config.database_url)
        app.state.windows = {name: deque(maxlen=config.drift_window) for name in engine.categories}
        app.state.alarms = {name: deque(maxlen=config.drift_window) for name in engine.categories}
        logger.info("Service ready")
        try:
            yield
        finally:
            await batcher.stop()
            app.state.audit.close()

    app = FastAPI(
        title="PatchSentry",
        version=__version__,
        description="Unsupervised visual anomaly detection with calibrated false alarm rates.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error request_id=%s path=%s", request_id, request.url.path)
            response = JSONResponse({"detail": "Internal server error."}, status_code=500)
        response.headers["x-request-id"] = request_id
        response.headers["x-process-time-ms"] = f"{(time.perf_counter() - started) * 1000:.2f}"
        return response

    def known_category(request: Request, category: str) -> InferenceEngine:
        engine: InferenceEngine = request.app.state.engine
        if category not in engine.categories:
            raise HTTPException(404, f"Unknown category {category!r}.")
        return engine

    @app.get("/healthz", tags=["ops"], summary="Liveness: the process is up.")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", tags=["ops"], summary="Readiness: the model and database are usable.")
    async def readyz(request: Request) -> JSONResponse:
        database_ok = await run_in_threadpool(request.app.state.audit.is_healthy)
        status_code = 200 if database_ok else 503
        return JSONResponse(
            {"status": "ready" if database_ok else "degraded", "database": database_ok},
            status_code=status_code,
        )

    @app.get("/metrics", tags=["ops"], summary="Prometheus metrics.")
    async def prometheus(request: Request) -> Response:
        metrics.queue_depth.set(request.app.state.batcher.queue_depth)
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/v1/categories", response_model=list[CategoryInfo], tags=["inspection"])
    async def list_categories(request: Request) -> list[CategoryInfo]:
        engine: InferenceEngine = request.app.state.engine
        return [
            CategoryInfo(
                name=name,
                threshold=model.thresholds[engine.default_alpha],
                target_false_alarm_rate=float(engine.default_alpha),
                calibration_size=model.calibration_scores.size,
                memory_bank_size=model.index.bank.shape[0],
            )
            for name, model in engine.categories.items()
        ]

    @app.post(
        "/v1/inspect/{category}",
        response_model=InspectionResponse,
        tags=["inspection"],
        summary="Inspect one image.",
        responses={
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
            504: {"model": ErrorResponse},
        },
    )
    async def inspect(
        request: Request,
        category: str,
        file: Annotated[UploadFile, File(description="JPEG, PNG, BMP or TIFF image.")],
        include_heatmap: Annotated[bool, Query()] = False,
    ) -> InspectionResponse:
        started = time.perf_counter()
        engine = known_category(request, category)

        data = await file.read(config.max_upload_bytes + 1)
        if len(data) > config.max_upload_bytes:
            metrics.rejected.labels("too_large").inc()
            raise HTTPException(413, f"Upload exceeds {config.max_upload_bytes} bytes.")
        if not data:
            metrics.rejected.labels("empty").inc()
            raise HTTPException(400, "The uploaded file is empty.")

        try:
            image = await run_in_threadpool(
                decode_image, data, engine.image_size, config.max_image_pixels
            )
        except HTTPException:
            metrics.rejected.labels("invalid_image").inc()
            raise

        try:
            result: InferenceResult = await asyncio.wait_for(
                request.app.state.batcher.submit((image, category)), config.request_timeout_s
            )
        except QueueFullError as error:
            metrics.rejected.labels("overloaded").inc()
            raise HTTPException(503, "Inference queue is full; retry later.") from error
        except TimeoutError as error:
            metrics.rejected.labels("timeout").inc()
            raise HTTPException(504, "Inference timed out.") from error

        latency = time.perf_counter() - started
        model_version = str(engine.manifest["version"])
        inspection_id = await run_in_threadpool(
            request.app.state.audit.record,
            category,
            hashlib.sha256(data).hexdigest(),
            result.score,
            result.p_value,
            result.threshold,
            result.is_anomalous,
            model_version,
            latency * 1000,
        )

        heatmap = None
        if include_heatmap:
            heatmap = await run_in_threadpool(
                lambda: render_heatmap(image, engine.pixel_map(result.patch_map), result.threshold)
            )

        request.app.state.windows[category].append(result.p_value)
        request.app.state.alarms[category].append(result.is_anomalous)
        decision = "anomalous" if result.is_anomalous else "normal"
        metrics.requests.labels(category, decision).inc()
        metrics.latency.observe(latency)
        return InspectionResponse(
            inspection_id=inspection_id,
            category=category,
            anomaly_score=result.score,
            p_value=result.p_value,
            threshold=result.threshold,
            target_false_alarm_rate=float(engine.default_alpha),
            is_anomalous=result.is_anomalous,
            latency_ms=latency * 1000,
            model_version=model_version,
            heatmap_png_base64=heatmap,
        )

    @app.get(
        "/v1/inspections/{inspection_id}", tags=["inspection"], summary="Read an audit record."
    )
    async def get_inspection(request: Request, inspection_id: str) -> dict[str, object]:
        record: dict[str, object] | None = await run_in_threadpool(
            request.app.state.audit.get, inspection_id
        )
        if record is None:
            raise HTTPException(404, "Inspection not found.")
        return record

    @app.get("/v1/drift/{category}", response_model=DriftResponse, tags=["monitoring"])
    async def drift(request: Request, category: str) -> DriftResponse:
        known_category(request, category)
        window = list(request.app.state.windows[category])
        alarms = list(request.app.state.alarms[category])
        if len(window) < MIN_DRIFT_SAMPLES:
            return DriftResponse(category=category, window_size=len(window), drift_suspected=False)
        statistic, p_value = uniformity_ks(np.asarray(window))
        return DriftResponse(
            category=category,
            window_size=len(window),
            ks_statistic=statistic,
            ks_p_value=p_value,
            drift_suspected=p_value < DRIFT_P_VALUE,
            alarm_rate=float(np.mean(alarms)),
        )

    return app
