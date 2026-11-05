"""Service configuration, read from ``PATCHSENTRY_*`` environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PATCHSENTRY_", env_file=".env", extra="ignore")

    bundle_dir: Path = Path("artifacts/bundle")
    categories: list[str] | None = None
    use_quantized: bool = False
    providers: list[str] = ["CPUExecutionProvider"]
    index_kind: str = "flat"
    intra_op_threads: int = Field(default=0, ge=0)

    max_batch_size: int = Field(default=16, ge=1)
    max_wait_ms: float = Field(default=5.0, ge=0)
    max_queue_size: int = Field(default=256, ge=1)
    request_timeout_s: float = Field(default=30.0, gt=0)

    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(default=40_000_000, ge=1)

    database_url: str = "sqlite:///artifacts/audit.db"
    drift_window: int = Field(default=200, ge=20)
    log_level: str = "INFO"
