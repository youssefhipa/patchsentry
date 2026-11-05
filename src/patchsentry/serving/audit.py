"""Append-only audit log of inspection decisions (PostgreSQL in Compose, SQLite locally)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Boolean, DateTime, Float, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class Inspection(Base):
    __tablename__ = "inspections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    image_sha256: Mapped[str] = mapped_column(String(64))
    score: Mapped[float] = mapped_column(Float)
    p_value: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    is_anomalous: Mapped[bool] = mapped_column(Boolean, index=True)
    model_version: Mapped[str] = mapped_column(String(64))
    latency_ms: Mapped[float] = mapped_column(Float)


class AuditLog:
    """Stores decisions with the model version, so any alarm can be traced and reproduced later.

    Only an image hash is stored, not the image: production images are often commercially sensitive
    and would grow the table quickly.
    """

    def __init__(self, database_url: str) -> None:
        if database_url.startswith("sqlite:///"):
            Path(database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(database_url, pool_pre_ping=True)
        Base.metadata.create_all(self._engine)

    def record(
        self,
        category: str,
        image_sha256: str,
        score: float,
        p_value: float,
        threshold: float,
        is_anomalous: bool,
        model_version: str,
        latency_ms: float,
    ) -> str:
        inspection_id = str(uuid.uuid4())
        with Session(self._engine) as session:
            session.add(
                Inspection(
                    id=inspection_id,
                    created_at=datetime.now(UTC),
                    category=category,
                    image_sha256=image_sha256,
                    score=score,
                    p_value=p_value,
                    threshold=threshold,
                    is_anomalous=is_anomalous,
                    model_version=model_version,
                    latency_ms=latency_ms,
                )
            )
            session.commit()
        return inspection_id

    def get(self, inspection_id: str) -> dict[str, object] | None:
        with Session(self._engine) as session:
            row = session.get(Inspection, inspection_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "created_at": row.created_at.isoformat(),
                "category": row.category,
                "image_sha256": row.image_sha256,
                "score": row.score,
                "p_value": row.p_value,
                "threshold": row.threshold,
                "is_anomalous": row.is_anomalous,
                "model_version": row.model_version,
                "latency_ms": row.latency_ms,
            }

    def count(self) -> int:
        with Session(self._engine) as session:
            return int(session.scalar(select(func.count()).select_from(Inspection)) or 0)

    def is_healthy(self) -> bool:
        try:
            self.count()
        except Exception:
            return False
        return True

    def close(self) -> None:
        self._engine.dispose()
