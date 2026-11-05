"""Public API schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InspectionResponse(BaseModel):
    inspection_id: str
    category: str
    anomaly_score: float = Field(description="PatchCore image score. Higher is more anomalous.")
    p_value: float = Field(
        description="Conformal p-value: how unusual this score is among held-out normal images."
    )
    threshold: float = Field(description="Conformal alarm threshold in force for this decision.")
    target_false_alarm_rate: float
    is_anomalous: bool
    latency_ms: float
    model_version: str
    heatmap_png_base64: str | None = Field(
        default=None, description="Colour heatmap, returned when include_heatmap=true."
    )


class CategoryInfo(BaseModel):
    name: str
    threshold: float
    target_false_alarm_rate: float
    calibration_size: int
    memory_bank_size: int


class DriftResponse(BaseModel):
    category: str
    window_size: int
    ks_statistic: float | None = None
    ks_p_value: float | None = None
    drift_suspected: bool = Field(
        description="True if p-values in the window are far from uniform (KS p-value below 0.001)."
    )
    alarm_rate: float | None = None


class ErrorResponse(BaseModel):
    detail: str
