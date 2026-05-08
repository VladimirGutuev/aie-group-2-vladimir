from __future__ import annotations

from pydantic import BaseModel, Field


class DetectionItem(BaseModel):
    bbox: list[int] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    text: str
    confidence: float


class PredictResponse(BaseModel):
    plate: str | None = Field(None, description="Best plate string, or null if not found")
    confidence: float = 0.0
    detections: list[DetectionItem] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"


class MetricsResponse(BaseModel):
    requests_total: int
    predict_total: int
    predict_failed: int
    plates_recognized: int
