from __future__ import annotations

import io
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from ..pipeline import AnprPipeline
from .schemas import DetectionItem, HealthResponse, MetricsResponse, PredictResponse

logger = logging.getLogger("anpr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(
    title="Smart Parking ANPR",
    description="License plate recognition service for SCUD/parking automation.",
    version="0.1.0",
)

_pipeline: AnprPipeline | None = None
_metrics: dict[str, int] = {
    "requests_total": 0,
    "predict_total": 0,
    "predict_failed": 0,
    "plates_recognized": 0,
}


def get_pipeline() -> AnprPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AnprPipeline()
    return _pipeline


def _read_image(data: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Invalid image file") from exc
    return np.array(image)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    return MetricsResponse(**_metrics)


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    _metrics["requests_total"] += 1
    _metrics["predict_total"] += 1

    if not file.content_type or not file.content_type.startswith("image/"):
        _metrics["predict_failed"] += 1
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        data = await file.read()
        image = _read_image(data)
        pipeline = get_pipeline()
        reads = pipeline.run(image)
    except HTTPException:
        _metrics["predict_failed"] += 1
        raise
    except Exception:
        _metrics["predict_failed"] += 1
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail="Internal pipeline error")

    _metrics["plates_recognized"] += len(reads)
    logger.info("predict: %d plates found", len(reads))

    if not reads:
        return PredictResponse()

    best = max(reads, key=lambda r: r.confidence)
    return PredictResponse(
        plate=best.text,
        confidence=best.confidence,
        detections=[
            DetectionItem(bbox=list(r.bbox), text=r.text, confidence=r.confidence)
            for r in reads
        ],
    )
