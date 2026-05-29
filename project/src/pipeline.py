from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import settings
from .models.detector import Detection, PlateDetector
from .models.ocr import OcrResult, PlateOCR


@dataclass
class PlateRead:
    bbox: tuple[int, int, int, int]
    text: str
    confidence: float


def _build_ocr():
    """Construct the OCR engine selected in settings (easyocr | crnn)."""
    if settings.ocr_engine == "crnn":
        from .models.crnn_ocr import CrnnOCR

        return CrnnOCR(weights=settings.crnn_weights, gpu=settings.use_gpu)
    return PlateOCR(
        langs=settings.ocr_lang_list,
        min_confidence=settings.min_confidence,
        gpu=settings.use_gpu,
    )


class AnprPipeline:
    """Two-stage license plate recognition pipeline: detector -> OCR."""

    def __init__(
        self,
        detector: PlateDetector | None = None,
        ocr=None,
    ) -> None:
        self.detector = detector or PlateDetector(
            weights=settings.detector_weights,
            enabled=settings.use_detector,
        )
        self.ocr = ocr or _build_ocr()

    def run(self, image: np.ndarray) -> list[PlateRead]:
        detections: list[Detection] = self.detector.detect(image)
        plates: list[PlateRead] = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            ocr_results: list[OcrResult] = self.ocr.read(crop)
            if not ocr_results:
                continue
            best = max(ocr_results, key=lambda r: r.confidence)
            plates.append(
                PlateRead(
                    bbox=det.bbox,
                    text=best.text,
                    confidence=best.confidence * det.confidence,
                )
            )
        return plates
