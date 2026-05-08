from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float


class PlateDetector:
    """YOLOv8 wrapper. Loaded lazily; if disabled returns whole frame as single bbox."""

    def __init__(self, weights: str, enabled: bool = False) -> None:
        self.weights = weights
        self.enabled = enabled
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO

        self._model = YOLO(self.weights)

    def detect(self, image: np.ndarray) -> list[Detection]:
        if not self.enabled:
            h, w = image.shape[:2]
            return [Detection(bbox=(0, 0, w, h), confidence=1.0)]

        self._load()
        results = self._model(image, verbose=False)
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                detections.append(
                    Detection(
                        bbox=(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        confidence=conf,
                    )
                )
        return detections
