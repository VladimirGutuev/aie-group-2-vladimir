from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OcrResult:
    text: str
    confidence: float


class PlateOCR:
    """EasyOCR wrapper with lazy loading."""

    def __init__(self, langs: list[str], min_confidence: float = 0.0) -> None:
        self.langs = langs
        self.min_confidence = min_confidence
        self._reader = None

    def _load(self) -> None:
        if self._reader is not None:
            return
        import easyocr

        self._reader = easyocr.Reader(self.langs, gpu=False)

    def read(self, image: np.ndarray) -> list[OcrResult]:
        self._load()
        raw = self._reader.readtext(image)
        results: list[OcrResult] = []
        for _bbox, text, conf in raw:
            if conf < self.min_confidence:
                continue
            cleaned = "".join(ch for ch in text if ch.isalnum()).upper()
            if not cleaned:
                continue
            results.append(OcrResult(text=cleaned, confidence=float(conf)))
        return results
