from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Latin equivalents of the Russian plate alphabet (А В Е К М Н О Р С Т У Х) + digits.
RU_ALLOWLIST = "ABCEHKMOPTXY0123456789"


@dataclass
class OcrResult:
    text: str
    confidence: float


class PlateOCR:
    """EasyOCR wrapper with lazy loading and optional preprocessing/allowlist."""

    def __init__(
        self,
        langs: list[str],
        min_confidence: float = 0.0,
        preprocess: bool = False,
        allowlist: str | None = None,
        gpu: bool = False,
    ) -> None:
        self.langs = langs
        self.min_confidence = min_confidence
        self.preprocess = preprocess
        self.allowlist = allowlist
        self.gpu = gpu
        self._reader = None

    def _load(self) -> None:
        if self._reader is not None:
            return
        import easyocr

        self._reader = easyocr.Reader(self.langs, gpu=self.gpu)

    def _prep(self, image: np.ndarray) -> np.ndarray:
        import cv2

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        longest = max(h, w)
        if longest < 200:
            scale = 200.0 / longest
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def read(self, image: np.ndarray) -> list[OcrResult]:
        self._load()
        img = self._prep(image) if self.preprocess else image
        kwargs = {"allowlist": self.allowlist} if self.allowlist else {}
        raw = self._reader.readtext(img, **kwargs)
        results: list[OcrResult] = []
        for _bbox, text, conf in raw:
            if conf < self.min_confidence:
                continue
            cleaned = "".join(ch for ch in text if ch.isalnum()).upper()
            if not cleaned:
                continue
            results.append(OcrResult(text=cleaned, confidence=float(conf)))
        return results
