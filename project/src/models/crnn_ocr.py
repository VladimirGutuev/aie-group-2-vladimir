from __future__ import annotations

from pathlib import Path

import numpy as np

from .ocr import OcrResult


class CrnnOCR:
    """Inference wrapper around a trained CRNN checkpoint.

    Exposes the same ``read(image) -> list[OcrResult]`` interface as PlateOCR,
    so it is a drop-in OCR engine for the pipeline. Expects a cropped plate
    image (RGB or grayscale numpy array).
    """

    def __init__(self, weights: str = "artifacts/crnn.pt", gpu: bool = False) -> None:
        self.weights = weights
        self.gpu = gpu
        self._model = None
        self._codec = None
        self._device = None
        self._img_h = 32
        self._img_w = 128

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        from ..data.ocr_dataset import CharCodec
        from .crnn import CRNN

        if not Path(self.weights).exists():
            raise FileNotFoundError(
                f"CRNN weights not found at {self.weights}. Train with src.train_ocr first."
            )
        self._device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.weights, map_location=self._device)
        self._codec = CharCodec(ckpt["alphabet"])
        self._img_h = ckpt.get("img_h", 32)
        self._img_w = ckpt.get("img_w", 128)
        self._model = CRNN(self._codec.num_classes, **ckpt.get("arch", {})).to(self._device)
        self._model.load_state_dict(ckpt["model"])
        self._model.eval()

    def read(self, image: np.ndarray) -> list[OcrResult]:
        import cv2
        import torch

        self._load()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        resized = cv2.resize(gray, (self._img_w, self._img_h)).astype(np.float32) / 255.0
        tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)              # (1, T, C)
            probs = logits.softmax(2)
            conf, idx = probs.max(dim=2)              # greedy
            text = self._codec.decode(idx[0].cpu().tolist())
            # mean confidence over non-blank predicted steps as a rough score
            mask = idx[0] != self._codec.blank
            score = float(conf[0][mask].mean().cpu()) if mask.any() else 0.0

        if not text:
            return []
        return [OcrResult(text=text, confidence=score)]
