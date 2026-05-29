"""Evaluate the pretrained Nomeroff Net OCR on a Nomeroff OCR split (reference).

Run inside the dedicated `nomeroff` conda env (it pins its own torch/deps).

Usage::

    python -m src.eval_nomeroff --data-dir <dataset> --split test --limit 500
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from .data.nomeroff import load_samples
from .evaluate import levenshtein
from .postprocess import normalize_plate


def build_detector():
    # nomeroff-net 4.0.1 predates PyTorch 2.6's weights_only=True default and
    # ships checkpoints with custom globals. We trust the official weights, so
    # force the legacy load behaviour before importing their code.
    import torch

    _orig_load = torch.load

    def _patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load

    from nomeroff_net.pipes.number_plate_text_readers.text_detector import TextDetector

    # In nomeroff-net 4.0.1 the high-level TextDetector.predict hands raw zone
    # arrays to the low-level OCR.predict, which expects an already-preprocessed
    # tensor (its own preprocess() is never called). Patch it to preprocess when
    # given a non-tensor input.
    from nomeroff_net.pipes.number_plate_text_readers.base.ocr import OCR

    _orig_predict = OCR.predict

    def _patched_predict(self, xs, return_acc=False):
        if not isinstance(xs, torch.Tensor):
            xs = self.preprocess(xs)
        return _orig_predict(self, xs, return_acc)

    OCR.predict = _patched_predict

    return TextDetector({"ru": {"for_regions": ["ru"], "model_path": "latest"}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test", "val"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", default="artifacts/nomeroff_eval.csv")
    args = parser.parse_args()

    samples = load_samples(args.data_dir, split=args.split, limit=args.limit)
    print(f"Loaded {len(samples)} samples from split='{args.split}'")

    detector = build_detector()

    exact = 0
    total_cer = 0.0
    n = 0
    errors = []
    for start in range(0, len(samples), args.batch_size):
        batch = samples[start:start + args.batch_size]
        imgs = [cv2.cvtColor(cv2.imread(str(s.image_path)), cv2.COLOR_BGR2RGB) for s in batch]
        # tell the reader these are single-line Russian plates
        preds = detector.predict(imgs, ["ru"] * len(imgs), [1] * len(imgs))
        for s, pred in zip(batch, preds):
            gt = normalize_plate(s.text)
            pred = normalize_plate(pred if isinstance(pred, str) else str(pred))
            is_exact = pred == gt
            cer = levenshtein(pred, gt) / max(len(gt), 1)
            exact += int(is_exact)
            total_cer += cer
            n += 1
            if not is_exact:
                errors.append({"gt": gt, "pred": pred, "cer": round(cer, 4)})
        print(f"  processed {min(start + args.batch_size, len(samples))}/{len(samples)}")

    fsa = exact / n if n else 0.0
    cer = total_cer / n if n else 0.0
    print(f"[Nomeroff] FSA={fsa:.4f}  CER={cer:.4f}  ({n} imgs)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n", "fsa", "cer"])
        writer.writerow(["nomeroff", n, round(fsa, 4), round(cer, 4)])
    print(f"Summary -> {out_path}; {len(errors)} errors")


if __name__ == "__main__":
    main()
