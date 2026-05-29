"""Evaluate OCR quality on a Nomeroff OCR dataset split.

Compares a baseline reader (raw crop, default EasyOCR) against an improved
configuration (grayscale + CLAHE preprocessing + Russian-plate allowlist) using
two metrics:

- Full Sequence Accuracy (FSA): share of plates predicted exactly right.
- Character Error Rate (CER): mean Levenshtein distance normalized by GT length.

Usage::

    python -m src.evaluate --data-dir <path-to-ocr-dataset> --split test \
        --limit 500 --gpu --out artifacts/eval_results.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# EasyOCR prints a download progress bar using block glyphs; on Windows a
# redirected stdout defaults to cp1251 and chokes on them. Force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from .data.nomeroff import Sample, load_samples
from .models.ocr import RU_ALLOWLIST, PlateOCR
from .postprocess import normalize_plate


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def best_text(reader: PlateOCR, image: np.ndarray) -> str:
    results = reader.read(image)
    if not results:
        return ""
    best = max(results, key=lambda r: r.confidence)
    return normalize_plate(best.text)


def evaluate(reader: PlateOCR, samples: list[Sample]) -> dict:
    exact = 0
    total_cer = 0.0
    rows = []
    for s in samples:
        gt = normalize_plate(s.text)
        pred = best_text(reader, load_image(s.image_path))
        is_exact = pred == gt
        cer = levenshtein(pred, gt) / max(len(gt), 1)
        exact += int(is_exact)
        total_cer += cer
        rows.append({"image": s.image_path.name, "gt": gt, "pred": pred,
                     "exact": int(is_exact), "cer": round(cer, 4)})
    n = len(samples)
    return {
        "n": n,
        "fsa": exact / n if n else 0.0,
        "cer": total_cer / n if n else 0.0,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Path to Nomeroff OCR dataset root")
    parser.add_argument("--split", default="test", choices=["train", "test", "val"])
    parser.add_argument("--limit", type=int, default=500, help="Max samples (None for all)")
    parser.add_argument("--langs", default="en", help="EasyOCR languages, comma-separated")
    parser.add_argument("--gpu", action="store_true", help="Run EasyOCR on GPU")
    parser.add_argument("--out", default="artifacts/eval_results.csv", help="CSV output path")
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    samples = load_samples(args.data_dir, split=args.split, limit=args.limit)
    print(f"Loaded {len(samples)} samples from split='{args.split}'")

    configs = {
        "baseline": PlateOCR(langs=langs, gpu=args.gpu),
        "improved": PlateOCR(langs=langs, gpu=args.gpu, preprocess=True, allowlist=RU_ALLOWLIST),
    }

    summary = []
    detailed: dict[str, list[dict]] = {}
    for name, reader in configs.items():
        t0 = time.time()
        res = evaluate(reader, samples)
        dt = time.time() - t0
        per_img = dt / max(res["n"], 1)
        print(f"[{name:8s}] FSA={res['fsa']:.3f}  CER={res['cer']:.3f}  "
              f"({res['n']} imgs, {per_img*1000:.0f} ms/img)")
        summary.append({"config": name, "n": res["n"],
                        "fsa": round(res["fsa"], 4), "cer": round(res["cer"], 4),
                        "ms_per_img": round(per_img * 1000, 1)})
        detailed[name] = res["rows"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "n", "fsa", "cer", "ms_per_img"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSummary written to {out_path}")

    err_path = out_path.with_name("eval_errors.csv")
    with open(err_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "image", "gt", "pred", "exact", "cer"])
        writer.writeheader()
        for name, rows in detailed.items():
            for r in rows:
                if not r["exact"]:
                    writer.writerow({"config": name, **r})
    print(f"Per-image errors written to {err_path}")


if __name__ == "__main__":
    main()
