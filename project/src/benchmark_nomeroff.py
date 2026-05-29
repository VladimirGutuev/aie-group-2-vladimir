"""Benchmark the Nomeroff OCR model efficiency (params, latency, throughput).

Run inside the `nomeroff` conda env. Mirrors src.benchmark for a fair compare.

Usage::

    python -m src.benchmark_nomeroff --n 1000 --batch-size 128
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from .eval_nomeroff import build_detector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    import torch

    detector = build_detector()
    ocr = detector.detectors[0]
    model = ocr.model
    device = next(model.parameters()).device
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print("Model: Nomeroff OCR")
    print(f"  parameters : {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  device     : {device}")

    # build dummy plate crops (RGB uint8) and preprocess via their own pipeline
    h, w = ocr.height, ocr.width
    crop = (np.random.rand(h, w, 3) * 255).astype(np.uint8)

    one = ocr.preprocess([crop])
    with torch.no_grad():
        for _ in range(3):
            model(one)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.n):
            model(one)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / args.n * 1000

    batch = ocr.preprocess([crop] * args.batch_size)
    iters = max(1, args.n // args.batch_size)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(iters):
            model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    throughput = iters * args.batch_size / (time.time() - t0)

    print(f"  [{device.type.upper()}] latency={latency_ms:.2f} ms/img  "
          f"throughput={throughput:.0f} img/s")


if __name__ == "__main__":
    main()
