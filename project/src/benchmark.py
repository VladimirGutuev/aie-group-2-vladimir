"""Benchmark CRNN OCR efficiency: parameter count, weight size, latency, throughput.

Usage::

    python -m src.benchmark --weights artifacts/crnn.pt --gpu --n 1000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from .data.ocr_dataset import CharCodec
from .models.crnn import CRNN


def measure(model, device, img_h, img_w, n: int, batch_size: int) -> dict:
    model.eval()
    # warmup
    dummy = torch.rand(batch_size, 1, img_h, img_w, device=device)
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # single-image latency
    one = torch.rand(1, 1, img_h, img_w, device=device)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n):
            model(one)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / n * 1000

    # batched throughput
    t0 = time.time()
    iters = max(1, n // batch_size)
    with torch.no_grad():
        for _ in range(iters):
            model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    throughput = iters * batch_size / (time.time() - t0)

    return {"latency_ms": latency_ms, "throughput": throughput}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="artifacts/crnn.pt")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location="cpu")
    codec = CharCodec(ckpt["alphabet"])
    img_h, img_w = ckpt.get("img_h", 32), ckpt.get("img_w", 128)
    model = CRNN(codec.num_classes)
    model.load_state_dict(ckpt["model"])

    n_params = sum(p.numel() for p in model.parameters())
    size_mb = Path(args.weights).stat().st_size / 1e6
    print(f"Model: CRNN")
    print(f"  parameters : {n_params:,} ({n_params/1e6:.2f}M)")
    print(f"  weight size: {size_mb:.1f} MB")

    for use_gpu in ([False, True] if args.gpu else [False]):
        dev = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        if use_gpu and dev.type != "cuda":
            continue
        model.to(dev)
        res = measure(model, dev, img_h, img_w, args.n, args.batch_size)
        print(f"  [{dev.type.upper()}] latency={res['latency_ms']:.2f} ms/img  "
              f"throughput={res['throughput']:.0f} img/s")


if __name__ == "__main__":
    main()
