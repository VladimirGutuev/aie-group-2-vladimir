"""Evaluate a trained CRNN checkpoint on a Nomeroff OCR split (FSA + CER).

Usage::

    python -m src.eval_crnn --data-dir <dataset> --split test \
        --weights artifacts/crnn.pt --gpu
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from .data.nomeroff import load_samples
from .data.ocr_dataset import CharCodec, OcrDataset, collate
from .evaluate import levenshtein
from .models.crnn import CRNN
from .postprocess import normalize_plate


def greedy_decode(logits: torch.Tensor, codec: CharCodec) -> list[str]:
    idx = logits.argmax(dim=2).cpu().tolist()
    return [codec.decode(seq) for seq in idx]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test", "val"])
    parser.add_argument("--weights", default="artifacts/crnn.pt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out", default="artifacts/crnn_eval.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.weights, map_location=device)
    codec = CharCodec(ckpt["alphabet"])
    model = CRNN(codec.num_classes, **ckpt.get("arch", {})).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    samples = load_samples(args.data_dir, split=args.split, limit=args.limit)
    ds = OcrDataset(samples, codec, ckpt["img_h"], ckpt["img_w"])
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, collate_fn=collate)
    print(f"Loaded {len(samples)} samples from split='{args.split}'")

    exact = 0
    total_cer = 0.0
    n = 0
    errors = []
    with torch.no_grad():
        for images, _targets, _lengths, texts in dl:
            preds = greedy_decode(model(images.to(device)), codec)
            for pred, gt in zip(preds, texts):
                pred = normalize_plate(pred)
                is_exact = pred == gt
                cer = levenshtein(pred, gt) / max(len(gt), 1)
                exact += int(is_exact)
                total_cer += cer
                n += 1
                if not is_exact:
                    errors.append({"gt": gt, "pred": pred, "cer": round(cer, 4)})

    fsa = exact / n if n else 0.0
    cer = total_cer / n if n else 0.0
    print(f"[CRNN] FSA={fsa:.4f}  CER={cer:.4f}  ({n} imgs)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["config", "n", "fsa", "cer"])
        writer.writerow(["crnn", n, round(fsa, 4), round(cer, 4)])
    with open(out_path.with_name("crnn_errors.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["gt", "pred", "cer"])
        writer.writeheader()
        writer.writerows(errors)
    print(f"Summary -> {out_path}; {len(errors)} errors logged")


if __name__ == "__main__":
    main()
