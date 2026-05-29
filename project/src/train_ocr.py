"""Train a CRNN+CTC plate OCR model on a Nomeroff OCR dataset.

Usage::

    python -m src.train_ocr --data-dir <dataset> --epochs 15 --batch-size 256 \
        --gpu --out artifacts/crnn.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from .data.nomeroff import load_samples
from .data.ocr_dataset import CharCodec, OcrDataset, build_alphabet, collate
from .models.crnn import CRNN
from .postprocess import normalize_plate


def greedy_decode(logits: torch.Tensor, codec: CharCodec) -> list[str]:
    # logits: (B, T, C)
    idx = logits.argmax(dim=2).cpu().tolist()
    return [codec.decode(seq) for seq in idx]


@torch.no_grad()
def evaluate_fsa(model, loader, codec, device) -> float:
    model.eval()
    correct = total = 0
    for images, _targets, _lengths, texts in loader:
        logits = model(images.to(device))
        preds = greedy_decode(logits, codec)
        for pred, gt in zip(preds, texts):
            correct += int(pred == gt)
            total += 1
    return correct / total if total else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-h", type=int, default=32)
    parser.add_argument("--img-w", type=int, default=128)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--last-channels", type=int, default=512)
    parser.add_argument("--rnn-hidden", type=int, default=256)
    parser.add_argument("--rnn-layers", type=int, default=2)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--out", default="artifacts/crnn.pt")
    args = parser.parse_args()

    arch = {"last_channels": args.last_channels, "rnn_hidden": args.rnn_hidden,
            "rnn_layers": args.rnn_layers}

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_s = load_samples(args.data_dir, split="train", limit=args.limit_train)
    val_s = load_samples(args.data_dir, split="val", limit=2000)
    print(f"train={len(train_s)}  val={len(val_s)}")

    alphabet = build_alphabet(train_s)
    codec = CharCodec(alphabet)
    print(f"alphabet ({len(alphabet)}): {alphabet}")

    train_ds = OcrDataset(train_s, codec, args.img_h, args.img_w)
    val_ds = OcrDataset(val_s, codec, args.img_h, args.img_w)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, collate_fn=collate, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, collate_fn=collate)

    model = CRNN(codec.num_classes, **arch).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M  arch={arch}")
    criterion = nn.CTCLoss(blank=codec.blank, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_fsa = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        for images, targets, target_lengths, _texts in train_dl:
            images = images.to(device)
            logits = model(images)                       # (B, T, C)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, B, C)
            input_lengths = torch.full((images.size(0),), logits.size(1),
                                       dtype=torch.long)
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running += loss.item()
        scheduler.step()

        fsa = evaluate_fsa(model, val_dl, codec, device)
        dt = time.time() - t0
        print(f"epoch {epoch:2d}  loss={running/len(train_dl):.4f}  "
              f"val_FSA={fsa:.4f}  ({dt:.0f}s)")

        if fsa > best_fsa:
            best_fsa = fsa
            torch.save({"model": model.state_dict(), "alphabet": alphabet,
                        "img_h": args.img_h, "img_w": args.img_w, "arch": arch}, out_path)
            with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump({"alphabet": alphabet, "img_h": args.img_h,
                           "img_w": args.img_w, "arch": arch, "best_val_fsa": best_fsa}, f,
                          ensure_ascii=False, indent=2)
            print(f"  saved best (val_FSA={best_fsa:.4f}) -> {out_path}")

    print(f"\nBest val FSA: {best_fsa:.4f}")


if __name__ == "__main__":
    main()
