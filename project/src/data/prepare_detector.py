"""Download a HuggingFace license-plate detection dataset and convert to YOLO format.

Produces::

    <out>/images/{train,val,test}/*.jpg
    <out>/labels/{train,val,test}/*.txt   # class cx cy w h  (normalized)
    <out>/data.yaml

Usage::

    python -m src.data.prepare_detector --out C:/.../datasets/plates_yolo
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

HF_SPLIT_TO_YOLO = {"train": "train", "validation": "val", "test": "test"}


def convert(out_dir: Path, dataset_name: str, config: str) -> None:
    from datasets import load_dataset

    for hf_split, yolo_split in HF_SPLIT_TO_YOLO.items():
        try:
            ds = load_dataset(dataset_name, config, split=hf_split)
        except Exception as exc:  # noqa: BLE001 - report and skip missing splits
            print(f"  split '{hf_split}' skipped: {type(exc).__name__}: {exc}")
            continue

        img_dir = out_dir / "images" / yolo_split
        lbl_dir = out_dir / "labels" / yolo_split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        n = 0
        for i, ex in enumerate(ds):
            img = ex["image"].convert("RGB")
            w, h = img.size
            objects = ex["objects"]
            bboxes = objects["bbox"]  # COCO [x, y, w, h]
            if not bboxes:
                continue
            stem = f"{i:06d}"
            img.save(img_dir / f"{stem}.jpg", quality=92)
            lines = []
            for x, y, bw, bh in bboxes:
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            n += 1
        print(f"  {yolo_split}: {n} images")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "names:\n  0: plate\n",
        encoding="utf-8",
    )
    print(f"data.yaml -> {data_yaml}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output YOLO dataset dir")
    parser.add_argument("--dataset", default="keremberke/license-plate-object-detection")
    parser.add_argument("--config", default="full")
    args = parser.parse_args()
    convert(Path(args.out), args.dataset, args.config)


if __name__ == "__main__":
    main()
