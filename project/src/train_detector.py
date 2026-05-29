"""Train a YOLOv8 license-plate detector on a YOLO-format dataset.

Usage::

    python -m src.train_detector --data <plates_yolo>/data.yaml --epochs 40 --device 0
"""
from __future__ import annotations

import argparse
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Base weights")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    parser.add_argument("--project", default="artifacts/detector")
    parser.add_argument("--name", default="yolov8n_plate")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )
    metrics = model.val(data=args.data, split="test", device=args.device)
    print(f"TEST  mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    print(f"Best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
