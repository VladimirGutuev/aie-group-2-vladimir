from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    image_path: Path
    text: str


def load_samples(data_dir: str | Path, split: str = "test", limit: int | None = None) -> list[Sample]:
    """Load (image, ground-truth text) pairs from a Nomeroff OCR dataset split.

    Expected layout::

        data_dir/<split>/ann/*.json   # {"description": "A123BC777", ...}
        data_dir/<split>/img/*.png

    The dataset archive often unpacks into a single versioned subfolder; if the
    given ``data_dir`` does not directly contain ``<split>``, we look one level
    deeper for the first child that does.
    """
    base = Path(data_dir)
    split_dir = base / split
    if not split_dir.exists():
        for child in sorted(base.iterdir()):
            if child.is_dir() and (child / split).exists():
                split_dir = child / split
                break

    ann_dir = split_dir / "ann"
    img_dir = split_dir / "img"
    if not ann_dir.exists() or not img_dir.exists():
        raise FileNotFoundError(
            f"Could not find ann/ and img/ under {split_dir}. Check --data-dir and --split."
        )

    samples: list[Sample] = []
    for ann_file in sorted(ann_dir.glob("*.json")):
        with open(ann_file, encoding="utf-8") as f:
            meta = json.load(f)
        text = (meta.get("description") or "").strip()
        if not text:
            continue
        img_path = img_dir / (ann_file.stem + ".png")
        if not img_path.exists():
            candidates = list(img_dir.glob(ann_file.stem + ".*"))
            if not candidates:
                continue
            img_path = candidates[0]
        samples.append(Sample(image_path=img_path, text=text))
        if limit is not None and len(samples) >= limit:
            break
    return samples
