from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ..postprocess import normalize_plate
from .nomeroff import Sample


class CharCodec:
    """Maps plate characters <-> integer indices for CTC (blank = last index)."""

    def __init__(self, alphabet: str) -> None:
        self.alphabet = alphabet
        self.char_to_idx = {c: i for i, c in enumerate(alphabet)}
        self.blank = len(alphabet)
        self.num_classes = len(alphabet) + 1

    def encode(self, text: str) -> list[int]:
        return [self.char_to_idx[c] for c in text if c in self.char_to_idx]

    def decode(self, indices: list[int]) -> str:
        # collapse repeats, drop blanks (greedy CTC decode)
        out: list[str] = []
        prev = -1
        for idx in indices:
            if idx != prev and idx != self.blank:
                out.append(self.alphabet[idx])
            prev = idx
        return "".join(out)


def build_alphabet(samples: list[Sample]) -> str:
    chars = set()
    for s in samples:
        chars.update(normalize_plate(s.text))
    return "".join(sorted(chars))


class OcrDataset(Dataset):
    def __init__(self, samples: list[Sample], codec: CharCodec, img_h: int = 32, img_w: int = 128) -> None:
        self.samples = samples
        self.codec = codec
        self.img_h = img_h
        self.img_w = img_w

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = Image.open(s.image_path).convert("L").resize((self.img_w, self.img_h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
        label = torch.tensor(self.codec.encode(normalize_plate(s.text)), dtype=torch.long)
        return tensor, label, normalize_plate(s.text)


def collate(batch):
    images, labels, texts = zip(*batch)
    images = torch.stack(images, 0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    targets = torch.cat(labels) if labels else torch.tensor([], dtype=torch.long)
    return images, targets, label_lengths, list(texts)
