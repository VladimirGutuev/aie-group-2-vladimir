from __future__ import annotations

# Russian plate letters are visually identical to a subset of Latin letters.
# Nomeroff `description` stores them as Latin; EasyOCR may return Cyrillic.
# Map the look-alikes to Latin so comparison is fair regardless of OCR output.
_CYR_TO_LAT = {
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H",
    "О": "O", "Р": "P", "С": "C", "Т": "T", "У": "Y", "Х": "X",
}


def normalize_plate(text: str) -> str:
    """Uppercase, transliterate Cyrillic look-alikes, keep only alphanumerics."""
    out: list[str] = []
    for ch in text.upper():
        ch = _CYR_TO_LAT.get(ch, ch)
        if ch.isalnum():
            out.append(ch)
    return "".join(out)
