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


# Confusion fixes for the Russian plate format `L DDD LL DD(D)`
# (L = letter at index 0,4,5; D = digit at index 1,2,3,6,7,8).
# On digit positions a letter that looks like a digit is replaced, and vice versa.
_LETTER_POS = frozenset({0, 4, 5})
_DIGIT_LIKE = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1",
               "Z": "7", "S": "5", "B": "8", "G": "6", "T": "7", "A": "4"}
_LETTER_LIKE = {"0": "O", "8": "B", "4": "A"}  # only map to valid RU letters


def correct_ru_plate(text: str) -> str:
    """Fix digit/letter confusions using the Russian plate positional mask.

    Applied only to 8- or 9-character strings (standard car plate length);
    other lengths are returned unchanged.
    """
    if len(text) not in (8, 9):
        return text
    out: list[str] = []
    for i, ch in enumerate(text):
        if i in _LETTER_POS:
            out.append(_LETTER_LIKE.get(ch, ch) if ch.isdigit() else ch)
        else:
            out.append(_DIGIT_LIKE.get(ch, ch) if ch.isalpha() else ch)
    return "".join(out)
