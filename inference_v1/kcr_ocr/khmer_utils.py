"""Shared Khmer text utilities: Unicode normalization, script detection, segmentation.

The normalizer is the correctness-critical piece of the whole pipeline. Khmer encodes an
orthographic syllable as a base character followed by combining marks (register shifters,
subscript consonants introduced by COENG U+17D2, dependent vowels, and signs). The *same*
visible syllable can be typed/stored in different orders; if we don't canonicalize, OCR labels
disagree with each other and the model learns noise.

`normalize()` is intended to be **idempotent** (normalize(normalize(x)) == normalize(x)) and is
applied identically to rendered text and to labels, so the round-trip check holds by construction.

This is a best-effort reordering based on the documented Khmer canonical storage order. For
production-grade correctness consider ICU or a vetted Khmer library; `roundtrip_check.py` enforces
idempotency to catch regressions.
"""
from __future__ import annotations

import re
import unicodedata

# --- Khmer Unicode ranges ---------------------------------------------------
KHMER_MAIN = (0x1780, 0x17FF)        # main Khmer block
KHMER_SYMBOLS = (0x19E0, 0x19FF)     # Khmer Symbols block

COENG = 0x17D2                       # subscript combiner: COENG + consonant => subscript
ROBAT = 0x17CC
ZWSP = "​"                      # zero-width space (Khmer "word" separator, often noise)
ZWNJ = "‌"
ZWJ = "‍"

CONSONANTS = range(0x1780, 0x17A3)           # ក..អ
INDEP_VOWELS = range(0x17A5, 0x17B4)         # independent vowels
DEP_VOWELS = range(0x17B6, 0x17C6)           # dependent vowels (pre/above/below/post)
SHIFTERS = {0x17C9, 0x17CA}                  # muusikatoan / triisap (register shifters)
ABOVE_SIGNS = {0x17C6, 0x17C7, 0x17C8}       # nikahit / reahmuk / yuukaleapintu
OTHER_SIGNS = {0x17CB, 0x17CD, 0x17CE, 0x17CF, 0x17D0, 0x17D1, 0x17D3, 0x17DD}

# sentence terminators: khan (។), bariyoosan (៕), plus ASCII fallbacks
SENT_TERMINATORS = "។៕!?"

# ranks define canonical order of combining marks within a cluster (lower = earlier)
_RANK_ROBAT = 1
_RANK_COENG = 2     # coeng+consonant units (kept in original relative order)
_RANK_SHIFTER = 3
_RANK_VOWEL = 4
_RANK_ABOVE = 5
_RANK_OTHER = 6


def _cp(ch: str) -> int:
    return ord(ch)


def is_khmer_char(ch: str) -> bool:
    c = ord(ch)
    return KHMER_MAIN[0] <= c <= KHMER_MAIN[1] or KHMER_SYMBOLS[0] <= c <= KHMER_SYMBOLS[1]


def is_khmer_base(ch: str) -> bool:
    """A character that can start a cluster (consonant or independent vowel)."""
    c = ord(ch)
    return c in CONSONANTS or c in INDEP_VOWELS or c in (0x17A3, 0x17A4)


def _is_combining(ch: str) -> bool:
    c = ord(ch)
    return (
        c == ROBAT
        or c == COENG
        or c in DEP_VOWELS
        or c in SHIFTERS
        or c in ABOVE_SIGNS
        or c in OTHER_SIGNS
    )


def _mark_rank(ch: str) -> int:
    c = ord(ch)
    if c == ROBAT:
        return _RANK_ROBAT
    if c in SHIFTERS:
        return _RANK_SHIFTER
    if c in DEP_VOWELS:
        return _RANK_VOWEL
    if c in ABOVE_SIGNS:
        return _RANK_ABOVE
    return _RANK_OTHER


def _reorder_cluster(base: str, marks: list[str]) -> str:
    """Reorder the combining marks of one cluster into canonical storage order.

    Coeng+consonant pairs are treated as indivisible units and keep their original relative
    order (reordering subscripts changes meaning). Other marks are stable-sorted by rank.
    """
    units: list[tuple[int, int, str]] = []   # (rank, original_index, text)
    i = 0
    order = 0
    while i < len(marks):
        ch = marks[i]
        if ord(ch) == COENG and i + 1 < len(marks):
            # glue COENG + following consonant (or independent) into one unit
            unit = ch + marks[i + 1]
            units.append((_RANK_COENG, order, unit))
            i += 2
        else:
            units.append((_mark_rank(ch), order, ch))
            i += 1
        order += 1
    units.sort(key=lambda u: (u[0], u[1]))   # stable on original order within a rank
    return base + "".join(u[2] for u in units)


def normalize(text: str) -> str:
    """Canonicalize Khmer text. Idempotent. Applied to both rendered text and labels."""
    if not text:
        return text
    # 1) NFC and strip zero-width controls that are usually noise in scraped text
    text = unicodedata.normalize("NFC", text)
    text = text.replace(ZWSP, "").replace(ZWNJ, "").replace(ZWJ, "")
    # 2) normalize whitespace (keep single spaces and newlines)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 3) reorder combining marks cluster by cluster
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if is_khmer_base(ch):
            j = i + 1
            marks: list[str] = []
            while j < n and (_is_combining(text[j]) or ord(text[j]) == COENG):
                marks.append(text[j])
                j += 1
                # COENG consumes the following consonant as part of the cluster
                if marks and ord(marks[-1]) == COENG and j < n:
                    marks.append(text[j])
                    j += 1
            out.append(_reorder_cluster(ch, marks) if marks else ch)
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out).strip()


def khmer_ratio(text: str) -> float:
    """Fraction of *letter-like* characters that are Khmer. Whitespace/digits/punct ignored."""
    letters = [c for c in text if c.isalpha() or is_khmer_char(c)]
    if not letters:
        return 0.0
    khmer = sum(1 for c in letters if is_khmer_char(c))
    return khmer / len(letters)


def contains_khmer(text: str) -> bool:
    return any(is_khmer_char(c) for c in text)


def split_sentences(text: str) -> list[str]:
    """Split into sentence-ish units on Khmer terminators and newlines. Khmer has no
    inter-word spaces, so we never split on spaces."""
    parts = re.split(rf"(?<=[{re.escape(SENT_TERMINATORS)}])|\n+", text)
    return [p.strip() for p in parts if p and p.strip()]


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


if __name__ == "__main__":
    # tiny self-test / demo
    samples = [
        "ភាសាខ្មែរ",            # basic
        "កម្ពុជា",              # coeng
        "ប្រទេស​កម្ពុជា",      # contains ZWSP
    ]
    for s in samples:
        n = normalize(s)
        assert normalize(n) == n, f"not idempotent: {s!r}"
        print(f"{s!r:30} -> {n!r:30} khmer_ratio={khmer_ratio(n):.2f}")
    print("idempotency OK")
