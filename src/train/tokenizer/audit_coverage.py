"""Audit the trained tokenizer's coverage of Khmer + ASCII codepoints.

For each codepoint in scope, encodes it standalone and reports:
- single-token (atomic in vocab) — IDEAL
- multi-token (built from smaller pieces) — OK, but slower
- maps to <unk> — BAD, model can't produce this character

Usage:
  python -m src.train.tokenizer.audit_coverage
"""
from __future__ import annotations

import sys

import sentencepiece as spm

# Force UTF-8 on the Windows console so Khmer codepoints print without cp949 errors.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from src.corpus.common import DATA

MODEL = DATA / "tokenizer" / "khmer_ocr.model"

RANGES = [
    ("Khmer consonants",      0x1780, 0x17A2),
    ("Khmer independent vowels", 0x17A3, 0x17B5),
    ("Khmer dependent vowels",   0x17B6, 0x17C5),
    ("Khmer signs (+ coeng)",    0x17C6, 0x17DD),
    ("Khmer numerals",        0x17E0, 0x17E9),
    ("Khmer divinatory",      0x17F0, 0x17F9),
    ("Khmer Symbols block",   0x19E0, 0x19FF),
    ("ASCII printable",       0x0020, 0x007E),
]


def main() -> None:
    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL))
    unk_id = sp.piece_to_id("<unk>")

    grand_total = {"single": 0, "multi": 0, "unk": 0, "n": 0}
    print(f"Vocab size: {sp.get_piece_size()}\n")

    for label, lo, hi in RANGES:
        per = {"single": 0, "multi": 0, "unk": 0}
        missing: list[str] = []
        multi_examples: list[str] = []
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            ids = sp.encode(ch, out_type=int)
            if not ids or all(i == unk_id for i in ids):
                per["unk"] += 1
                missing.append(f"U+{cp:04X} {ch}")
            elif len(ids) == 1 and ids[0] != unk_id:
                per["single"] += 1
            else:
                per["multi"] += 1
                if len(multi_examples) < 5:
                    pieces = [sp.id_to_piece(i) for i in ids]
                    multi_examples.append(f"U+{cp:04X} {ch} -> {pieces}")
        total = sum(per.values())
        print(f"{label:34s}  single={per['single']:>3} / multi={per['multi']:>3} / unk={per['unk']:>3}  (of {total})")
        if missing:
            print(f"  MISSING: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
        if multi_examples:
            for ex in multi_examples:
                print(f"  multi  : {ex}")
        for k in per:
            grand_total[k] += per[k]
        grand_total["n"] += total

    print()
    print(f"TOTAL: {grand_total['single']} atomic, {grand_total['multi']} multi-piece, "
          f"{grand_total['unk']} unk (of {grand_total['n']} codepoints scanned)")


if __name__ == "__main__":
    main()
