"""Assemble the SentencePiece training corpus: Khmer LM + a small English mix + special tokens.

  python -m src.train.tokenizer.prepare_corpus [--english-mb 50] [--out PATH]

The Khmer corpus is pulled from data/corpus/lm_corpus/lm-*.txt (already khmer_utils.normalize()-
applied during build). English is sampled from wikimedia/wikipedia 20231101.en streaming, capped
by size to avoid dominating the vocab. A header file of all literal special tokens is prepended so
SentencePiece treats them as atomic units worth keeping in vocab.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.corpus.common import DATA, LM_DIR

OUT_DIR = DATA / "tokenizer"
OUT_FILE = OUT_DIR / "train_text.txt"

# Special tokens we want SentencePiece to see (as plain text) so the trainer learns to keep them
# whole. The actual atomic-symbol guarantee comes from --user_defined_symbols at train time.
SPECIAL_LITERALS = [
    "<bos>", "<eos>", "<pad>", "<unk>",
    "<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>",
    "[[", "]]", ",",
    "#", "##", "###", "|", "-", "*", "`",      # markdown markers
]


def write_special_header(fh) -> None:
    for tok in SPECIAL_LITERALS:
        fh.write(tok + "\n")


def write_khmer(fh) -> int:
    n = 0
    for shard in sorted(LM_DIR.glob("lm-*.txt")):
        with open(shard, encoding="utf-8") as src:
            for line in src:
                line = line.strip()
                if not line:
                    continue
                fh.write(line + "\n")
                n += 1
    return n


def write_english(fh, max_bytes: int) -> int:
    """Stream English Wikipedia and write up to ``max_bytes`` of clean text."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[english] `datasets` not installed; skipping English sample")
        return 0
    written = 0
    n = 0
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    for row in ds:
        text = (row.get("text") or "").strip()
        if len(text) < 200:
            continue
        # one paragraph per line to match Khmer shards
        for para in text.split("\n"):
            para = para.strip()
            if len(para) < 50:
                continue
            line_bytes = len(para.encode("utf-8")) + 1
            if written + line_bytes > max_bytes:
                return n
            fh.write(para + "\n")
            written += line_bytes
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--english-mb", type=int, default=50,
                    help="cap English sample size in MB (0 to skip)")
    ap.add_argument("--out", type=Path, default=OUT_FILE)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        write_special_header(fh)
        khm = write_khmer(fh)
        eng = write_english(fh, args.english_mb * 1024 * 1024) if args.english_mb > 0 else 0

    size_mb = args.out.stat().st_size / 1024 / 1024
    print(f"[corpus] Khmer lines: {khm:,}")
    print(f"[corpus] English lines: {eng:,}")
    print(f"[corpus] -> {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
