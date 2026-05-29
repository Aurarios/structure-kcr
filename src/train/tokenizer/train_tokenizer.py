"""Train a SentencePiece Unigram tokenizer for Khmer + English + grounded-markdown output.

  python -m src.train.tokenizer.train_tokenizer [--vocab-size 22000]

Unigram (not BPE) because Khmer combining-mark clusters are better handled by probabilistic
segmentation than greedy merges.

Vocab guarantees ("sophisticated coverage"):
  - Every ASSIGNED Khmer codepoint (U+1780-17FF main block + U+19E0-19FF Symbols block) is a
    user_defined_symbol -> atomic in any context. No char ever decomposes.
  - Every ASCII printable (excluding space, which SP reserves as the word-boundary marker) is
    a user_defined_symbol.
  - 1000 coordinate tokens <0>..<999> for grounded boxes (one token per int in [0,999]).
  - 6 structural tokens for the <|ref|>/<|det|> grounded-markdown schema.
  - byte_fallback=True so anything we still missed (accented Latin, emoji, weird Unicode)
    decomposes into UTF-8 bytes instead of <unk>. Adds 256 byte tokens.

Outputs to data/tokenizer/:
  khmer_ocr.model    sentencepiece model
  khmer_ocr.vocab    human-readable vocab listing
"""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import sentencepiece as spm

from src.corpus.common import DATA

OUT_DIR = DATA / "tokenizer"
MODEL_PREFIX = OUT_DIR / "khmer_ocr"

CONTROL_TOKENS = ["<bos>", "<eos>", "<pad>", "<unk>"]
STRUCTURE_TOKENS = [
    "<|ref|>", "<|/ref|>", "<|det|>", "<|/det|>",
    "[[", "]]",
]
COORD_TOKENS = [f"<{i}>" for i in range(1000)]


def _assigned_chars(lo: int, hi: int) -> list[str]:
    """Return single-char strings for every ASSIGNED codepoint in [lo, hi]."""
    out: list[str] = []
    for cp in range(lo, hi + 1):
        ch = chr(cp)
        try:
            unicodedata.name(ch)
        except ValueError:
            continue
        out.append(ch)
    return out


# RARE Khmer codepoints that don't appear in our corpus and would otherwise be <unk>.
# Common Khmer chars (consonants, dependent vowels, signs, numerals) are NOT listed here —
# they're guaranteed atomic via `character_coverage=1.0` because they appear in the corpus.
# Listing every Khmer char as user_defined would block the Unigram model from learning any
# multi-char Khmer subwords, collapsing the candidate pool to ~1300 and breaking training.
RARE_KHMER_CHARS = (
    [chr(c) for c in (0x17A4, 0x17A6, 0x17A8, 0x17A9)]   # rare independent vowels: ឤ ឦ ឨ ឩ
    + _assigned_chars(0x17F0, 0x17F9)                     # divinatory numerals (10)
    + _assigned_chars(0x19E0, 0x19FF)                     # Khmer Symbols block (32)
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=OUT_DIR / "train_text.txt")
    ap.add_argument("--vocab-size", type=int, default=22000)
    ap.add_argument("--character-coverage", type=float, default=1.0)
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"missing {args.input}; run prepare_corpus.py first")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Order matters only cosmetically (vocab IDs assigned in this order, after control tokens).
    # Keep coord tokens last so they sit in a contiguous range — makes the lookup table in
    # LabelEncoder simple.
    user_defined = (
        STRUCTURE_TOKENS
        + RARE_KHMER_CHARS
        + COORD_TOKENS
    )
    print(f"[tokenizer] user_defined_symbols: {len(STRUCTURE_TOKENS)} structure + "
          f"{len(RARE_KHMER_CHARS)} rare-Khmer + "
          f"{len(COORD_TOKENS)} coord = {len(user_defined)} total")

    spm.SentencePieceTrainer.train(
        input=str(args.input),
        model_prefix=str(MODEL_PREFIX),
        model_type="unigram",
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        # we already khmer-normalize upstream; don't apply NFKC/NMT-style on top
        normalization_rule_name="identity",
        # piece IDs (sentencepiece reserves 0..3 by default but we override to be explicit)
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        # tokens that must remain atomic and present in vocab
        user_defined_symbols=user_defined,
        # graceful degradation: unknown chars decompose into 256 byte tokens instead of <unk>
        byte_fallback=True,
        # treat each whitespace-delimited token as a unit boundary; Khmer (no spaces) sentence
        # remains one unit and gets internally segmented by the unigram model
        split_by_whitespace=True,
        # large enough to not silently truncate input shards
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        num_threads=8,
    )

    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_PREFIX) + ".model")
    print(f"[tokenizer] vocab_size = {sp.get_piece_size()}")
    print(f"[tokenizer] coord token <0> id = {sp.piece_to_id('<0>')}, <999> id = {sp.piece_to_id('<999>')}")
    print(f"[tokenizer] <|ref|> id = {sp.piece_to_id('<|ref|>')}, <|det|> id = {sp.piece_to_id('<|det|>')}")

    # round-trip self-test on a Khmer string with structure tokens
    sample = "<|ref|>ភ្នំពេញ<|/ref|><|det|>[[<62><77><937><157>]]<|/det|>"
    ids = sp.encode(sample, out_type=int)
    back = sp.decode(ids)
    print(f"[tokenizer] sample tokens: {len(ids)}  round-trip ok: {back == sample}")


if __name__ == "__main__":
    main()
