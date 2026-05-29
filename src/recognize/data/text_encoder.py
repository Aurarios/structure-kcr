"""Raw-text encoder for the line recognizer — wraps the existing SentencePiece model.

Reuses data/tokenizer/khmer_ocr.model (22k Unigram, full Khmer+ASCII, byte_fallback). Unlike the
AR LabelEncoder it produces plain line-text token sequences (no <|ref|>/<|det|>/coord scaffolding).

- Attention decoder targets: [bos] + ids + [eos]
- CTC targets: ids only (blank == pad id == 0)

khmer_utils.normalize is applied to every text (idempotent, correctness-critical: must match the
normalized space the tokenizer was trained on and that page labels use).
"""
from __future__ import annotations

from pathlib import Path

import sentencepiece as spm

from src import khmer_utils as ku


class TextEncoder:
    def __init__(self, sp_model_path: Path | str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_model_path))
        self.bos = self.sp.piece_to_id("<bos>")
        self.eos = self.sp.piece_to_id("<eos>")
        self.pad = self.sp.piece_to_id("<pad>")
        self.blank = self.pad  # CTC blank reuses pad id (pad never appears as a real target)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode_attn(self, text: str) -> list[int]:
        ids = self.sp.encode(ku.normalize(text), out_type=int)
        return [self.bos, *ids, self.eos]

    def encode_ctc(self, text: str) -> list[int]:
        return self.sp.encode(ku.normalize(text), out_type=int)

    def decode(self, ids: list[int]) -> str:
        out = [i for i in ids if i not in (self.bos, self.eos, self.pad)]
        return self.sp.decode(out)

    def ctc_collapse(self, ids: list[int]) -> str:
        """Greedy CTC decode: collapse repeats, drop blanks, then SentencePiece-decode."""
        prev = -1
        kept: list[int] = []
        for i in ids:
            if i != prev and i != self.blank:
                kept.append(i)
            prev = i
        return self.sp.decode([i for i in kept if i not in (self.bos, self.eos, self.pad)])
