"""Turn a synthetic label JSON into a token-id sequence for training.

Per-block emission (in DOM order, which is reading order):
    <|ref|> {block_text} <|/ref|> <|det|>[[<x1><y1><x2><y2>]]<|/det|>
Whole-sequence wrap:
    <bos> ... <eos>

Coordinates come from `bbox_norm` (already clipped to [0,999] in src/build_dataset.py:
_normalize_leaves). Each coordinate is one atomic token (<0>..<999>), enabled by the user_defined
symbols in the SentencePiece trainer.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import sentencepiece as spm

REF_O, REF_C = "<|ref|>", "<|/ref|>"
DET_O, DET_C = "<|det|>", "<|/det|>"
LBR, RBR = "[[", "]]"


class LabelEncoder:
    def __init__(self, sp_model_path: Path | str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(sp_model_path))
        self.bos = self.sp.piece_to_id("<bos>")
        self.eos = self.sp.piece_to_id("<eos>")
        self.pad = self.sp.piece_to_id("<pad>")
        self._ref_o = self.sp.piece_to_id(REF_O)
        self._ref_c = self.sp.piece_to_id(REF_C)
        self._det_o = self.sp.piece_to_id(DET_O)
        self._det_c = self.sp.piece_to_id(DET_C)
        self._lbr = self.sp.piece_to_id(LBR)
        self._rbr = self.sp.piece_to_id(RBR)
        # cache coord-token ids for fast lookup
        self._coord = [self.sp.piece_to_id(f"<{i}>") for i in range(1000)]

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()

    def encode_block(self, text: str, bbox_norm: list[int]) -> list[int]:
        text_ids = self.sp.encode(text, out_type=int)
        x1, y1, x2, y2 = (max(0, min(999, int(c))) for c in bbox_norm)
        return [
            self._ref_o, *text_ids, self._ref_c,
            self._det_o, self._lbr,
            self._coord[x1], self._coord[y1], self._coord[x2], self._coord[y2],
            self._rbr, self._det_c,
        ]

    def encode_label(self, label: dict[str, Any], max_len: int | None = None) -> list[int]:
        ids: list[int] = [self.bos]
        for blk in label.get("blocks", []):
            text = (blk.get("text") or "").strip()
            bbox = blk.get("bbox_norm")
            if not text or not bbox or len(bbox) != 4:
                continue
            ids.extend(self.encode_block(text, bbox))
            if max_len and len(ids) >= max_len - 1:
                ids = ids[: max_len - 1]
                break
        ids.append(self.eos)
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)
