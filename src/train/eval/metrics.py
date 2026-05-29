"""Evaluation metrics for Khmer document OCR.

- ``cer`` — Character Error Rate (Levenshtein / reference length). Standard OCR metric.
- ``syllable_er`` — Khmer orthographic-syllable ER. Khmer has no word boundaries, so WER is
  ill-defined; orthographic syllables (a base consonant plus its coengs/vowels) are the natural
  grouping. Implemented by clustering combining marks with their preceding base.
- ``box_iou`` — pairwise intersection-over-union for normalized [0,999] boxes.
- ``match_blocks`` — Hungarian-style greedy matching of predicted to ground-truth blocks by IoU.

All functions are pure and tensor-free; they operate on Python strings / lists of ints.
"""
from __future__ import annotations

import unicodedata
from typing import Iterable

# Khmer Unicode block subranges relevant to clustering.
# - Base consonants/independents: U+1780..U+17B3
# - Dependent vowel signs: U+17B6..U+17C5
# - Various signs (nikahit, reahmuk, etc): U+17C6..U+17D3
# - Coeng (subscript marker): U+17D2  (joins the FOLLOWING consonant under the previous base)
_COMBINING_CATS = {"Mn", "Mc", "Me"}


def _is_khmer_combining(ch: str) -> bool:
    if not ch:
        return False
    cp = ord(ch)
    if cp == 0x17D2:                        # coeng
        return True
    if 0x17B6 <= cp <= 0x17D3:              # dependent vowels / signs
        return True
    return unicodedata.category(ch) in _COMBINING_CATS


def khmer_syllables(text: str) -> list[str]:
    """Cluster Khmer combining marks with their preceding base. Non-Khmer runs split per-codepoint."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        cluster = ch
        j = i + 1
        # if the base is a Khmer codepoint or any letter, gobble subsequent combining marks +
        # coeng+next pairs ("្X" sequences)
        while j < n:
            nxt = text[j]
            if ord(nxt) == 0x17D2 and j + 1 < n:
                cluster += nxt + text[j + 1]
                j += 2
                continue
            if _is_khmer_combining(nxt):
                cluster += nxt
                j += 1
                continue
            break
        out.append(cluster)
        i = j
    return out


def _levenshtein(a: list, b: list) -> int:
    """Standard dynamic-programming edit distance on token sequences."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,           # deletion
                curr[j - 1] + 1,       # insertion
                prev[j - 1] + (0 if ca == cb else 1),  # substitution
            )
        prev = curr
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate. Returns 0.0 on empty reference (matches common convention)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    d = _levenshtein(list(reference), list(hypothesis))
    return d / len(reference)


def syllable_er(reference: str, hypothesis: str) -> float:
    """Error rate at the Khmer-orthographic-syllable level."""
    ref = khmer_syllables(reference)
    hyp = khmer_syllables(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    d = _levenshtein(ref, hyp)
    return d / len(ref)


# ---------------------------------------------------------------------------
# Box metrics (operate on normalized [0,999] xyxy boxes; same scale on both sides)

Box = tuple[int, int, int, int]


def box_iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_blocks(
    pred_boxes: list[Box], gt_boxes: list[Box], iou_thresh: float = 0.5
) -> list[tuple[int, int, float]]:
    """Greedy max-IoU matching of predicted boxes to ground-truth boxes.

    Returns a list of (pred_idx, gt_idx, iou) for matches above ``iou_thresh``. Each prediction and
    each ground-truth box is used at most once. Greedy by descending IoU — good enough for OCR
    block matching where boxes rarely overlap heavily.
    """
    pairs: list[tuple[float, int, int]] = []
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            v = box_iou(pb, gb)
            if v >= iou_thresh:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for v, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j)
        matches.append((i, j, v))
    return matches


def box_pr(
    pred_boxes: list[Box], gt_boxes: list[Box], iou_thresh: float = 0.5
) -> dict[str, float]:
    """Precision / Recall / F1 / mean-IoU for box detection at a fixed IoU threshold."""
    matches = match_blocks(pred_boxes, gt_boxes, iou_thresh)
    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    mean_iou = sum(m[2] for m in matches) / max(1, tp)
    return {"precision": precision, "recall": recall, "f1": f1,
            "mean_iou": mean_iou, "tp": float(tp), "fp": float(fp), "fn": float(fn)}


# ---------------------------------------------------------------------------
# Convenience: page-level summary combining text + box

def page_metrics(
    pred_blocks: Iterable[dict], gt_blocks: Iterable[dict], iou_thresh: float = 0.5
) -> dict[str, float]:
    """Compute combined metrics on matched blocks.

    Each block: {"text": str, "bbox_norm": [x1,y1,x2,y2]}. Boxes are matched by IoU; for matched
    pairs we compute CER+SyllER on the text. Unmatched predictions/refs contribute to box P/R only.
    """
    pred = list(pred_blocks); gt = list(gt_blocks)
    pred_boxes = [tuple(b["bbox_norm"]) for b in pred]
    gt_boxes = [tuple(b["bbox_norm"]) for b in gt]
    matches = match_blocks(pred_boxes, gt_boxes, iou_thresh)
    box = box_pr(pred_boxes, gt_boxes, iou_thresh)

    cer_vals: list[float] = []
    syll_vals: list[float] = []
    for i, j, _ in matches:
        ref = gt[j].get("text", "") or ""
        hyp = pred[i].get("text", "") or ""
        cer_vals.append(cer(ref, hyp))
        syll_vals.append(syllable_er(ref, hyp))

    return {
        **box,
        "cer": sum(cer_vals) / max(1, len(cer_vals)) if cer_vals else float("nan"),
        "syllable_er": sum(syll_vals) / max(1, len(syll_vals)) if syll_vals else float("nan"),
        "matched_blocks": float(len(matches)),
    }
