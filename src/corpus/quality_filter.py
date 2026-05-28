"""LM-grade quality filtering of the deduped corpus.

Heuristics (each doc is *flagged* with reasons, not silently dropped — kept docs go to
data/corpus/filtered/, rejected docs with reasons go to data/corpus/filtered_rejected/ for audit):
  - length bounds
  - symbol / digit ratio too high (menus, tables-of-numbers, junk)
  - excessive repetition (spam, boilerplate)
  - Khmer language-id confidence. Uses fasttext lid.176 if a model is available
    (env KHMER_LID_MODEL or data/fasttext/lid.176.ftz); otherwise falls back to the khmer_ratio
    already stored as lang_mix.

    python -m src.corpus.quality_filter [--min-chars 25] [--max-symbol-ratio 0.3] [--min-khmer 0.6]
"""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from .. import khmer_utils as ku
from .common import DEDUP_DIR, FILTERED_DIR, FILTERED_REJECTED_DIR, ShardWriter, doc_from_row, iter_jsonl_dir

_lid_model = None
_lid_tried = False


def _load_lid():
    global _lid_model, _lid_tried
    if _lid_tried:
        return _lid_model
    _lid_tried = True
    path = os.environ.get("KHMER_LID_MODEL") or "data/fasttext/lid.176.ftz"
    if Path(path).exists():
        try:
            import fasttext
            _lid_model = fasttext.load_model(path)
            print(f"[quality] using fasttext lid model: {path}")
        except Exception as e:
            print(f"[quality] fasttext load failed ({e}); using khmer_ratio proxy")
    else:
        print("[quality] no fasttext lid model found; using khmer_ratio proxy for language id")
    return _lid_model


def _lang_km_conf(text: str, lang_mix: float | None) -> float:
    model = _load_lid()
    if model is None:
        return lang_mix if lang_mix is not None else ku.khmer_ratio(text)
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    return float(probs[0]) if labels and labels[0] == "__label__km" else 0.0


def _symbol_ratio(text: str) -> float:
    t = [c for c in text if not c.isspace()]
    if not t:
        return 1.0
    sym = sum(1 for c in t if not (c.isalnum() or ku.is_khmer_char(c)))
    return sym / len(t)


def _repetition_ratio(text: str) -> float:
    """Fraction of the top duplicated line — high => boilerplate/spam."""
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return 0.0
    counts = Counter(lines)
    return counts.most_common(1)[0][1] / len(lines)


def assess(row: dict, min_chars: int, max_symbol: float, min_khmer: float,
           max_rep: float) -> list[str]:
    text = row.get("text", "")
    reasons = []
    if len(text) < min_chars:
        reasons.append("too_short")
    if _symbol_ratio(text) > max_symbol:
        reasons.append("symbol_ratio")
    if _repetition_ratio(text) > max_rep:
        reasons.append("repetition")
    if _lang_km_conf(text, row.get("lang_mix")) < min_khmer:
        reasons.append("low_khmer_conf")
    return reasons


def run(min_chars: int, max_symbol: float, min_khmer: float, max_rep: float) -> dict:
    files = sorted(DEDUP_DIR.glob("*.jsonl"))
    if not files:
        print(f"No deduped shards in {DEDUP_DIR}. Run dedup first.")
        return {}

    stats = {"in": 0, "kept": 0, "rejected": 0}
    reason_counts: Counter = Counter()
    with ShardWriter(FILTERED_DIR, "lm") as keep, ShardWriter(FILTERED_REJECTED_DIR, "rej") as rej:
        for row in tqdm(iter_jsonl_dir(DEDUP_DIR), desc="quality", unit="doc"):
            stats["in"] += 1
            reasons = assess(row, min_chars, max_symbol, min_khmer, max_rep)
            if reasons:
                stats["rejected"] += 1
                reason_counts.update(reasons)
                row = dict(row, extra={**row.get("extra", {}), "reject_reasons": reasons})
                rej.write(row)
            else:
                keep.write(doc_from_row(row))
                stats["kept"] += 1
    stats["reasons"] = dict(reason_counts)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="LM-grade quality filter")
    ap.add_argument("--min-chars", type=int, default=25)
    ap.add_argument("--max-symbol-ratio", type=float, default=0.30)
    ap.add_argument("--min-khmer", type=float, default=0.60)
    ap.add_argument("--max-repetition", type=float, default=0.50)
    args = ap.parse_args()

    stats = run(args.min_chars, args.max_symbol_ratio, args.min_khmer, args.max_repetition)
    print("\nquality_filter stats:")
    for k, v in stats.items():
        print(f"  {k:12} {v}")
    print(f"-> kept: {FILTERED_DIR}\n-> rejected (audit): {FILTERED_REJECTED_DIR}")


if __name__ == "__main__":
    main()
