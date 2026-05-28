"""Clean + normalize raw corpus docs into data/corpus/clean/.

For each raw Doc:
  - normalize Khmer Unicode (canonical coeng/vowel ordering, strip zero-width controls)
  - split into reasonable segments (paragraphs) so downstream rendering/LM packing has clean units
  - script-purity filter: keep docs that are mostly Khmer, but preserve intentional Khmer+English
  - record khmer_ratio (lang_mix) and char count for later quality filtering

    python -m src.corpus.clean_normalize [--min-khmer 0.5] [--min-chars 20]
"""
from __future__ import annotations

import argparse
import re

from tqdm import tqdm

from .. import khmer_utils as ku
from .common import CLEAN_DIR, RAW_DIR, Doc, ShardWriter, iter_jsonl_dir

_HTML_TAG = re.compile(r"<[^>]+>")
_CSS_STYLE = re.compile(r"\b[\w-]+\s*:\s*[^;{}]+;")          # leftover inline CSS declarations
_HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_URL = re.compile(r"https?://\S+")


def clean_text(text: str) -> str:
    # strip residual HTML/CSS/markup that leaks from some web/wiki sources, then normalize Khmer
    text = _HTML_TAG.sub(" ", text)
    text = _CSS_STYLE.sub(" ", text)
    text = _HTML_ENTITY.sub(" ", text)
    text = _URL.sub(" ", text)
    return ku.normalize(text)


def process(min_khmer: float, min_chars: int, max_chars: int) -> dict:
    stats = {"in": 0, "kept": 0, "dropped_short": 0, "dropped_nonkhmer": 0, "segments": 0}
    raw_files = sorted(RAW_DIR.glob("*.jsonl"))
    if not raw_files:
        print(f"No raw shards in {RAW_DIR}. Run: python -m src.corpus.registry --fetch --smoke")
        return stats

    with ShardWriter(CLEAN_DIR, "clean") as w:
        for row in tqdm(iter_jsonl_dir(RAW_DIR), desc="clean", unit="doc"):
            stats["in"] += 1
            text = clean_text(row.get("text", ""))
            if not text:
                continue
            # split long docs into paragraph segments; keep short docs whole
            segments = ku.split_paragraphs(text) or [text]
            for seg in segments:
                seg = seg.strip()
                if len(seg) < min_chars or len(seg) > max_chars:
                    stats["dropped_short"] += 1
                    continue
                ratio = ku.khmer_ratio(seg)
                if ratio < min_khmer:
                    stats["dropped_nonkhmer"] += 1
                    continue
                w.write(Doc(
                    text=seg,
                    source=row.get("source", "unknown"),
                    license=row.get("license", "unknown"),
                    url=row.get("url"),
                    lang_mix=round(ratio, 3),
                    n_chars=len(seg),
                    non_commercial=bool(row.get("non_commercial", False)),
                    fetched_at=row.get("fetched_at"),
                    extra=row.get("extra", {}),
                ))
                stats["kept"] += 1
                stats["segments"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="clean + normalize raw corpus")
    ap.add_argument("--min-khmer", type=float, default=0.5,
                    help="min fraction of letters that must be Khmer (lower keeps more mixed text)")
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=20000)
    args = ap.parse_args()

    stats = process(args.min_khmer, args.min_chars, args.max_chars)
    print("\nclean_normalize stats:")
    for k, v in stats.items():
        print(f"  {k:18} {v}")
    print(f"-> {CLEAN_DIR}")


if __name__ == "__main__":
    main()
