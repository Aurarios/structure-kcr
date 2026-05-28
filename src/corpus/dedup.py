"""Deduplicate the cleaned corpus: exact + MinHash near-duplicate removal across ALL sources.

Web crawls (OSCAR/CC100/mC4/CulturaX) overlap heavily, so cross-source dedup is essential before
the corpus is used for LM training. Khmer has no inter-word spaces, so near-dup shingling uses
character n-grams rather than word tokens.

    python -m src.corpus.dedup [--threshold 0.8] [--ngram 5] [--num-perm 128]

Reads data/corpus/clean/ -> writes data/corpus/dedup/.
"""
from __future__ import annotations

import argparse
import hashlib

from tqdm import tqdm

from .common import CLEAN_DIR, DEDUP_DIR, ShardWriter, doc_from_row, iter_jsonl_dir


def _char_shingles(text: str, k: int) -> set[str]:
    text = text.replace("\n", " ")
    if len(text) <= k:
        return {text}
    return {text[i : i + k] for i in range(len(text) - k + 1)}


def dedup(threshold: float, ngram: int, num_perm: int) -> dict:
    from datasketch import MinHash, MinHashLSH

    files = sorted(CLEAN_DIR.glob("*.jsonl"))
    if not files:
        print(f"No clean shards in {CLEAN_DIR}. Run clean_normalize first.")
        return {}

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    seen_exact: set[str] = set()
    stats = {"in": 0, "kept": 0, "exact_dup": 0, "near_dup": 0}

    with ShardWriter(DEDUP_DIR, "dedup") as w:
        for idx, row in enumerate(tqdm(iter_jsonl_dir(CLEAN_DIR), desc="dedup", unit="doc")):
            stats["in"] += 1
            text = row.get("text", "")
            if not text:
                continue
            # exact dedup
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if h in seen_exact:
                stats["exact_dup"] += 1
                continue
            seen_exact.add(h)
            # near-dup via MinHash/LSH
            m = MinHash(num_perm=num_perm)
            for sh in _char_shingles(text, ngram):
                m.update(sh.encode("utf-8"))
            if lsh.query(m):
                stats["near_dup"] += 1
                continue
            lsh.insert(f"d{idx}", m)
            w.write(doc_from_row(row))
            stats["kept"] += 1
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="exact + near-dup deduplication")
    ap.add_argument("--threshold", type=float, default=0.8, help="Jaccard threshold for near-dup")
    ap.add_argument("--ngram", type=int, default=5, help="character shingle size")
    ap.add_argument("--num-perm", type=int, default=128)
    args = ap.parse_args()

    stats = dedup(args.threshold, args.ngram, args.num_perm)
    print("\ndedup stats:")
    for k, v in stats.items():
        print(f"  {k:12} {v}")
    print(f"-> {DEDUP_DIR}")


if __name__ == "__main__":
    main()
