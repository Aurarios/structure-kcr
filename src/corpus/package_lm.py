"""Package the filtered corpus into the final reusable Khmer LM dataset + dataset card.

Outputs to data/corpus/lm_corpus/:
  - lm-*.jsonl        sharded {text, source, license, ...} (for HF datasets / streaming)
  - lm-*.txt          plain-text shards, one document per line (for raw LM tokenization)
  - dataset_card.md   human-readable summary: per-source counts, license breakdown, NC flags, stats
  - dataset_stats.json machine-readable stats

    python -m src.corpus.package_lm [--smoke] [--max-docs N] [--permissive-only]

`--permissive-only` excludes sources flagged non_commercial (e.g. khPOS) so the corpus can be used
without NC restrictions.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from .common import FILTERED_DIR, LM_DIR, ShardWriter, doc_from_row, iter_jsonl_dir, now_iso


def _est_tokens(n_chars: int) -> int:
    # rough proxy for a subword tokenizer on Khmer; refine later with the real tokenizer on GPU box
    return max(1, n_chars // 3)


def package(max_docs: int | None, permissive_only: bool) -> dict:
    files = sorted(FILTERED_DIR.glob("*.jsonl"))
    if not files:
        print(f"No filtered shards in {FILTERED_DIR}. Run quality_filter first.")
        return {}

    LM_DIR.mkdir(parents=True, exist_ok=True)
    per_source = Counter()
    per_license = Counter()
    chars_per_source = defaultdict(int)
    nc_docs = 0
    total_docs = 0
    total_chars = 0

    txt_idx = 0
    txt_count = 0
    txt_fh = open(LM_DIR / f"lm-{txt_idx:05d}.txt", "w", encoding="utf-8")

    with ShardWriter(LM_DIR, "lm") as jsonl_w:
        for row in iter_jsonl_dir(FILTERED_DIR):
            if permissive_only and row.get("non_commercial"):
                continue
            doc = doc_from_row(row)
            n = doc.n_chars or len(doc.text)
            jsonl_w.write(doc)
            # plain text shard (newlines within a doc collapsed to spaces -> one line per doc)
            if txt_count >= 50_000:
                txt_fh.close()
                txt_idx += 1
                txt_fh = open(LM_DIR / f"lm-{txt_idx:05d}.txt", "w", encoding="utf-8")
                txt_count = 0
            txt_fh.write(doc.text.replace("\n", " ") + "\n")
            txt_count += 1

            per_source[doc.source] += 1
            per_license[doc.license] += 1
            chars_per_source[doc.source] += n
            nc_docs += int(bool(doc.non_commercial))
            total_docs += 1
            total_chars += n
            if max_docs and total_docs >= max_docs:
                break
    txt_fh.close()

    stats = {
        "created_at": now_iso(),
        "total_docs": total_docs,
        "total_chars": total_chars,
        "est_tokens": _est_tokens(total_chars),
        "non_commercial_docs": nc_docs,
        "permissive_only": permissive_only,
        "per_source_docs": dict(per_source),
        "per_source_chars": dict(chars_per_source),
        "per_license_docs": dict(per_license),
    }
    (LM_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_card(stats)
    return stats


def _write_card(s: dict) -> None:
    lines = [
        "# Khmer LM Corpus — Dataset Card",
        "",
        f"_Generated {s['created_at']}_",
        "",
        f"- **Documents:** {s['total_docs']:,}",
        f"- **Characters:** {s['total_chars']:,}",
        f"- **Estimated tokens:** ~{s['est_tokens']:,} (char/3 proxy)",
        f"- **Non-commercial-licensed docs:** {s['non_commercial_docs']:,}"
        + ("  ⚠️ present" if s["non_commercial_docs"] else ""),
        f"- **Permissive-only build:** {s['permissive_only']}",
        "",
        "## Documents per source",
        "",
        "| Source | Docs | Chars |",
        "| --- | ---: | ---: |",
    ]
    for src, n in sorted(s["per_source_docs"].items(), key=lambda x: -x[1]):
        lines.append(f"| {src} | {n:,} | {s['per_source_chars'].get(src,0):,} |")
    lines += ["", "## License breakdown", "", "| License | Docs |", "| --- | ---: |"]
    for lic, n in sorted(s["per_license_docs"].items(), key=lambda x: -x[1]):
        lines.append(f"| {lic} | {n:,} |")
    lines += [
        "",
        "## Notes",
        "- Text is Khmer-Unicode-normalized (canonical coeng/vowel ordering, zero-width controls stripped).",
        "- Deduplicated (exact + MinHash near-dup) across all sources.",
        "- Quality-filtered (length, symbol ratio, repetition, Khmer language-id).",
        "- Non-commercial sources are flagged; rebuild with `--permissive-only` to exclude them.",
    ]
    (LM_DIR / "dataset_card.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="package the reusable Khmer LM corpus")
    ap.add_argument("--smoke", action="store_true", help="cap at 500 docs")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--permissive-only", action="store_true",
                    help="exclude non-commercial sources (e.g. khPOS)")
    args = ap.parse_args()

    max_docs = 500 if args.smoke else args.max_docs
    stats = package(max_docs, args.permissive_only)
    if stats:
        print("\npackage_lm stats:")
        print(f"  docs={stats['total_docs']:,} chars={stats['total_chars']:,} "
              f"~tokens={stats['est_tokens']:,} nc={stats['non_commercial_docs']:,}")
        print(f"-> {LM_DIR} (dataset_card.md, dataset_stats.json)")


if __name__ == "__main__":
    main()
