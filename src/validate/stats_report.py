"""Dataset stats: coverage per font/template, block-type counts, char/box distributions.

    python -m src.validate.stats_report
Writes data/manifests/stats_report.json and prints a summary.
"""
from __future__ import annotations

import json
import statistics
from collections import Counter

from ..corpus.common import DATA

LABELS = DATA / "synthetic" / "labels"
OUT = DATA / "manifests" / "stats_report.json"


def run() -> dict:
    files = sorted(LABELS.glob("*.json"))
    if not files:
        print(f"No labels in {LABELS}. Run build_dataset first.")
        return {}

    per_font = Counter()
    per_template = Counter()
    block_types = Counter()
    chars_per_page = []
    boxes_per_page = []

    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        per_font[d["font"]] += 1
        per_template[d["template"]] += 1
        boxes_per_page.append(len(d["blocks"]))
        page_chars = 0
        for b in d["blocks"]:
            block_types[b["block_type"]] += 1
            page_chars += len(b["text"])
        chars_per_page.append(page_chars)

    stats = {
        "pages": len(files),
        "per_font": dict(per_font),
        "per_template": dict(per_template),
        "block_types": dict(block_types),
        "chars_per_page": {
            "mean": round(statistics.mean(chars_per_page), 1),
            "min": min(chars_per_page),
            "max": max(chars_per_page),
        },
        "boxes_per_page": {
            "mean": round(statistics.mean(boxes_per_page), 1),
            "min": min(boxes_per_page),
            "max": max(boxes_per_page),
        },
    }
    OUT.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pages: {stats['pages']}")
    print("\nper template:")
    for k, v in sorted(per_template.items(), key=lambda x: -x[1]):
        print(f"  {k:18} {v}")
    print("\nper font:")
    for k, v in sorted(per_font.items(), key=lambda x: -x[1]):
        print(f"  {k:24} {v}")
    print("\nblock types:")
    for k, v in sorted(block_types.items(), key=lambda x: -x[1]):
        print(f"  {k:14} {v}")
    print(f"\nchars/page mean={stats['chars_per_page']['mean']}  "
          f"boxes/page mean={stats['boxes_per_page']['mean']}")
    print(f"-> {OUT}")
    return stats


if __name__ == "__main__":
    run()
