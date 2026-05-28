"""Box sanity gate: bounding boxes must be in-bounds, non-degenerate, and reasonably ordered.

Checks per label:
  - every box within [0, image_dim]
  - x1 < x2 and y1 < y2 (positive area)
  - normalized boxes within [0, 999]
  - reading-order: y-centers largely non-decreasing for single-column pages (multi-column pages
    are exempt from the strict y check)

    python -m src.validate.box_sanity [--max-show 10]
"""
from __future__ import annotations

import argparse
import json

from ..corpus.common import DATA

LABELS = DATA / "synthetic" / "labels"
SINGLE_COL_TEMPLATES = {"article_single", "doc_with_table", "form_label_value", "mixed_km_en"}


def _check_label(label: dict) -> list[str]:
    issues = []
    w, h = label["image_width"], label["image_height"]
    blocks = label["blocks"]
    for i, b in enumerate(blocks):
        x1, y1, x2, y2 = b["bbox"]
        if not (0 <= x1 <= w + 1 and 0 <= x2 <= w + 1 and 0 <= y1 <= h + 1 and 0 <= y2 <= h + 1):
            issues.append(f"block {i} out of bounds {b['bbox']} (img {w}x{h})")
        if x2 <= x1 or y2 <= y1:
            issues.append(f"block {i} non-positive area {b['bbox']}")
        nb = b.get("bbox_norm")
        if nb and not all(0 <= v <= 999 for v in nb):
            issues.append(f"block {i} norm out of range {nb}")
    # reading order (single-column only)
    if label["template"] in SINGLE_COL_TEMPLATES:
        ys = [(b["bbox"][1] + b["bbox"][3]) / 2 for b in blocks if b["block_type"] != "table_cell"]
        inversions = sum(1 for a, c in zip(ys, ys[1:]) if c < a - h * 0.02)
        if inversions > max(1, len(ys) * 0.1):
            issues.append(f"reading order: {inversions} y-inversions")
    return issues


def check(max_show: int) -> dict:
    files = sorted(LABELS.glob("*.json"))
    if not files:
        print(f"No labels in {LABELS}. Run build_dataset first.")
        return {"total": 0, "clean": 0, "violations": 0}

    total = clean = 0
    violation_pages = 0
    shown = 0
    for f in files:
        total += 1
        issues = _check_label(json.loads(f.read_text(encoding="utf-8")))
        if not issues:
            clean += 1
        else:
            violation_pages += 1
            if shown < max_show:
                shown += 1
                print(f"\n✗ {f.name}")
                for it in issues[:5]:
                    print(f"   - {it}")
    print(f"\nbox_sanity: {clean}/{total} clean, {violation_pages} pages with violations")
    if violation_pages == 0:
        print("✓ GATE PASS")
    else:
        print("✗ GATE FAIL — inspect boxes (often augmentation rotation or column breaks)")
    return {"total": total, "clean": clean, "violations": violation_pages}


def main() -> None:
    ap = argparse.ArgumentParser(description="bounding-box sanity gate")
    ap.add_argument("--max-show", type=int, default=10)
    args = ap.parse_args()
    res = check(args.max_show)
    raise SystemExit(0 if res["violations"] == 0 else 1)


if __name__ == "__main__":
    main()
