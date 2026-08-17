"""Dataset distribution report + class-balance GATE.

Measures what the detector will actually train on (collect_obj_boxes det targets), per class and
per layout, plus style coverage when labels carry V5 `meta`. The --gate flag turns the known
failure mode (starved classes -> F1≈0) into a hard, testable invariant: exit 1 if any class share
is below its floor.

  python -m src.dataset_tools.stats --root E:/kcr-v3 --per-layout 400
  python -m src.dataset_tools.stats --root E:/kcr-v5 --per-layout 400 --gate
  python -m src.dataset_tools.stats --manifest data/manifests_v3/val.jsonl --out stats.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.detect.data.build_obj_targets import CLASSES_V4, collect_obj_boxes
from src.dataset_tools.common import iter_pages, load_label

# minimum share of det targets per class (the gate). The 1.5% default guards the historically
# near-zero classes (chart/formula were 0.5-0.65% in V3 and scored F1~0.83/starved). Calibrated
# per-class floors: hand_drawing (source assets scarce), table (1 region per whole table), image
# (large-object class, one big easy box per instance, F1 0.949 in V4 at a similar share — share-of-
# line-boxes under-represents its learning signal; fresh V5 renders measure ~1.6% anyway).
DEFAULT_FLOOR = 0.015
SPECIAL_FLOORS = {"hand_drawing": 0.005, "table": 0.005, "image": 0.012}


def collect_stats(refs: list[dict]) -> dict:
    pages_per_layout = Counter(r["layout"] for r in refs)
    targets = Counter()
    targets_by_layout: dict[str, Counter] = defaultdict(Counter)
    line_h: dict[str, list] = defaultdict(list)
    style = {"fonts": Counter(), "title_fonts": Counter(), "colors": Counter(),
             "bgs": Counter(), "engine": Counter()}
    n_ok = 0
    for r in refs:
        label = load_label(r)
        if label is None:
            continue
        n_ok += 1
        for (x1, y1, x2, y2), cid in collect_obj_boxes(label):
            name = CLASSES_V4[cid]
            targets[name] += 1
            targets_by_layout[r["layout"]][name] += 1
            line_h[name].append(y2 - y1)
        meta = label.get("meta") or {}
        if meta:
            style["fonts"][meta.get("font_body") or label.get("font") or "?"] += 1
            style["title_fonts"][meta.get("font_title", "?")] += 1
            style["colors"][str(meta.get("color", "?"))] += 1
            style["bgs"][str(meta.get("bg", "?"))] += 1
            style["engine"][meta.get("engine", "curated")] += 1
        elif label.get("font"):
            style["fonts"][label["font"]] += 1

    total = sum(targets.values())
    return {
        "pages": n_ok,
        "pages_per_layout": dict(pages_per_layout.most_common()),
        "det_targets_total": total,
        "class_share": {c: round(targets.get(c, 0) / max(1, total), 5) for c in CLASSES_V4},
        "class_counts": {c: targets.get(c, 0) for c in CLASSES_V4},
        "line_height_median": {c: round(st.median(v), 1) for c, v in line_h.items() if v},
        "per_layout_class_counts": {k: dict(v.most_common()) for k, v in targets_by_layout.items()},
        "style": {k: dict(v.most_common(25)) for k, v in style.items() if v},
    }


def run_gate(stats: dict, default_floor: float, special: dict[str, float]) -> list[str]:
    fails = []
    for c in CLASSES_V4:
        floor = special.get(c, default_floor)
        share = stats["class_share"].get(c, 0.0)
        if share < floor:
            fails.append(f"{c}: {share:.4f} < floor {floor:.4f} "
                         f"({stats['class_counts'].get(c, 0)} boxes)")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description="Dataset distribution stats + class-balance gate")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="versioned dataset root (per-layout dirs)")
    src.add_argument("--manifest", type=Path, help="manifest jsonl")
    ap.add_argument("--per-layout", type=int, default=400, help="labels sampled per layout (0=all)")
    ap.add_argument("--gate", action="store_true", help="exit 1 if any class below its floor")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR)
    ap.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = ap.parse_args()

    refs = iter_pages(root=args.root, manifest=args.manifest)
    if args.per_layout:
        # PROPORTIONAL sampling (not even-per-layout): "composed" is one dir holding ~50% of all
        # pages — even sampling would weight it like any 6%-sized layout and bias every class share.
        import random as _r
        rng = _r.Random(0)
        by_lay: dict[str, list] = {}
        for r in refs:
            by_lay.setdefault(r["layout"], []).append(r)
        budget = args.per_layout * len(by_lay)
        total = len(refs)
        sampled: list[dict] = []
        for lay in sorted(by_lay):
            lst = by_lay[lay]
            k = min(len(lst), max(40, round(budget * len(lst) / max(1, total))))
            rng.shuffle(lst)
            sampled.extend(lst[:k])
        refs = sampled
    print(f"[stats] analyzing {len(refs)} pages")
    s = collect_stats(refs)

    print(f"\n== pages per layout (total {s['pages']}) ==")
    for k, v in s["pages_per_layout"].items():
        print(f"  {k:24s} {v:7d}")
    print(f"\n== det-target class share (total {s['det_targets_total']}) ==")
    for c in sorted(CLASSES_V4, key=lambda c: -s["class_share"][c]):
        print(f"  {c:14s} {100 * s['class_share'][c]:6.2f}%  ({s['class_counts'][c]})")
    if s.get("style"):
        eng = s["style"].get("engine")
        if eng:
            print(f"\n== engine mix == {eng}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n[stats] wrote {args.out}")

    if args.gate:
        fails = run_gate(s, args.floor, SPECIAL_FLOORS)
        if fails:
            print("\n[GATE FAIL] starved classes:")
            for f in fails:
                print(f"  {f}")
            sys.exit(1)
        print("\n[GATE PASS] all class floors met")


if __name__ == "__main__":
    main()
