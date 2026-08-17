"""Contact sheet of detector TRAINING TARGETS (collect_obj_boxes overlays).

Successor of root-level visualize_dataset.py, now working on either a versioned dataset root or a
manifest. Verifies the dataset is correct BEFORE training (line-level boxes, classes, table regions).

  python -m src.dataset_tools.viz --root E:/kcr-v5 --n 64 --out _dataset_viz_v5
  python -m src.dataset_tools.viz --manifest data/manifests_v3/train.jsonl --n 60 --block-level
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.detect.data.build_obj_targets import CLASSES_V4
from src.dataset_tools.common import COLORS, draw_targets, iter_pages, sample_even


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize detector training targets")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path)
    src.add_argument("--manifest", type=Path)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", default="_dataset_viz")
    ap.add_argument("--block-level", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    refs = sample_even(iter_pages(root=args.root, manifest=args.manifest), args.n, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    line_level = not args.block_level

    cards, cls_counts, per_page = [], Counter(), []
    for i, ref in enumerate(refs):
        img, boxes = draw_targets(ref, line_level=line_level)
        if img is None:
            continue
        name = f"{i:03d}_{ref['layout']}.jpg"
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cards.append((name, ref["layout"], len(boxes)))
        per_page.append(len(boxes))
        for _, cid in boxes:
            cls_counts[CLASSES_V4[cid]] += 1

    print(f"[viz] {len(cards)} pages  avg boxes/page {np.mean(per_page):.1f}")
    missing = [c for c in CLASSES_V4 if c not in cls_counts]
    if missing:
        print(f"[viz] classes with ZERO boxes in sample: {missing}")

    legend = " ".join(
        f'<span style="background:rgb({c[2]},{c[1]},{c[0]});padding:2px 6px;margin:2px;'
        f'border-radius:3px;color:#000;font-size:12px">{n}</span>'
        for n, c in [(n, COLORS[n]) for n in CLASSES_V4])
    grid = "".join(
        f'<div style="display:inline-block;margin:6px;vertical-align:top;width:360px">'
        f'<img src="{nm}" style="width:360px;border:1px solid #444" loading="lazy">'
        f'<div style="color:#ccc;font-size:12px">{nm} · {lt} · {nb} boxes</div></div>'
        for nm, lt, nb in cards)
    (out / "index.html").write_text(
        f'<html><body style="background:#111;font-family:sans-serif">'
        f'<h3 style="color:#eee">detector targets — {"LINE" if line_level else "BLOCK"}-level '
        f'({len(cards)} pages)</h3><div>{legend}</div><hr>{grid}</body></html>', encoding="utf-8")
    print(f"[viz] -> {(out / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
