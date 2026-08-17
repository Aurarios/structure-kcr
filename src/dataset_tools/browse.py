"""Find and eyeball dataset pages by content: class, layout, font — overlay contact sheet.

The debugging story for rare classes: "show me 20 pages that contain a formula" without scrolling
through thousands of files.

  python -m src.dataset_tools.browse --root E:/kcr-v3 --class chart --n 20
  python -m src.dataset_tools.browse --root E:/kcr-v5 --layout composed --class table --n 12
  python -m src.dataset_tools.browse --manifest data/manifests_v3/val.jsonl --font Bayon --n 12
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

from src.detect.data.build_obj_targets import CLASSES_V4, collect_obj_boxes
from src.dataset_tools.common import draw_targets, iter_pages, load_label


def matches(ref: dict, want_cls: str | None, want_font: str | None) -> bool:
    if not (want_cls or want_font):
        return True
    label = load_label(ref)
    if label is None:
        return False
    if want_cls:
        names = {CLASSES_V4[cid] for _, cid in collect_obj_boxes(label)}
        if want_cls not in names:
            return False
    if want_font:
        meta = label.get("meta") or {}
        fonts = {label.get("font", ""), meta.get("font_body", ""), meta.get("font_title", "")}
        if not any(want_font.lower() in f.lower() for f in fonts):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Browse dataset pages by class/layout/font")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path)
    src.add_argument("--manifest", type=Path)
    ap.add_argument("--class", dest="cls", default=None, choices=list(CLASSES_V4) + [None])
    ap.add_argument("--layout", default=None)
    ap.add_argument("--font", default=None, help="substring match on body/title font")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="_browse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scan-cap", type=int, default=4000, help="max labels to scan for matches")
    args = ap.parse_args()

    refs = iter_pages(root=args.root, manifest=args.manifest,
                      layouts=[args.layout] if args.layout else None)
    random.Random(args.seed).shuffle(refs)
    hits, scanned = [], 0
    for ref in refs:
        scanned += 1
        if matches(ref, args.cls, args.font):
            hits.append(ref)
            if len(hits) >= args.n:
                break
        if scanned >= args.scan_cap:
            break
    print(f"[browse] {len(hits)} matches (scanned {scanned}/{len(refs)})")
    if not hits:
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cards = []
    for i, ref in enumerate(hits):
        img, boxes = draw_targets(ref)
        if img is None:
            continue
        name = f"{i:03d}_{ref['layout']}.jpg"
        cv2.imwrite(str(out / name), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        cards.append((name, ref["layout"], Path(ref["label"]).stem, len(boxes)))

    q = " ".join(f"{k}={v}" for k, v in
                 [("class", args.cls), ("layout", args.layout), ("font", args.font)] if v) or "all"
    grid = "".join(
        f'<div style="display:inline-block;margin:6px;vertical-align:top;width:380px">'
        f'<img src="{nm}" style="width:380px;border:1px solid #444" loading="lazy">'
        f'<div style="color:#ccc;font-size:12px">{pid} · {lt} · {nb} boxes</div></div>'
        for nm, lt, pid, nb in cards)
    (out / "index.html").write_text(
        f'<html><body style="background:#111;font-family:sans-serif">'
        f'<h3 style="color:#eee">browse: {q} ({len(cards)} pages)</h3>{grid}</body></html>',
        encoding="utf-8")
    print(f"[browse] -> {(out / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
