"""Shared helpers for dataset tools: page iteration (per-layout root OR manifest), target drawing.

A "page ref" is a dict {label: Path, image: Path, layout: str} regardless of source, so every tool
works on both a versioned dataset root (E:/kcr-v5/<layout>/{images,labels}) and a manifest jsonl.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.corpus.common import DATA
from src.detect.data.build_obj_targets import CLASSES_V4, collect_obj_boxes

ROOT = DATA.parent

_rng = np.random.RandomState(1)
COLORS = {c: tuple(int(x) for x in _rng.randint(70, 256, 3)) for c in CLASSES_V4}
COLORS.update({"text": (0, 200, 0), "title": (255, 0, 200), "heading": (210, 0, 255),
               "subheading": (255, 120, 0), "list_item": (0, 180, 230), "caption": (180, 180, 0),
               "table": (0, 0, 255), "table_cell": (0, 170, 0), "table_head": (0, 90, 255),
               "image": (255, 80, 0), "chart": (255, 170, 0), "formula": (0, 210, 255),
               "signature": (230, 0, 230), "form_label": (0, 160, 210), "form_value": (90, 90, 240)})


def _resolve(p: str) -> Path:
    pp = Path(p.replace("\\", "/"))
    return pp if pp.is_absolute() else ROOT / pp


def _image_for(labels_dir: Path, stem: str) -> Path | None:
    images = labels_dir.parent / "images"
    for ext in (".jpg", ".png", ".jpeg"):
        if (images / f"{stem}{ext}").exists():
            return images / f"{stem}{ext}"
    return None


def iter_pages(root: Path | None = None, manifest: Path | None = None,
               layouts: list[str] | None = None) -> list[dict]:
    """Page refs from a per-layout dataset root or a manifest jsonl (exactly one required)."""
    refs: list[dict] = []
    if root is not None:
        for ld in sorted(d for d in Path(root).iterdir() if (d / "labels").is_dir()):
            if layouts and ld.name not in layouts:
                continue
            for lf in sorted((ld / "labels").glob("*.json")):
                refs.append({"label": lf, "image": _image_for(ld / "labels", lf.stem),
                             "layout": ld.name})
    elif manifest is not None:
        for line in Path(manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if layouts and row.get("layout_type") not in layouts:
                continue
            refs.append({"label": _resolve(row["label"]), "image": _resolve(row["image"]),
                         "layout": row.get("layout_type", "?")})
    else:
        raise ValueError("need --root or --manifest")
    return refs


def sample_even(refs: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Sample ~n refs with even coverage across layouts."""
    rng = random.Random(seed)
    by_lay: dict[str, list] = {}
    for r in refs:
        by_lay.setdefault(r["layout"], []).append(r)
    out: list[dict] = []
    per = max(1, n // max(1, len(by_lay)))
    for lay in sorted(by_lay):
        rng.shuffle(by_lay[lay])
        out.extend(by_lay[lay][:per])
    rng.shuffle(out)
    return out[:n]


def load_label(ref: dict) -> dict | None:
    try:
        return json.loads(Path(ref["label"]).read_text(encoding="utf-8"))
    except Exception:
        return None


def draw_targets(ref: dict, line_level: bool = True):
    """Image with `collect_obj_boxes()` training targets drawn. Returns (img, boxes) or (None, [])."""
    label = load_label(ref)
    img = cv2.imread(str(ref["image"])) if ref.get("image") else None
    if label is None or img is None:
        return None, []
    boxes = collect_obj_boxes(label, line_level=line_level)
    for (x1, y1, x2, y2), cid in boxes:
        name = CLASSES_V4[cid]
        col = COLORS.get(name, (140, 140, 140))
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        ins = 2          # inset so stacked line boxes show a visible gap
        cv2.rectangle(img, (x1 + ins, y1 + ins), (max(x1 + ins + 1, x2 - ins),
                      max(y1 + ins + 1, y2 - ins)), col, 3 if name == "table" else 2)
        if name != "text":
            cv2.putText(img, name, (x1 + 1, max(11, y1 + 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, col, 1, cv2.LINE_AA)
    return img, boxes
