"""Detector dataset: page image + DBNet target maps, built from existing line boxes.

Reuses the SAME manifest + deterministic val split as the AR baseline (reads `label` paths from
data/manifests/{train,val}.jsonl) so detector metrics are comparable. No re-render: line boxes come
straight from each label's `blocks[].lines`.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.corpus.common import DATA
from src.detect.data.build_det_targets import collect_line_boxes, make_db_targets
from src.train.data.dataset import _aug_np, _aug_pil  # reuse box-safe pixel aug

MANIFESTS = DATA / "manifests"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def letterbox_with_scale(img: Image.Image, size: int) -> tuple[np.ndarray, float]:
    """Resize preserving aspect ratio into a size×size white canvas. Returns (array, scale)."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR).convert("RGB")
    arr = np.full((size, size, 3), 255, dtype=np.uint8)
    arr[:nh, :nw] = np.asarray(img)
    return arr, scale


class DetDataset(Dataset):
    def __init__(self, manifest_path: Path | str, image_size: int = 960, train: bool = False,
                 shrink_ratio: float = 0.4):
        self.rows = _load_manifest(Path(manifest_path))
        self.size = image_size
        self.train = train
        self.shrink_ratio = shrink_ratio
        self.root = DATA.parent

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        row = self.rows[i]
        label_path = self.root / row["label"].replace("\\", "/")
        with open(label_path, encoding="utf-8") as f:
            label = json.load(f)
        img_path = self.root / row["image"].replace("\\", "/")
        img = Image.open(img_path)
        if self.train:
            img = _aug_pil(img, random)

        # vertical-squash augmentation: compress the page vertically so lines get closer ->
        # teaches the detector to SEPARATE tightly-spaced lines (the real-doc merging failure).
        # No re-render needed: derived from the existing generous-spacing synthetic pages.
        sq = 1.0
        if self.train and random.random() < 0.5:
            sq = random.uniform(0.55, 0.95)
            w, h = img.size
            img = img.resize((w, max(1, int(h * sq))), Image.BILINEAR)

        arr, scale = letterbox_with_scale(img, self.size)
        if self.train:
            arr = _aug_np(arr, random)

        boxes = [((x1 * scale, y1 * sq * scale, x2 * scale, y2 * sq * scale), cid)
                 for ((x1, y1, x2, y2), cid) in collect_line_boxes(label, with_class=True)]
        prob, prob_mask, thresh, thresh_mask, class_map = make_db_targets(
            boxes, self.size, self.size, shrink_ratio=self.shrink_ratio)

        pixel = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return {
            "pixel_values": pixel,
            "prob": torch.from_numpy(prob),
            "prob_mask": torch.from_numpy(prob_mask),
            "thresh": torch.from_numpy(thresh),
            "thresh_mask": torch.from_numpy(thresh_mask),
            "class_map": torch.from_numpy(class_map).long(),
        }


def gt_boxes_for_eval(label: dict[str, Any]) -> list[list[int]]:
    """Ground-truth line boxes normalized to [0,999] for box_pr evaluation."""
    w = label["image_width"]
    h = label["image_height"]
    out: list[list[int]] = []
    for (x1, y1, x2, y2) in collect_line_boxes(label):
        out.append([
            max(0, min(999, round(x1 / w * 999))), max(0, min(999, round(y1 / h * 999))),
            max(0, min(999, round(x2 / w * 999))), max(0, min(999, round(y2 / h * 999))),
        ])
    return out
