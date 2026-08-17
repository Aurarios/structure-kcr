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

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.corpus.common import DATA
from src.detect.data.build_det_targets import collect_line_boxes, make_db_targets
from src.train.data.dataset import _aug_np, _aug_pil  # reuse box-safe pixel aug

cv2.setNumThreads(0)   # avoid cv2 threading crashes inside DataLoader workers
MANIFESTS = DATA / "manifests"


def _scan_aug(arr: np.ndarray, rng) -> np.ndarray:
    """Box-safe scan/photo degradation on the letterboxed page (no geometry change).

    Closes the synthetic->production gap for the DETECTOR: real inputs are phone photos / scans that
    are blurry, low-DPI, unevenly lit, JPEG-crushed and noisy. Training on clean renders makes faint
    edges low-probability (the bubble-edge misses we measured). Each op fires independently.
    """
    try:
        h, w = arr.shape[:2]
        # 1) RESOLUTION DEGRADATION (key for blur): downscale then upscale
        if rng.random() < 0.5:
            f = rng.uniform(0.4, 0.85)
            small = cv2.resize(arr, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
            arr = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        # 2) blur (defocus / motion)
        if rng.random() < 0.35:
            k = rng.choice([3, 5])
            if rng.random() < 0.5:
                arr = cv2.GaussianBlur(arr, (k, k), 0)
            else:                                    # horizontal motion blur
                ker = np.zeros((k, k), np.float32); ker[k // 2, :] = 1.0 / k
                arr = cv2.filter2D(arr, -1, ker)
        # 3) uneven lighting gradient
        if rng.random() < 0.3:
            gx = np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.95, 1.2), w, dtype=np.float32)
            arr = np.clip(arr.astype(np.float32) * gx[None, :, None], 0, 255).astype(np.uint8)
        # 4) paper tint / colour cast
        if rng.random() < 0.3:
            tint = np.array([rng.uniform(0.9, 1.0), rng.uniform(0.94, 1.03), rng.uniform(0.97, 1.07)])
            arr = np.clip(arr.astype(np.float32) * tint, 0, 255).astype(np.uint8)
        # 5) sensor noise
        if rng.random() < 0.35:
            arr = np.clip(arr.astype(np.int16) +
                          np.random.normal(0, rng.uniform(3, 16), arr.shape).astype(np.int16),
                          0, 255).astype(np.uint8)
        # 6) JPEG artifacts
        if rng.random() < 0.4:
            ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(35, 80)])
            if ok:
                arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except cv2.error:
        return np.ascontiguousarray(arr)
    return np.ascontiguousarray(arr)


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


def _warp(arr: np.ndarray, boxes, rng, size: int,
          jitter: tuple[float, float] = (0.015, 0.06), max_deg: float = 3.0):
    """Box-aware perspective + rotation warp (for angled phone/scan capture of cards & pages).

    Transforms the page corners by a small random jitter and recomputes each box's axis-aligned
    bound from its warped corners, so the DBNet targets stay correct. Degenerate boxes are dropped.

    NOTE on `jitter`/`max_deg`: each warped box becomes the axis-aligned HULL of its rotated
    corners, so a wide thin line inflates vertically by ~sin(angle)*width — at the defaults a
    full-width line's GT can grow ~3x, and the model then LEARNS loose boxes. The line-level V4/V5
    object detector passes tamer values (obj_dataset.py); the defaults preserve the frozen DBNet
    (block-level) behavior.
    """
    try:
        j = size * rng.uniform(*jitter)
        def jit():
            return rng.uniform(0, j)
        src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
        dst = np.float32([[jit(), jit()], [size - jit(), jit()],
                          [size - jit(), size - jit()], [jit(), size - jit()]])
        M = cv2.getPerspectiveTransform(src, dst)
        # compose a small rotation about the centre
        ang = rng.uniform(-max_deg, max_deg)
        R = cv2.getRotationMatrix2D((size / 2, size / 2), ang, 1.0)
        R = np.vstack([R, [0, 0, 1]]).astype(np.float32)
        M = (M @ R).astype(np.float32)
        arr2 = cv2.warpPerspective(arr, M, (size, size), borderValue=(255, 255, 255),
                                   flags=cv2.INTER_LINEAR)
        out = []
        for (x1, y1, x2, y2), cid in boxes:
            pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
            tp = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
            nx1, ny1 = max(0.0, float(tp[:, 0].min())), max(0.0, float(tp[:, 1].min()))
            nx2, ny2 = min(float(size), float(tp[:, 0].max())), min(float(size), float(tp[:, 1].max()))
            if nx2 - nx1 >= 2 and ny2 - ny1 >= 2:
                out.append(((nx1, ny1, nx2, ny2), cid))
        return np.ascontiguousarray(arr2), out
    except cv2.error:
        return arr, boxes


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
            arr = _scan_aug(arr, random)        # blur / low-res / lighting / noise / JPEG

        boxes = [((x1 * scale, y1 * sq * scale, x2 * scale, y2 * sq * scale), cid)
                 for ((x1, y1, x2, y2), cid) in collect_line_boxes(label, with_class=True)]
        # box-aware perspective/rotation (angled capture) — recomputes boxes so targets stay valid
        if self.train and random.random() < 0.3:
            arr, boxes = _warp(arr, boxes, random, self.size)
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
