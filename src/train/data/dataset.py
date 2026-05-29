"""PyTorch Dataset + collator over the rendered synthetic OCR pages.

Reads manifests at data/manifests/{train,val}.jsonl (written by src/build_dataset.py).
Each row: {image, prompt, answer, source, label} — we use `image` + `label` (path to the
syn_*.json with blocks + bbox_norm).

The collator pads token sequences to the longest in batch and image-letterboxes each page to a
fixed (H, W) so the encoder receives a uniform tensor. Variable-width inputs are handled by
resizing to fit within (H, W) preserving aspect ratio, then padding with 255 (white).

Training-time augmentation (when train=True) adds per-epoch pixel diversity on top of the
render-time augmentation already baked into the JPEGs. ONLY box-safe ops here — no geometric
transforms, because labels carry tokenized bbox coordinates that would become wrong.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from src.corpus.common import DATA
from src.train.data.label_encoder import LabelEncoder

MANIFESTS = DATA / "manifests"


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def letterbox(img: Image.Image, target_h: int, target_w: int) -> np.ndarray:
    """Resize preserving aspect ratio, then pad with white to (target_h, target_w, 3)."""
    w, h = img.size
    scale = min(target_w / w, target_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR).convert("RGB")
    arr = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    arr[:nh, :nw] = np.asarray(img)
    return arr


# --- Training-time augmentation (box-safe; pixel-only, no geometric ops) -----------------

def _aug_pil(img: Image.Image, rng: random.Random) -> Image.Image:
    """Color / blur jitter on the source PIL image, BEFORE letterbox so padding stays clean."""
    if rng.random() < 0.6:
        # stack brightness / contrast / saturation
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.85, 1.15))  # saturation
    if rng.random() < 0.3:
        # small hue shift via HSV channel rotation
        hsv = np.asarray(img.convert("HSV")).copy()
        shift = int(rng.uniform(-8, 8))
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + shift) % 256
        img = Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
    if rng.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    return img


def _aug_np(arr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Noise + cutout on the letterboxed numpy array."""
    h, w = arr.shape[:2]
    if rng.random() < 0.25:
        sigma = rng.uniform(3.0, 12.0)
        noise = np.random.normal(0, sigma, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.2:
        # cutout: paint a small rect with the dominant (paper) color
        ch = max(8, int(h * rng.uniform(0.03, 0.08)))
        cw = max(8, int(w * rng.uniform(0.03, 0.08)))
        cy = rng.randint(0, h - ch)
        cx = rng.randint(0, w - cw)
        fill = int(np.median(arr))
        arr[cy:cy + ch, cx:cx + cw] = fill
    return arr


class OcrDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path | str,
        encoder: LabelEncoder,
        image_size: tuple[int, int],   # (H, W)
        max_label_len: int = 1024,
        train: bool = False,           # enables per-epoch pixel augmentation
    ):
        self.rows = _load_manifest(Path(manifest_path))
        self.enc = encoder
        self.H, self.W = image_size
        self.max_label_len = max_label_len
        self.train = train
        # data root = parent of `data/` (so manifest's "data\\synthetic\\..." resolves correctly)
        self.root = DATA.parent

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        # manifest stores paths with whatever separator was used at write time (Windows backslashes
        # are common here); normalize before joining.
        img_path = self.root / row["image"].replace("\\", "/")
        label_path = self.root / row["label"].replace("\\", "/")
        with open(label_path, encoding="utf-8") as f:
            label = json.load(f)
        img = Image.open(img_path)
        # Pre-letterbox augmentation (color/blur). Uses module-level `random` whose state is
        # per-DataLoader-worker, so each worker produces independent augmentation streams.
        if self.train:
            img = _aug_pil(img, random)
        arr = letterbox(img, self.H, self.W)
        if self.train:
            arr = _aug_np(arr, random)
        # CHW float, normalized to [0,1]; ImageNet mean/std applied later in the model preprocessor
        pixel = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        token_ids = self.enc.encode_label(label, max_len=self.max_label_len)
        return {
            "pixel_values": pixel,
            "labels": torch.tensor(token_ids, dtype=torch.long),
        }


class Collator:
    """Pad variable-length label sequences; images are already fixed-size from letterbox."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(int(b["labels"].shape[0]) for b in batch)
        labels = torch.full((len(batch), max_len), self.pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            t = b["labels"]
            labels[i, : t.shape[0]] = t
        # HF loss ignore_index convention: replace pad with -100 so cross_entropy skips them
        labels_for_loss = labels.clone()
        labels_for_loss[labels_for_loss == self.pad_id] = -100
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        return {"pixel_values": pixel_values, "labels": labels_for_loss}
