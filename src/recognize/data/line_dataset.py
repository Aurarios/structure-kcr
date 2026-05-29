"""Recognizer dataset: line crop (H=48, variable W) -> image + attn/CTC targets.

Reads the manifest written by build_line_crops.py (rows {crop, text}); crop paths are relative to
the dataset root (the --out-dir used at build time, e.g. E:\\kcr-recognize\\lines). Crops are already
saved at H=48, so no resize here. Light pixel augmentation at train time; the collator right-pads
images to the batch-max width and pads attention targets.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.recognize.data.text_encoder import TextEncoder

CROP_H = 48
DOWNSAMPLE = 8        # stem reduces width by 8 -> CTC time steps = W // 8


def _aug(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.5:
        a = rng.uniform(0.8, 1.2); b = rng.uniform(-15, 15)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        sigma = rng.uniform(2.0, 10.0)
        img = np.clip(img.astype(np.int16) + np.random.normal(0, sigma, img.shape).astype(np.int16),
                      0, 255).astype(np.uint8)
    if rng.random() < 0.15:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


class LineDataset(Dataset):
    def __init__(self, manifest_path: Path | str, root: Path | str, encoder: TextEncoder,
                 train: bool = False, max_w: int = 1200):
        self.root = Path(root)
        self.enc = encoder
        self.train = train
        self.max_w = max_w
        self.rows: list[dict[str, Any]] = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        img = cv2.imread(str(self.root / row["crop"]))
        if img is None:
            img = np.full((CROP_H, 32, 3), 255, np.uint8)
        if img.shape[0] != CROP_H:
            scale = CROP_H / img.shape[0]
            img = cv2.resize(img, (max(8, int(img.shape[1] * scale)), CROP_H))
        if img.shape[1] > self.max_w:
            img = cv2.resize(img, (self.max_w, CROP_H))
        if self.train:
            img = _aug(img, random)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixel = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        text = row["text"]
        return {
            "pixel": pixel,
            "width": pixel.shape[2],
            "attn": torch.tensor(self.enc.encode_attn(text), dtype=torch.long),
            "ctc": torch.tensor(self.enc.encode_ctc(text), dtype=torch.long),
        }


class RecCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        maxw = max(b["width"] for b in batch)
        maxw = ((maxw + DOWNSAMPLE - 1) // DOWNSAMPLE) * DOWNSAMPLE   # multiple of 8
        B = len(batch)
        images = torch.ones(B, 3, CROP_H, maxw)                       # white pad (1.0)
        input_lengths = torch.zeros(B, dtype=torch.long)
        for i, b in enumerate(batch):
            w = b["width"]
            images[i, :, :, :w] = b["pixel"]
            input_lengths[i] = w // DOWNSAMPLE

        # attention targets (CE, ignore pad)
        maxl = max(b["attn"].shape[0] for b in batch)
        attn = torch.full((B, maxl), self.pad_id, dtype=torch.long)
        for i, b in enumerate(batch):
            attn[i, : b["attn"].shape[0]] = b["attn"]

        # CTC targets: flattened + lengths
        ctc_targets = torch.cat([b["ctc"] for b in batch]) if batch else torch.empty(0, dtype=torch.long)
        ctc_lengths = torch.tensor([b["ctc"].shape[0] for b in batch], dtype=torch.long)
        return {"images": images, "input_lengths": input_lengths,
                "attn": attn, "ctc_targets": ctc_targets, "ctc_lengths": ctc_lengths}
