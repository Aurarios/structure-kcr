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

# OpenCV's internal threading throws "Unknown C++ exception" when called from many DataLoader
# worker processes; disabling it makes cv2 calls in workers safe (we parallelize via workers).
cv2.setNumThreads(0)

CROP_H = 48
DOWNSAMPLE = 8        # stem reduces width by 8 -> CTC time steps = W // 8


def _aug(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Scan-realistic augmentation to close the synthetic->real gap.

    Diagnosis (confirmed by isolation): the recognizer reads synthetic renders near-perfectly but
    fails on real scans, because it only ever saw crisp high-DPI crops. Real low-DPI scans are
    blurry after upscale, paper-tinted, noisy, skewed, JPEG-degraded -> out of distribution. This
    simulates those conditions so the model learns scan-robust features. Each op fires independently.
    """
    try:
        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        # 1) RESOLUTION DEGRADATION (the key one): downscale then upscale -> low-DPI scan blur
        if rng.random() < 0.5:
            f = rng.uniform(0.35, 0.8)
            small = cv2.resize(img, (max(4, int(w * f)), max(4, int(h * f))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        # 2) PAPER TINT / colour cast (cream, gray, warm) — real scans aren't pure white
        if rng.random() < 0.4:
            tint = np.array([rng.uniform(0.92, 1.0), rng.uniform(0.95, 1.02), rng.uniform(0.98, 1.06)])
            img = np.clip(img.astype(np.float32) * tint, 0, 255).astype(np.uint8)
        # 3) LIGHTING GRADIENT (uneven scan illumination)
        if rng.random() < 0.25:
            grad = np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.95, 1.15), w, dtype=np.float32)
            img = np.clip(img.astype(np.float32) * grad[None, :, None], 0, 255).astype(np.uint8)
        # 4) brightness / contrast / gamma (lighting + ink darkness)
        if rng.random() < 0.5:
            a = rng.uniform(0.7, 1.3); b = rng.uniform(-25, 25)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if rng.random() < 0.25:
            g = rng.uniform(0.7, 1.4)
            lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)], np.uint8)
            img = cv2.LUT(img, lut)
        # 5) blur (scan defocus / anti-alias)
        if rng.random() < 0.3 and h >= 7 and w >= 7:
            k = rng.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        # 6) slight rotation/skew (page not perfectly flat)
        if rng.random() < 0.3:
            ang = rng.uniform(-2.0, 2.0)
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderValue=(245, 245, 245), flags=cv2.INTER_LINEAR)
        # 7) sensor + speckle noise
        if rng.random() < 0.4:
            sigma = rng.uniform(3.0, 18.0)
            img = np.clip(img.astype(np.int16) + np.random.normal(0, sigma, img.shape).astype(np.int16),
                          0, 255).astype(np.uint8)
        # 8) JPEG compression artifacts (lower quality than the training crops)
        if rng.random() < 0.4:
            q = rng.randint(35, 80)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except cv2.error:
        return np.ascontiguousarray(img)
    return img


class LineDataset(Dataset):
    def __init__(self, manifest_path: Path | str, root: Path | str, encoder: TextEncoder,
                 train: bool = False, max_w: int = 1200, max_chars: int = 220,
                 max_tgt_len: int = 256):
        self.root = Path(root)
        self.enc = encoder
        self.train = train
        self.max_w = max_w
        self.max_tgt_len = max_tgt_len    # hard clamp on target token length (crash guard)
        self.rows: list[dict[str, Any]] = []
        dropped = 0
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                # drop pathologically long "lines" (usually line-grouping anomalies); cheap
                # char-length proxy avoids tokenizing 3M texts at init
                if len(row.get("text", "")) > max_chars:
                    dropped += 1
                    continue
                self.rows.append(row)
        if dropped:
            print(f"[line_dataset] dropped {dropped} over-long lines (>{max_chars} chars)")

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
        attn = self.enc.encode_attn(text)
        ctc = self.enc.encode_ctc(text)
        # hard crash-guard: clamp to the model's positional-encoding capacity (rare after the
        # char filter; truncation keeps a trailing eos so the sequence stays well-formed)
        if len(attn) > self.max_tgt_len:
            attn = attn[: self.max_tgt_len - 1] + [self.enc.eos]
        if len(ctc) > self.max_tgt_len:
            ctc = ctc[: self.max_tgt_len]
        return {
            "pixel": pixel,
            "width": pixel.shape[2],
            "attn": torch.tensor(attn, dtype=torch.long),
            "ctc": torch.tensor(ctc, dtype=torch.long),
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
