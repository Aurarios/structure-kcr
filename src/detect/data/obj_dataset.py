"""V4 object-detection dataset: page image + region boxes/labels (input-px), from V3 labels.

Reuses the letterbox + box-safe augmentation already built for the DBNet dataset
(`letterbox_with_scale`, `_scan_aug`, `_warp`) and `collect_obj_boxes` for region/block targets.
Boxes are produced in the letterboxed input-px space (what `fcos_loss` expects). A custom
`collate_fn` keeps the variable per-image box count.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.corpus.common import DATA
from src.detect.data.build_obj_targets import collect_obj_boxes, layout_id
from src.detect.data.dataset import _load_manifest, _scan_aug, _warp, letterbox_with_scale
from src.train.data.dataset import _aug_np, _aug_pil


class ObjDetDataset(Dataset):
    def __init__(self, manifest_path: Path | str, image_size: int = 1024, train: bool = False,
                 taxonomy: str = "v4", with_layout: bool = False):
        self.rows = _load_manifest(Path(manifest_path))
        self.size = image_size
        self.train = train
        self.taxonomy = taxonomy          # "v4" (16 classes) or "v6" (adds word_bank)
        self.with_layout = with_layout    # emit page layout id for the auxiliary head
        self.root = DATA.parent

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        label = json.loads((self.root / row["label"].replace("\\", "/")).read_text(encoding="utf-8"))
        img = Image.open(self.root / row["image"].replace("\\", "/"))
        if self.train:
            img = _aug_pil(img, random)
        arr, scale = letterbox_with_scale(img, self.size)
        if self.train:
            arr = _aug_np(arr, random)
            arr = _scan_aug(arr, random)
        boxes = [((x1 * scale, y1 * scale, x2 * scale, y2 * scale), cid)
                 for (x1, y1, x2, y2), cid in collect_obj_boxes(label, taxonomy=self.taxonomy)]
        if self.train and random.random() < 0.25:
            # tame warp for LINE-level targets: at the DBNet defaults (1.5-6% jitter, +-3deg) a
            # full-width thin line's axis-aligned GT hull inflates ~3x and the detector learns
            # loose boxes (measured: F1@0.75 gap). Keep mild skew robustness only.
            arr, boxes = _warp(arr, boxes, random, self.size, jitter=(0.004, 0.02), max_deg=1.2)
        pixel = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float() / 255.0
        if boxes:
            bt = torch.tensor([b for b, _ in boxes], dtype=torch.float)
            lt = torch.tensor([c for _, c in boxes], dtype=torch.long)
        else:
            bt = torch.zeros(0, 4); lt = torch.zeros(0, dtype=torch.long)
        item = {"pixel_values": pixel, "boxes": bt, "labels": lt}
        if self.with_layout:
            item["layout"] = layout_id(row.get("layout_type"))
        return item


def collate_fn(batch: list[dict]) -> dict:
    pixels = torch.stack([b["pixel_values"] for b in batch], 0)
    targets = [{"boxes": b["boxes"], "labels": b["labels"], "layout": b.get("layout")}
               for b in batch]
    return {"pixel_values": pixels, "targets": targets}
