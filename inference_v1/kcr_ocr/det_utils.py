"""Letterbox helper extracted from the detector dataset (no training deps)."""
from __future__ import annotations

import numpy as np
from PIL import Image


def letterbox_with_scale(img: Image.Image, size: int):
    """Resize preserving aspect ratio into a size x size white canvas. Returns (array, scale)."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR).convert('RGB')
    arr = np.full((size, size, 3), 255, dtype=np.uint8)
    arr[:nh, :nw] = np.asarray(img)
    return arr, scale
