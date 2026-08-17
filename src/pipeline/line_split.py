"""Split a detected text block into recognizer line boxes via horizontal ink projection.

The V4 detector outputs region/block boxes (title, text paragraph, caption, table cell, form value,
...). The Khmer recognizer needs LINE crops. For multi-line text blocks we split by projecting ink
onto the y-axis and cutting at vertical gaps; single-line regions pass through unchanged. Fast, no
model. (If real-doc skew degrades projection, a DBNet-on-block fallback can replace `split_block`.)
"""
from __future__ import annotations

import numpy as np

# block classes that are typically multi-line and benefit from splitting
MULTILINE = {"text", "list_item", "caption"}


def split_block(gray: np.ndarray, box, min_line_h: int = 10, rel: float = 0.30, smooth: int = 3):
    """gray: full-page grayscale uint8. box: [x1,y1,x2,y2] page px. Returns list of line boxes.

    Cuts lines at valleys in the horizontal ink projection using a **relative** threshold
    (ink > rel*peak). Khmer stacks coeng/vowels into the inter-line gap, so the projection never
    drops to a clean valley -- an absolute threshold (the old `0.04*width`) saw the whole block as
    one band and never split. A relative cut finds the gaps; a final force-split handles any two
    lines whose valley still never dropped below rel*peak. Returns [box] if it looks single-line.
    """
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    H, W = gray.shape[:2]
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return [box]
    crop = gray[y1:y2, x1:x2]
    # ink = darker than a local threshold (Otsu-ish via mean - k*std)
    thr = max(60, int(crop.mean() - 0.5 * crop.std()))
    ink = (crop < thr).sum(axis=1).astype(np.float32)        # ink per row
    if ink.max() < 1:
        return [box]
    if smooth > 1:                                           # denoise the projection
        ink = np.convolve(ink, np.ones(smooth) / smooth, mode="same")
    peak = float(np.percentile(ink, 92))                     # robust line-center ink level
    if peak < 1:
        return [box]
    rows = ink > (rel * peak)                                # row is text (vs inter-line valley)
    bands = []
    s = None
    for i, on in enumerate(rows):
        if on and s is None:
            s = i
        elif not on and s is not None:
            bands.append((s, i)); s = None
    if s is not None:
        bands.append((s, len(rows)))
    bands = [(a, b) for a, b in bands if b - a >= min_line_h]
    if len(bands) <= 1:
        return [box]
    # force-split a band much taller than the median line (two adjacent lines fused because the
    # valley between them never dropped below rel*peak)
    heights = sorted(b - a for a, b in bands)
    med = heights[len(heights) // 2] or min_line_h
    out = []
    for a, b in bands:
        h = b - a
        k = int(round(h / med)) if med > 0 else 1
        if k >= 2 and h >= 2 * min_line_h:
            step = h / k
            out.extend((a + int(j * step), a + int((j + 1) * step)) for j in range(k))
        else:
            out.append((a, b))
    return [[float(x1), float(y1 + a), float(x2), float(y1 + b)] for a, b in out] or [box]


def lines_for_detection(gray: np.ndarray, box, class_name: str):
    """Line boxes to recognize for a detected region. Multi-line text classes are split; everything
    else (title/heading/cells/form values/...) is a single line."""
    if class_name in MULTILINE:
        return split_block(gray, box)
    return [box]
