"""Build DBNet training targets from text-line boxes.

The synthetic labels already store per-line rectangles (`blocks[].lines`, image-pixel space). The
detector ground truth is the union of all line boxes on a page. DBNet needs three maps:

  prob_map   : 1 inside the SHRUNK line box (Differentiable-Binarization "probability" target)
  thresh_map : a border band around each box, ramping thresh_max (at the box edge) -> thresh_min
               (at distance d outward/inward); only this band contributes to the L1 loss
  thresh_mask: 1 where the threshold band is defined, else 0

Because our line boxes are axis-aligned rectangles we shrink/expand with closed-form rectangle
offsets (d = A*(1-r^2)/L), avoiding pyclipper/shapely. Reference: Liao et al., "Real-time Scene
Text Detection with Differentiable Binarization" (AAAI 2020).
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

Box = tuple[float, float, float, float]

# Region-type taxonomy (index 0 = background). Restores the document structure the flat
# line-detector dropped: assembly maps these back to markdown (#, ##, -, tables, ...).
# `image`   = croppable non-text region (figure / photo / logo / stamp / signature) -> at inference
#             we crop it instead of sending it to the text recognizer.
# `formula` = math / equation region (also cropped; recognizer is Khmer/ASCII, not math).
CLASSES = ["background", "title", "section_header", "text", "list_item",
           "caption", "form_label", "form_value", "table_cell", "image", "formula"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
# region types we crop (no text recognition) at inference time
NONTEXT_CLASSES = {"image", "formula"}
# block_type values that don't map 1:1 to a class
_BLOCK_TYPE_ALIAS = {"byline": "caption", "form": "form_value", "list": "list_item",
                     "table": "table_cell", "figure": "image", "photo": "image",
                     "logo": "image", "stamp": "image", "signature": "image"}


def _class_id(block_type: str | None) -> int:
    bt = _BLOCK_TYPE_ALIAS.get(block_type or "", block_type or "text")
    return CLASS_TO_ID.get(bt, CLASS_TO_ID["text"])


def collect_line_boxes(label: dict[str, Any], with_class: bool = False):
    """Union of every block's line rectangles (image-pixel space).

    with_class=False -> list[Box] (back-compat). with_class=True -> list[(Box, class_id)].
    """
    out = []
    for blk in label.get("blocks", []):
        cid = _class_id(blk.get("block_type"))
        for ln in (blk.get("lines") or []):
            if len(ln) == 4:
                x1, y1, x2, y2 = ln
                if x2 > x1 and y2 > y1:
                    box = (float(x1), float(y1), float(x2), float(y2))
                    out.append((box, cid) if with_class else box)
    return out


def _rect_offset(box: Box, shrink_ratio: float) -> float:
    """DBNet polygon offset distance for an axis-aligned rectangle."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    area = w * h
    perim = 2.0 * (w + h)
    if perim <= 0:
        return 0.0
    return area * (1.0 - shrink_ratio ** 2) / perim


def make_db_targets(
    boxes,
    height: int,
    width: int,
    shrink_ratio: float = 0.4,
    thresh_min: float = 0.3,
    thresh_max: float = 0.7,
):
    """Rasterize DBNet targets at (height, width). Boxes are in that same pixel space.

    ``boxes`` may be a list of Box (class-agnostic) or list of (Box, class_id). When classes are
    given, also returns a class_map (int per pixel, 0=background) for the multi-class head.
    Returns (prob, prob_mask, thresh, thresh_mask, class_map).
    """
    prob = np.zeros((height, width), dtype=np.float32)
    prob_mask = np.ones((height, width), dtype=np.float32)
    thresh = np.zeros((height, width), dtype=np.float32)
    thresh_mask = np.zeros((height, width), dtype=np.float32)
    class_map = np.zeros((height, width), dtype=np.uint8)   # cv2-fillable; <256 classes

    for item in boxes:
        if isinstance(item[0], (tuple, list)):
            box, cid = item[0], int(item[1])
        else:
            box, cid = item, 0
        d = _rect_offset(box, shrink_ratio)
        x1, y1, x2, y2 = box

        # --- probability target: filled shrunk rectangle ---
        sx1, sy1 = x1 + d, y1 + d
        sx2, sy2 = x2 - d, y2 - d
        if sx2 - sx1 >= 1 and sy2 - sy1 >= 1:
            pt1 = (int(round(sx1)), int(round(sy1)))
            pt2 = (int(round(sx2)), int(round(sy2)))
            cv2.rectangle(prob, pt1, pt2, 1.0, thickness=-1)
            if cid > 0:
                cv2.rectangle(class_map, pt1, pt2, cid, thickness=-1)

        # --- threshold band: work on a padded ROI for speed/correctness ---
        pad = int(np.ceil(d)) + 2
        rx1, ry1 = max(0, int(np.floor(x1)) - pad), max(0, int(np.floor(y1)) - pad)
        rx2, ry2 = min(width, int(np.ceil(x2)) + pad), min(height, int(np.ceil(y2)) + pad)
        if rx2 <= rx1 or ry2 <= ry1 or d < 1:
            continue
        rw, rh = rx2 - rx1, ry2 - ry1

        # filled box mask within ROI
        box_mask = np.zeros((rh, rw), dtype=np.uint8)
        bx1, by1 = int(round(x1)) - rx1, int(round(y1)) - ry1
        bx2, by2 = int(round(x2)) - rx1, int(round(y2)) - ry1
        cv2.rectangle(box_mask, (bx1, by1), (bx2, by2), 1, thickness=-1)

        # distance to the box border = min(dist inside box to outside, dist outside box to box)
        dist_out = cv2.distanceTransform(1 - box_mask, cv2.DIST_L2, 5)   # 0 on box, grows outward
        dist_in = cv2.distanceTransform(box_mask, cv2.DIST_L2, 5)        # 0 outside, grows inward
        border_dist = np.where(box_mask > 0, dist_in, dist_out).astype(np.float32)

        band = border_dist <= d
        # value: thresh_max at the border (dist 0) -> thresh_min at distance d
        val = thresh_min + (thresh_max - thresh_min) * (1.0 - np.clip(border_dist / d, 0, 1))

        roi_thresh = thresh[ry1:ry2, rx1:rx2]
        roi_mask = thresh_mask[ry1:ry2, rx1:rx2]
        upd = band & (val > roi_thresh)
        roi_thresh[upd] = val[upd]
        roi_mask[band] = 1.0

    return prob, prob_mask, thresh, thresh_mask, class_map
