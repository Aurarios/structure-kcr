"""Detector inference: page image -> text-line boxes.

Decodes the DBNet probability map into axis-aligned line boxes: threshold -> connected components
-> bounding rect -> unclip (expand the shrunk region back out). Handles the letterbox scale so
output boxes are in ORIGINAL image-pixel coordinates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from pathlib import Path as _P
PROJECT_ROOT = _P(__file__).resolve().parent.parent  # bundle root (inference_v1)
from .det_utils import letterbox_with_scale
from .dbnet import build_detector

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

Box = list[float]


def _nms(boxes: list[tuple[Box, float]], iou_thresh: float = 0.6,
         contain_thresh: float = 0.55) -> list[Box]:
    """Greedy NMS on (box, score) pairs; keeps the higher-confidence box of an overlapping pair.

    Suppresses by BOTH (a) IoU and (b) containment = intersection / smaller-box area. The
    containment test is what removes line-detector duplicates that plain IoU misses: a short
    fragment box sitting inside a long line box has tiny IoU (small inter / large union) so it
    survives IoU-NMS, but its intersection covers most of its own area -> containment catches it.
    """
    boxes = sorted(boxes, key=lambda b: b[1], reverse=True)
    kept: list[Box] = []
    for box, _ in boxes:
        x1, y1, x2, y2 = box
        a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        drop = False
        for k in kept:
            ix1, iy1 = max(x1, k[0]), max(y1, k[1])
            ix2, iy2 = min(x2, k[2]), min(y2, k[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0.0:
                continue
            ka = max(0.0, k[2] - k[0]) * max(0.0, k[3] - k[1])
            iou = inter / (a + ka - inter + 1e-6)
            contain = inter / (min(a, ka) + 1e-6)      # how much of the smaller box is covered
            if iou > iou_thresh or contain > contain_thresh:
                drop = True
                break
        if not drop:
            kept.append(box)
    return kept


def _merge_same_line(boxes: list[Box], y_overlap: float = 0.6) -> list[Box]:
    """Union boxes that belong to the SAME text line but were split into pieces.

    A split shows up as two boxes with high VERTICAL overlap (same line band) that also overlap
    HORIZONTALLY (a leading/trailing fragment poking into the main box). We merge those into one box.
    Two genuinely separate same-row boxes (e.g. table cells / columns) have a horizontal GAP
    (no x-overlap), so they are left alone. Iterated to a fixed point.
    """
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out: list[Box] = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            ax1, ay1, ax2, ay2 = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                bx1, by1, bx2, by2 = boxes[j]
                vov = max(0.0, min(ay2, by2) - max(ay1, by1)) / (min(ay2 - ay1, by2 - by1) + 1e-6)
                hov = max(0.0, min(ax2, bx2) - max(ax1, bx1))   # absolute horizontal overlap (px)
                # A same-line split = the two pieces occupy (almost) the SAME vertical band
                # (vov >= y_overlap, default 0.8) and touch horizontally. Genuinely separate stacked
                # lines have low vertical overlap (they only graze by a few px) -> NOT merged.
                if vov >= y_overlap and hov > 0:
                    ax1, ay1 = min(ax1, bx1), min(ay1, by1)
                    ax2, ay2 = max(ax2, bx2), max(ay2, by2)
                    used[j] = True
                    changed = True
            used[i] = True
            out.append([ax1, ay1, ax2, ay2])
        boxes = out
    return boxes


def decode_boxes(prob_map: np.ndarray, prob_thresh: float = 0.3,
                 unclip_ratio: float = 2.0, min_size: int = 4,
                 edge_margin_frac: float = 0.18, min_confidence: float = 0.5,
                 min_area: int = 40) -> list[Box]:
    """prob_map (H,W) float in [0,1] -> list of [x1,y1,x2,y2] in prob-map pixel space.

    DBNet detects the SHRUNK text region; we expand it back to the true text extent. Because the
    training shrink offset was computed on the (larger) original box but the unclip distance is
    computed on the (smaller) detected box, a plain unclip under-recovers the width of thin lines
    and clips the first/last glyph. Fix: (1) a larger unclip_ratio, and (2) an explicit HORIZONTAL
    margin proportional to line height to guarantee edge glyphs are included (lines have no
    horizontal neighbours, so over-including a little background is harmless). Vertical expansion is
    kept to the unclip only, so boxes don't bleed into the closely-spaced line above/below.

    Robustness filters (matter on real, tight-spaced docs): drop components whose MEAN probability is
    low (spurious 'empty' boxes), drop tiny fragments (min_area), and NMS to remove overlaps.
    """
    H_map, W_map = prob_map.shape
    binmap = (prob_map >= prob_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    scored: list[tuple[Box, float]] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_size or h < min_size or w * h < min_area:
            continue
        # confidence = mean prob inside the filled component (kills marginal/empty activations)
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [c - [x, y]], -1, 1, thickness=-1)
        region = prob_map[y:y + h, x:x + w]
        conf = float(region[mask.astype(bool)].mean()) if mask.any() else 0.0
        if conf < min_confidence:
            continue
        area = w * h
        perim = 2.0 * (w + h)
        d = area * unclip_ratio / perim if perim > 0 else 0.0
        mh = max(3.0, h * edge_margin_frac)        # horizontal margin (recover edge glyphs)
        x1, y1 = x - d - mh, y - d
        x2, y2 = x + w + d + mh, y + h + d
        box = [max(0.0, x1), max(0.0, y1), min(W_map, x2), min(H_map, y2)]
        scored.append((box, conf))
    kept = _nms(scored, iou_thresh=0.6)
    # merge same-line splits (leading/trailing fragments) into one box; leaves separate
    # same-row boxes (columns/cells, which have a horizontal gap) untouched.
    return _merge_same_line(kept)


@torch.no_grad()
def detect_page(model, pil_img: Image.Image, size: int, device: str, cfg: dict,
                return_class: bool = False):
    """Returns line boxes in ORIGINAL image-pixel coordinates.

    return_class=False -> list[Box]. return_class=True -> list[(Box, class_id)], where class_id is
    the majority region-type predicted inside each box (for structure-aware assembly).
    """
    arr, scale = letterbox_with_scale(pil_img, size)
    px = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    px = ((px - IMAGENET_MEAN) / IMAGENET_STD).to(device, dtype=next(model.parameters()).dtype)
    out_m = model(px)
    prob = out_m["prob"][0, 0].float().cpu().numpy()
    class_argmax = None
    if return_class and "class_logits" in out_m:
        class_argmax = out_m["class_logits"][0].argmax(0).cpu().numpy()   # (size,size) int
    boxes = decode_boxes(prob, cfg.get("prob_thresh", 0.3),
                         cfg.get("box_unclip_ratio", 2.0), cfg.get("min_box_size", 4),
                         cfg.get("edge_margin_frac", 0.18), cfg.get("min_confidence", 0.5),
                         cfg.get("min_area", 40))
    w0, h0 = pil_img.size
    out = []
    for x1, y1, x2, y2 in boxes:
        ob = [max(0.0, min(w0, x1 / scale)), max(0.0, min(h0, y1 / scale)),
              max(0.0, min(w0, x2 / scale)), max(0.0, min(h0, y2 / scale))]
        if return_class:
            cid = 0
            if class_argmax is not None:
                rx1, ry1 = int(max(0, x1)), int(max(0, y1))
                rx2, ry2 = int(min(size, x2)), int(min(size, y2))
                if rx2 > rx1 and ry2 > ry1:
                    region = class_argmax[ry1:ry2, rx1:rx2].ravel()
                    region = region[region > 0]                  # ignore background pixels
                    if region.size:
                        cid = int(np.bincount(region).argmax())
            out.append((ob, cid))
        else:
            out.append(ob)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="DBNet detector inference")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--profile", default="single")
    ap.add_argument("--size", type=int, default=960, help="inference resolution; raise for dense/tall pages")
    ap.add_argument("--prob-thresh", type=float, default=0.3, help="lower -> higher recall (catches faint lines)")
    ap.add_argument("--unclip", type=float, default=2.0, help="box expansion ratio")
    ap.add_argument("--min-box-size", type=int, default=4)
    ap.add_argument("--min-confidence", type=float, default=0.5, help="drop low-mean-prob (empty) boxes")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None, help="optional: save image with boxes drawn")
    args = ap.parse_args()

    model = build_detector(args.profile)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval().to(args.device)

    img = Image.open(args.image)
    cfg = {"prob_thresh": args.prob_thresh, "box_unclip_ratio": args.unclip,
           "min_box_size": args.min_box_size, "min_confidence": args.min_confidence}
    boxes = detect_page(model, img, args.size, args.device, cfg)
    print(f"detected {len(boxes)} line boxes")
    for b in boxes[:20]:
        print("  ", [round(v, 1) for v in b])

    if args.out:
        im = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.imwrite(str(args.out), im)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
