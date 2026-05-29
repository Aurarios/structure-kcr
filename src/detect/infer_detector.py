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

from src.corpus.common import PROJECT_ROOT
from src.detect.data.dataset import letterbox_with_scale
from src.detect.model.dbnet import build_detector

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

Box = list[float]


def decode_boxes(prob_map: np.ndarray, prob_thresh: float = 0.3,
                 unclip_ratio: float = 2.0, min_size: int = 4,
                 edge_margin_frac: float = 0.18) -> list[Box]:
    """prob_map (H,W) float in [0,1] -> list of [x1,y1,x2,y2] in prob-map pixel space.

    DBNet detects the SHRUNK text region; we expand it back to the true text extent. Because the
    training shrink offset was computed on the (larger) original box but the unclip distance is
    computed on the (smaller) detected box, a plain unclip under-recovers the width of thin lines
    and clips the first/last glyph. Fix: (1) a larger unclip_ratio, and (2) an explicit HORIZONTAL
    margin proportional to line height to guarantee edge glyphs are included (lines have no
    horizontal neighbours, so over-including a little background is harmless). Vertical expansion is
    kept to the unclip only, so boxes don't bleed into the closely-spaced line above/below.
    """
    H_map, W_map = prob_map.shape
    binmap = (prob_map >= prob_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[Box] = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w < min_size or h < min_size:
            continue
        area = w * h
        perim = 2.0 * (w + h)
        d = area * unclip_ratio / perim if perim > 0 else 0.0
        mh = max(3.0, h * edge_margin_frac)        # horizontal safety margin (recover edge glyphs)
        x1, y1 = x - d - mh, y - d
        x2, y2 = x + w + d + mh, y + h + d
        boxes.append([max(0.0, x1), max(0.0, y1), min(W_map, x2), min(H_map, y2)])
    return boxes


@torch.no_grad()
def detect_page(model, pil_img: Image.Image, size: int, device: str, cfg: dict) -> list[Box]:
    """Returns line boxes in ORIGINAL image-pixel coordinates."""
    arr, scale = letterbox_with_scale(pil_img, size)
    px = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    px = ((px - IMAGENET_MEAN) / IMAGENET_STD).to(device, dtype=next(model.parameters()).dtype)
    prob = model(px)["prob"][0, 0].float().cpu().numpy()
    boxes = decode_boxes(prob, cfg.get("prob_thresh", 0.3),
                         cfg.get("box_unclip_ratio", 2.0), cfg.get("min_box_size", 4),
                         cfg.get("edge_margin_frac", 0.18))
    w0, h0 = pil_img.size
    out: list[Box] = []
    for x1, y1, x2, y2 in boxes:
        # map from letterboxed square back to original pixels (content placed at top-left, scale s)
        out.append([
            max(0.0, min(w0, x1 / scale)), max(0.0, min(h0, y1 / scale)),
            max(0.0, min(w0, x2 / scale)), max(0.0, min(h0, y2 / scale)),
        ])
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
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None, help="optional: save image with boxes drawn")
    args = ap.parse_args()

    model = build_detector(args.profile)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval().to(args.device)

    img = Image.open(args.image)
    cfg = {"prob_thresh": args.prob_thresh, "box_unclip_ratio": args.unclip,
           "min_box_size": args.min_box_size}
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
