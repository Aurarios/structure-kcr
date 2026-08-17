"""V4 object-detector inference: page image -> [(box, class_id, score)] in original image pixels.

Decodes FCOS per-level cls*centerness scores + l,t,r,b distance regression into boxes, applies
per-class greedy NMS (hand-rolled; the project avoids torchvision), and maps boxes back through the
letterbox to original-image coordinates.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.detect.data.dataset import letterbox_with_scale
from src.detect.model.fcos import STRIDES, build_obj_detector, locations_for
from src.detect.data.build_obj_targets import CLASSES_V4, NUM_CLASSES_V4

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> list[int]:
    """Greedy NMS on a single class. boxes [N,4], scores [N]."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest]); iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest]); iy2 = np.minimum(y2[i], y2[rest])
        iw = (ix2 - ix1).clip(0); ih = (iy2 - iy1).clip(0)
        inter = iw * ih
        iou = inter / (area[i] + area[rest] - inter + 1e-7)
        order = rest[iou <= iou_thresh]
    return keep


def _agnostic_nms(results: list, iou_thresh: float) -> list:
    """Class-agnostic NMS: drop overlapping boxes regardless of class, keep highest score.
    Per-class NMS leaves duplicates when one line fires as two classes (e.g. heading+subheading,
    text+list_item) — this collapses them to one box."""
    if len(results) <= 1:
        return results
    boxes = np.array([r[0] for r in results], dtype=np.float32)
    scores = np.array([r[2] for r in results], dtype=np.float32)
    return [results[i] for i in _nms(boxes, scores, iou_thresh)]


def _containment_suppress(results: list, frac: float) -> list:
    """Drop a box if >frac of its area sits inside a higher-scoring box (kills nested detections,
    e.g. a line box inside a block box). Off by default — line-level detection rarely nests."""
    out: list = []
    for r in sorted(results, key=lambda r: -r[2]):
        x1, y1, x2, y2 = r[0]
        area = max(1.0, (x2 - x1) * (y2 - y1))
        drop = False
        for k in out:
            kx1, ky1, kx2, ky2 = k[0]
            inter = max(0.0, min(x2, kx2) - max(x1, kx1)) * max(0.0, min(y2, ky2) - max(y1, ky1))
            if inter / area > frac:
                drop = True
                break
        if not drop:
            out.append(r)
    return out


@torch.no_grad()
def detect_page_obj(model, pil_img: Image.Image, size: int, device: str, cfg: dict):
    """Returns list of (box[x1,y1,x2,y2] original px, class_id, score)."""
    score_thresh = cfg.get("score_thresh", 0.3)
    nms_iou = cfg.get("nms_iou", 0.5)
    max_det = cfg.get("max_det", 300)
    num_classes = getattr(model, "num_classes", NUM_CLASSES_V4)
    # optional per-class thresholds (from det_eval --sweep), e.g. {"image": 0.45, "table_cell": 0.55}
    thr_t = None
    per_cls = cfg.get("score_thresh_per_class") or {}
    if per_cls:
        thr_t = torch.full((num_classes,), float(score_thresh))
        for name, t in per_cls.items():
            if name in CLASSES_V4:
                thr_t[CLASSES_V4.index(name)] = float(t)

    arr, scale = letterbox_with_scale(pil_img, size)
    px = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    px = ((px - IMAGENET_MEAN) / IMAGENET_STD).to(device, dtype=next(model.parameters()).dtype)
    out = model(px)
    locs = locations_for(out["sizes"], STRIDES, device)

    boxes_l, scores_l, cls_l = [], [], []
    for cls, reg, ctr, loc, stride in zip(out["cls"], out["reg"], out["ctr"], locs, STRIDES):
        cls = cls[0].permute(1, 2, 0).reshape(-1, num_classes).float().sigmoid()
        ctr = ctr[0].reshape(-1).float().sigmoid()
        reg = reg[0].permute(1, 2, 0).reshape(-1, 4).float()
        score = cls * ctr[:, None]
        sc, cl = score.max(1)
        keep = sc > (thr_t.to(sc.device)[cl] if thr_t is not None else score_thresh)
        if keep.any():
            ll = loc[keep]; r4 = reg[keep] * stride
            x, y = ll[:, 0], ll[:, 1]
            bx = torch.stack([x - r4[:, 0], y - r4[:, 1], x + r4[:, 2], y + r4[:, 3]], 1)
            boxes_l.append(bx.cpu()); scores_l.append(sc[keep].cpu()); cls_l.append(cl[keep].cpu())
    if not boxes_l:
        return []
    boxes = torch.cat(boxes_l).numpy()
    scores = torch.cat(scores_l).numpy()
    clss = torch.cat(cls_l).numpy()

    w0, h0 = pil_img.size
    results = []
    for c in np.unique(clss):
        m = clss == c
        kb, ks = boxes[m], scores[m]
        for i in _nms(kb, ks, nms_iou):
            b = kb[i] / scale
            x1 = float(max(0, min(w0, b[0]))); y1 = float(max(0, min(h0, b[1])))
            x2 = float(max(0, min(w0, b[2]))); y2 = float(max(0, min(h0, b[3])))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                results.append(([x1, y1, x2, y2], int(c), float(ks[i])))
    results.sort(key=lambda r: -r[2])
    if cfg.get("agnostic_nms", True):           # collapse cross-class duplicate boxes (default on)
        results = _agnostic_nms(results, nms_iou)
    if cfg.get("containment", False):           # drop nested boxes (default off; opt-in)
        results = _containment_suppress(results, cfg.get("contain_frac", 0.7))
    return results[:max_det]


def main() -> None:
    ap = argparse.ArgumentParser(description="V4 FCOS layout detector inference")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--profile", default="single")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    model = build_obj_detector(args.profile)
    model.load_state_dict(state.get("model", state))
    model.eval().to(args.device)

    img = Image.open(args.image)
    dets = detect_page_obj(model, img, args.size, args.device, {"score_thresh": args.score_thresh})
    print(f"detected {len(dets)} regions")
    for box, cid, sc in dets[:40]:
        print(f"  {CLASSES_V4[cid]:12s} {sc:.2f} {[round(v,1) for v in box]}")
    if args.out:
        import cv2
        im = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        for box, cid, sc in dets:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(im, CLASSES_V4[cid], (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)
        cv2.imwrite(str(args.out), im)
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
