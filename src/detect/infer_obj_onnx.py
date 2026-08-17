"""ONNX-Runtime CPU inference for the V4 FCOS detector — drop-in for detect_page_obj.

Loads an ONNX model exported by src.optimize.export_onnx (cls/reg/ctr concatenated P3..P7 for a
FIXED letterboxed size), precomputes the per-location centers + strides once, and decodes + NMSes
in numpy. Same return contract as infer_obj.detect_page_obj: [(box[x1,y1,x2,y2] orig-px, cls, score)].
"""
from __future__ import annotations

import numpy as np
import onnxruntime as ort
from PIL import Image

from src.detect.data.dataset import letterbox_with_scale
from src.detect.model.fcos import STRIDES
from src.detect.infer_obj import _nms

_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)


def _locations(size: int):
    """Concatenated (N,2) centers and (N,) stride per location, P3..P7, matching the export order."""
    locs, strides = [], []
    for s in STRIDES:
        h = w = size // s
        ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        loc = np.stack([(xs.reshape(-1) + 0.5) * s, (ys.reshape(-1) + 0.5) * s], 1).astype(np.float32)
        locs.append(loc)
        strides.append(np.full(h * w, s, np.float32))
    return np.concatenate(locs, 0), np.concatenate(strides, 0)


class OnnxObjDetector:
    def __init__(self, onnx_path: str, size: int, num_classes: int, threads: int = 0):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.size = size
        self.num_classes = num_classes
        self.locs, self.strides = _locations(size)

    def __call__(self, pil_img: Image.Image, cfg: dict):
        score_thresh = cfg.get("score_thresh", 0.3)
        nms_iou = cfg.get("nms_iou", 0.5)
        max_det = cfg.get("max_det", 300)

        arr, scale = letterbox_with_scale(pil_img, self.size)
        px = arr.transpose(2, 0, 1).astype(np.float32) / 255.0
        px = ((px - _MEAN) / _STD)[None]
        cls, reg, ctr = self.sess.run(None, {"image": px})
        cls, reg, ctr = cls[0], reg[0], ctr[0]               # (N,C),(N,4),(N,)

        score = cls * ctr[:, None]
        cl = score.argmax(1)
        sc = score[np.arange(len(cl)), cl]
        keep = sc > score_thresh
        if not keep.any():
            return []
        loc = self.locs[keep]; r4 = reg[keep] * self.strides[keep, None]
        x, y = loc[:, 0], loc[:, 1]
        boxes = np.stack([x - r4[:, 0], y - r4[:, 1], x + r4[:, 2], y + r4[:, 3]], 1)
        sc, cl = sc[keep], cl[keep]

        w0, h0 = pil_img.size
        results = []
        for c in np.unique(cl):
            m = cl == c
            kb, ks = boxes[m], sc[m]
            for i in _nms(kb, ks, nms_iou):
                b = kb[i] / scale
                x1 = float(max(0, min(w0, b[0]))); y1 = float(max(0, min(h0, b[1])))
                x2 = float(max(0, min(w0, b[2]))); y2 = float(max(0, min(h0, b[3])))
                if x2 - x1 >= 2 and y2 - y1 >= 2:
                    results.append(([x1, y1, x2, y2], int(c), float(ks[i])))
        results.sort(key=lambda r: -r[2])
        return results[:max_det]
