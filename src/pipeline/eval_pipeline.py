"""End-to-end two-stage evaluation, directly comparable to the AR baseline.

Runs detector -> recognizer -> assemble over val/test pages and scores the assembled BLOCK output
with src/train/eval/metrics.page_metrics (the same function the AR model's [gen-eval] uses), so the
two-stage and AR numbers are apples-to-apples.

  python -m src.pipeline.eval_pipeline --det-ckpt ... --rec-ckpt ... --manifest data/manifests/val.jsonl --n 300
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.corpus.common import PROJECT_ROOT
from src.pipeline.run_ocr import load_models, run_ocr
from src.train.eval.metrics import page_metrics

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def main() -> None:
    ap = argparse.ArgumentParser(description="end-to-end two-stage eval")
    ap.add_argument("--det-ckpt", type=Path, required=True)
    ap.add_argument("--rec-ckpt", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/manifests/val.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--det-profile", default="single")
    ap.add_argument("--rec-profile", default="single")
    ap.add_argument("--det-size", type=int, default=960)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    det, rec, enc = load_models(args.det_ckpt, args.rec_ckpt, args.det_profile,
                                args.rec_profile, args.device)
    det_cfg = {"image_size": args.det_size, "prob_thresh": 0.3,
               "box_unclip_ratio": 1.5, "min_box_size": 4}

    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8")][: args.n]
    agg: dict[str, list[float]] = {}
    for row in rows:
        label = json.loads((PROJECT_ROOT / row["label"].replace("\\", "/")).read_text(encoding="utf-8"))
        img = Image.open(PROJECT_ROOT / row["image"].replace("\\", "/"))
        res = run_ocr(img, det, rec, enc, det_cfg, args.device, merge_blocks=True)
        gt_blocks = [{"text": (b.get("text") or "").strip(), "bbox_norm": b["bbox_norm"]}
                     for b in label.get("blocks", []) if b.get("bbox_norm")]
        m = page_metrics(res["block_units"], gt_blocks, iou_thresh=args.iou)
        for k, v in m.items():
            if isinstance(v, (int, float)) and v == v:
                agg.setdefault(k, []).append(v)

    print("=" * 60); print(f"END-TO-END (n={len(rows)}, IoU={args.iou})"); print("=" * 60)
    for k in ("precision", "recall", "f1", "mean_iou", "cer", "syllable_er", "matched_blocks"):
        if k in agg:
            print(f"  {k:14} {np.mean(agg[k]):.4f}")


if __name__ == "__main__":
    main()
