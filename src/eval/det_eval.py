"""Standalone V4 detector evaluation: tightness-aware, per-layout, with threshold sweep.

Goes beyond the in-training eval (train_obj_detector.evaluate = per-page-averaged F1@0.5) by
accumulating TP/FP/FN globally (micro) per class, scoring at BOTH IoU 0.5 and 0.75, and reporting
the mean IoU of matched boxes — the box-tightness signal that F1@0.5 alone hides. Rows are shuffled
(fixed seed) before capping so per-layout manifests don't skew the sample.

  python -m src.eval.det_eval --ckpt data/checkpoints/detector_obj_v4_line/step_00007132 \
      --manifest data/manifests_v3/val.jsonl --profile single --size 1024
  ... --sweep            # per-class score-threshold sweep -> best-F1@0.5 threshold per class
  ... --out eval.json    # full metrics as JSON (for version-to-version diffing)
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.corpus.common import PROJECT_ROOT
from src.detect.data.build_obj_targets import CLASSES_V4, collect_obj_boxes
from src.detect.infer_obj import detect_page_obj
from src.detect.model.fcos import build_obj_detector
from src.train.eval.metrics import box_pr

SWEEP_THRESHOLDS = [round(0.05 + 0.05 * i, 2) for i in range(11)]   # 0.05 .. 0.55


def _load_rows(manifest: Path, max_pages: int, seed: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    random.Random(seed).shuffle(rows)        # per-layout manifests are grouped; cap fairly
    return rows[:max_pages] if max_pages else rows


def _resolve(p: str) -> Path:
    return PROJECT_ROOT / p.replace("\\", "/")   # absolute paths (E:/...) win over PROJECT_ROOT


def _load_model(ckpt: Path, profile: str, device: str):
    if ckpt.is_dir():
        ckpt = ckpt / "model.pt"
    state = torch.load(ckpt, map_location="cpu")
    model = build_obj_detector(profile)
    model.load_state_dict(state.get("model", state))
    model.eval().to(device)
    return model


class _Acc:
    """Micro accumulator for one (class) or (layout) bucket at one IoU threshold."""

    def __init__(self):
        self.tp = self.fp = self.fn = 0.0
        self.iou_sum = 0.0

    def add(self, m: dict):
        self.tp += m["tp"]; self.fp += m["fp"]; self.fn += m["fn"]
        self.iou_sum += m["mean_iou"] * m["tp"]

    def metrics(self) -> dict:
        p = self.tp / max(1.0, self.tp + self.fp)
        r = self.tp / max(1.0, self.tp + self.fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                "mean_iou": round(self.iou_sum / max(1.0, self.tp), 4),
                "tp": int(self.tp), "fp": int(self.fp), "fn": int(self.fn)}


@torch.no_grad()
def evaluate_detector(model, rows, size: int, device: str, det_cfg: dict,
                      iou_thrs=(0.5, 0.75), sweep: bool = False, quiet: bool = False) -> dict:
    by_cls = {t: defaultdict(_Acc) for t in iou_thrs}           # iou_thr -> cid -> acc
    by_layout = defaultdict(_Acc)                                # layout -> acc (@ first thr)
    sweep_acc = {t: defaultdict(_Acc) for t in SWEEP_THRESHOLDS} if sweep else None
    run_thr = min(SWEEP_THRESHOLDS) if sweep else det_cfg.get("score_thresh", 0.3)
    cfg = dict(det_cfg, score_thresh=run_thr)

    n_done, t0 = 0, time.time()
    for row in rows:
        label_p, image_p = _resolve(row["label"]), _resolve(row["image"])
        if not label_p.exists() or not image_p.exists():
            continue
        label = json.loads(label_p.read_text(encoding="utf-8"))
        dets = detect_page_obj(model, Image.open(image_p), size, device, cfg)

        gt_by = defaultdict(list)
        for (x1, y1, x2, y2), cid in collect_obj_boxes(label):
            gt_by[cid].append((x1, y1, x2, y2))
        pred_by = defaultdict(list)                              # cid -> [(box, score)]
        for box, cid, sc in dets:
            pred_by[cid].append((tuple(box), sc))

        for cid in set(pred_by) | set(gt_by):
            gts = gt_by.get(cid, [])
            main_preds = [b for b, s in pred_by.get(cid, [])
                          if s >= det_cfg.get("score_thresh", 0.3)]
            for t in iou_thrs:
                by_cls[t][cid].add(box_pr(main_preds, gts, t))
            by_layout[row.get("layout_type", "?")].add(box_pr(main_preds, gts, iou_thrs[0]))
            if sweep:
                for t in SWEEP_THRESHOLDS:
                    preds_t = [b for b, s in pred_by.get(cid, []) if s >= t]
                    sweep_acc[t][cid].add(box_pr(preds_t, gts, 0.5))

        n_done += 1
        if not quiet and n_done % 100 == 0:
            print(f"  [{n_done}/{len(rows)}] {n_done / (time.time() - t0):.1f} pages/s")

    out = {"pages": n_done, "per_class": {}, "per_layout": {}, "sweep": None}
    for cid in sorted(set().union(*[set(a) for a in by_cls.values()])):
        name = CLASSES_V4[cid]
        out["per_class"][name] = {f"@{t}": by_cls[t][cid].metrics() for t in iou_thrs}
    for lay, acc in sorted(by_layout.items()):
        out["per_layout"][lay] = acc.metrics()
    f1s = {t: [out["per_class"][n][f"@{t}"]["f1"] for n in out["per_class"]] for t in iou_thrs}
    out["summary"] = {f"meanF1@{t}": round(float(np.mean(f1s[t])), 4) for t in iou_thrs}
    ious = [out["per_class"][n][f"@{iou_thrs[0]}"]["mean_iou"] for n in out["per_class"]
            if out["per_class"][n][f"@{iou_thrs[0]}"]["tp"] > 0]
    out["summary"]["mean_matched_iou"] = round(float(np.mean(ious)), 4) if ious else 0.0

    if sweep:
        best = {}
        for cid in sorted(set().union(*[set(a) for a in sweep_acc.values()])):
            curve = {t: sweep_acc[t][cid].metrics()["f1"] for t in SWEEP_THRESHOLDS}
            bt = max(curve, key=curve.get)
            best[CLASSES_V4[cid]] = {"best_thresh": bt, "best_f1": round(curve[bt], 4),
                                     "curve": curve}
        out["sweep"] = best
    return out


def print_report(m: dict) -> None:
    print(f"\n== detector eval ({m['pages']} pages) ==")
    print(f"  {'class':14s} {'F1@.5':>6s} {'P@.5':>6s} {'R@.5':>6s} {'F1@.75':>7s} {'mIoU':>6s} {'gt':>6s}")
    for name, d in sorted(m["per_class"].items(), key=lambda kv: -kv[1]["@0.5"]["f1"]):
        a, b = d["@0.5"], d.get("@0.75", {})
        gt = a["tp"] + a["fn"]
        print(f"  {name:14s} {a['f1']:6.3f} {a['precision']:6.3f} {a['recall']:6.3f} "
              f"{b.get('f1', 0):7.3f} {a['mean_iou']:6.3f} {gt:6d}")
    s = m["summary"]
    print(f"  {'MEAN':14s} {s['meanF1@0.5']:6.3f} {'':6s} {'':6s} {s.get('meanF1@0.75', 0):7.3f} "
          f"{s['mean_matched_iou']:6.3f}")
    print("\n  -- per layout (micro F1@0.5) --")
    for lay, d in sorted(m["per_layout"].items(), key=lambda kv: kv[1]["f1"]):
        print(f"  {lay:22s} f1 {d['f1']:.3f}  p {d['precision']:.3f}  r {d['recall']:.3f}  "
              f"miou {d['mean_iou']:.3f}")
    if m.get("sweep"):
        print("\n  -- per-class best score threshold (F1@0.5) --")
        for name, d in sorted(m["sweep"].items()):
            print(f"  {name:14s} thresh {d['best_thresh']:.2f}  f1 {d['best_f1']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="V4 detector standalone eval (F1@.5/.75 + IoU tightness)")
    ap.add_argument("--ckpt", type=Path, required=True, help="step dir or model.pt")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--profile", default="single", choices=["single", "parallel"])
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--max-pages", type=int, default=400)
    ap.add_argument("--score-thresh", type=float, default=0.3)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--sweep", action="store_true", help="per-class score-threshold sweep")
    ap.add_argument("--out", type=Path, default=None, help="write full metrics JSON here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = _load_model(args.ckpt, args.profile, args.device)
    rows = _load_rows(args.manifest, args.max_pages)
    print(f"[det_eval] ckpt={args.ckpt} profile={args.profile} size={args.size} "
          f"pages={len(rows)} device={args.device} sweep={args.sweep}")
    det_cfg = {"score_thresh": args.score_thresh, "nms_iou": args.nms_iou, "agnostic_nms": True}
    m = evaluate_detector(model, rows, args.size, args.device, det_cfg, sweep=args.sweep)
    m["config"] = {"ckpt": str(args.ckpt), "manifest": str(args.manifest), "size": args.size,
                   "profile": args.profile, "score_thresh": args.score_thresh,
                   "nms_iou": args.nms_iou, "max_pages": args.max_pages}
    print_report(m)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(m, indent=1), encoding="utf-8")
        print(f"\n[det_eval] wrote {args.out}")


if __name__ == "__main__":
    main()
