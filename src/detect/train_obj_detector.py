"""Train the V4 FCOS object-detection layout detector.

  python -m src.detect.train_obj_detector --single --config obj_single_v4
  python -m src.detect.train_obj_detector --single --config obj_single_v4 --smoke 50   # overfit gate

Clones src/detect/train_detector.py scaffolding (Accelerator, cosine LR, bf16, per-step ckpts) but
with the FCOS loss and an object-detection dataset. Eval = per-class box F1@0.5 + mean (proxy mAP).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader

from src.corpus.common import PROJECT_ROOT
from src.detect.data.obj_dataset import ObjDetDataset, collate_fn
from src.detect.data.build_obj_targets import (collect_obj_boxes, CLASSES_V4, CLASSES_V6,
                                               NUM_CLASSES_V4, NUM_LAYOUTS)
from src.detect.infer_obj import detect_page_obj
from src.detect.model.fcos import build_obj_detector, count_params, fcos_loss
from src.train.eval.metrics import box_pr

CONFIG_DIR = Path(__file__).parent / "configs"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_config(name: str) -> dict:
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine_lr(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _normalize(pixels):
    return (pixels - IMAGENET_MEAN.to(pixels.device, pixels.dtype)) / IMAGENET_STD.to(pixels.device, pixels.dtype)


def save_ckpt(accelerator, model, step, cfg):
    if not accelerator.is_main_process:
        return
    d = PROJECT_ROOT / cfg["ckpt_dir"] / f"step_{step:08d}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save({"model": accelerator.unwrap_model(model).state_dict(), "step": step}, d / "model.pt")
    keep = int(cfg.get("keep_last_n_ckpts", 5))
    sibs = sorted((PROJECT_ROOT / cfg["ckpt_dir"]).glob("step_*"),
                  key=lambda p: int(p.name.split("_")[1]))
    for old in sibs[:-keep]:
        for f in old.rglob("*"):
            if f.is_file():
                f.unlink()
        old.rmdir()
    accelerator.print(f"[ckpt] saved -> {d}")


def _norm(b, w, h):
    return (max(0, min(999, round(b[0] / w * 999))), max(0, min(999, round(b[1] / h * 999))),
            max(0, min(999, round(b[2] / w * 999))), max(0, min(999, round(b[3] / h * 999))))


@torch.no_grad()
def evaluate(model, accelerator, cfg, val_rows, size, max_pages, taxonomy: str = "v4"):
    """Per-class box F1@0.5 + mean (proxy mAP) on the main process."""
    if not accelerator.is_main_process:
        return None
    classes = CLASSES_V6 if taxonomy == "v6" else CLASSES_V4
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    device = accelerator.device
    by_cls_f1 = defaultdict(list)
    for row in val_rows[:max_pages]:
        label = json.loads((PROJECT_ROOT / row["label"].replace("\\", "/")).read_text(encoding="utf-8"))
        img = Image.open(PROJECT_ROOT / row["image"].replace("\\", "/"))
        w, h = label["image_width"], label["image_height"]
        dets = detect_page_obj(unwrapped, img, size, device, cfg)
        pred_by = defaultdict(list); gt_by = defaultdict(list)
        for box, cid, _ in dets:
            pred_by[cid].append(_norm(box, w, h))
        for (x1, y1, x2, y2), cid in collect_obj_boxes(label, taxonomy=taxonomy):
            gt_by[cid].append(_norm((x1, y1, x2, y2), w, h))
        for cid in set(pred_by) | set(gt_by):
            m = box_pr(pred_by.get(cid, []), gt_by.get(cid, []), 0.5)
            by_cls_f1[cid].append(m["f1"])
    model.train()
    per_cls = {classes[c]: float(np.mean(v)) for c, v in by_cls_f1.items() if v}
    mean_f1 = float(np.mean(list(per_cls.values()))) if per_cls else 0.0
    return {"mAP_f1": mean_f1, "per_class": per_cls}


def train(cfg: dict) -> None:
    set_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    accelerator = Accelerator(gradient_accumulation_steps=cfg["grad_accum_steps"],
                              mixed_precision=cfg["mixed_precision"])
    size = cfg["image_size"]

    taxonomy = cfg.get("taxonomy", "v4")
    classes = CLASSES_V6 if taxonomy == "v6" else CLASSES_V4
    n_classes = len(classes)
    n_layouts = NUM_LAYOUTS if cfg.get("layout_head", False) else 0

    train_ds = ObjDetDataset(PROJECT_ROOT / cfg["manifests"]["train"], size, train=True,
                             taxonomy=taxonomy, with_layout=bool(n_layouts))
    if cfg.get("_smoke"):
        train_ds.rows = train_ds.rows[: cfg["_smoke"]]
    val_rows = ObjDetDataset(PROJECT_ROOT / cfg["manifests"]["val"], size,
                             taxonomy=taxonomy).rows
    if cfg.get("_smoke"):                                  # overfit gate: eval on the train pages
        val_rows = train_ds.rows

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size_per_device"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
                              collate_fn=collate_fn, persistent_workers=cfg["num_workers"] > 0)

    model = build_obj_detector(cfg["profile"], num_classes=n_classes, num_layouts=n_layouts)
    accelerator.print(f"[model] FCOS {cfg['profile']} params={count_params(model)/1e6:.1f}M "
                      f"classes={n_classes} ({taxonomy}) size={size} "
                      f"layout_head={'on' if n_layouts else 'off'}")
    o = cfg["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=o["lr"], betas=tuple(o["betas"]),
                                  weight_decay=o["weight_decay"])
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    steps_per_epoch = math.ceil(len(train_loader) / cfg["grad_accum_steps"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = cfg["scheduler"]["warmup_steps"]
    clip = cfg.get("gradient_clip_norm", 0.0)
    accelerator.print(f"[sched] steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup}")

    gstep = 0; t0 = time.time(); model.train()
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            with accelerator.accumulate(model):
                px = _normalize(batch["pixel_values"])
                out = model(px)
                losses = fcos_loss(out, batch["targets"], num_classes=n_classes,
                                   layout_weight=cfg.get("layout_loss_weight", 0.1))
                accelerator.backward(losses["loss"])
                if accelerator.sync_gradients and clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), clip)
                if accelerator.sync_gradients:
                    lr = cosine_lr(gstep, o["lr"], warmup, total_steps)
                    for g in optimizer.param_groups:
                        g["lr"] = lr
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                gstep += 1
                if gstep % cfg["log_every_steps"] == 0:
                    rate = gstep / max(time.time() - t0, 1e-9)
                    accelerator.print(f"epoch {epoch} step {gstep}/{total_steps} "
                                      f"loss {losses['loss'].item():.4f} (lc {losses['lc'].item():.3f} "
                                      f"lr {losses['lr'].item():.3f} lctr {losses['lctr'].item():.3f}"
                                      + (f" llay {losses['llay'].item():.3f}" if n_layouts else "")
                                      + f") npos {losses['n_pos']} lr {lr:.2e} {rate:.2f} st/s")
                if gstep % cfg["eval_every_steps"] == 0:
                    m = evaluate(model, accelerator, cfg, val_rows, size, cfg["eval_max_pages"], taxonomy)
                    if m:
                        top = sorted(m["per_class"].items(), key=lambda x: -x[1])[:6]
                        accelerator.print(f"  [eval] step {gstep} meanF1@.5 {m['mAP_f1']:.3f} | " +
                                          " ".join(f"{k}={v:.2f}" for k, v in top))
                if gstep % cfg["save_every_steps"] == 0:
                    save_ckpt(accelerator, model, gstep, cfg)

    m = evaluate(model, accelerator, cfg, val_rows, size, cfg["eval_max_pages"], taxonomy)
    if m:
        accelerator.print(f"[final] meanF1@.5 {m['mAP_f1']:.3f} per_class={m['per_class']}")
    save_ckpt(accelerator, model, gstep, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the V4 FCOS layout detector")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--single", action="store_true")
    g.add_argument("--parallel", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="cap train rows (overfit gate)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config or ("obj_single_v4" if args.single else "obj_parallel_v4"))
    if args.smoke:
        cfg["_smoke"] = args.smoke
    train(cfg)


if __name__ == "__main__":
    main()
