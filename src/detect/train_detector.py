"""Train the DBNet text-line detector.

  python -m src.detect.train_detector --single
  accelerate launch --multi_gpu --num_processes 3 -m src.detect.train_detector --parallel

Clones the scaffolding pattern of src/train/train.py (Accelerator, cosine LR, bf16, per-step
checkpoints, --resume) but for segmentation-style detection. Eval = box precision/recall/F1 via
the existing src/train/eval/metrics.py against ground-truth line boxes.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader

from src.corpus.common import PROJECT_ROOT
from src.detect.data.dataset import DetDataset, gt_boxes_for_eval
from src.detect.infer_detector import detect_page
from src.detect.model.dbnet import build_detector, count_params, db_loss
from src.train.eval.metrics import box_pr

CONFIG_DIR = Path(__file__).parent / "configs"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_config(profile: str) -> dict:
    with open(CONFIG_DIR / f"{profile}.yaml", encoding="utf-8") as f:
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


@torch.no_grad()
def evaluate(model, accelerator, cfg, val_rows, size, max_pages):
    """Box P/R/F1 at IoU 0.5 and 0.7 on the main process."""
    if not accelerator.is_main_process:
        return None
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    device = accelerator.device
    root = PROJECT_ROOT
    agg = {"f1_50": [], "f1_70": [], "iou": [], "recall_50": [], "prec_50": []}
    for row in val_rows[:max_pages]:
        label = json.loads((root / row["label"].replace("\\", "/")).read_text(encoding="utf-8"))
        img = Image.open(root / row["image"].replace("\\", "/"))
        w, h = label["image_width"], label["image_height"]
        pred = detect_page(unwrapped, img, size, device, cfg)
        pred_n = [[max(0, min(999, round(x1 / w * 999))), max(0, min(999, round(y1 / h * 999))),
                   max(0, min(999, round(x2 / w * 999))), max(0, min(999, round(y2 / h * 999)))]
                  for x1, y1, x2, y2 in pred]
        gt_n = gt_boxes_for_eval(label)
        m50 = box_pr([tuple(b) for b in pred_n], [tuple(b) for b in gt_n], 0.5)
        m70 = box_pr([tuple(b) for b in pred_n], [tuple(b) for b in gt_n], 0.7)
        agg["f1_50"].append(m50["f1"]); agg["f1_70"].append(m70["f1"])
        agg["iou"].append(m50["mean_iou"]); agg["recall_50"].append(m50["recall"])
        agg["prec_50"].append(m50["precision"])
    model.train()
    return {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}


def train(cfg: dict) -> None:
    set_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    accelerator = Accelerator(gradient_accumulation_steps=cfg["grad_accum_steps"],
                              mixed_precision=cfg["mixed_precision"])
    size = cfg["image_size"]

    train_ds = DetDataset(PROJECT_ROOT / cfg["manifests"]["train"], size, train=True,
                          shrink_ratio=cfg["shrink_ratio"])
    if cfg.get("_smoke"):
        train_ds.rows = train_ds.rows[: cfg["_smoke"]]
    val_rows = DetDataset(PROJECT_ROOT / cfg["manifests"]["val"], size).rows

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size_per_device"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
                              persistent_workers=cfg["num_workers"] > 0)

    model = build_detector(cfg["profile"])
    accelerator.print(f"[model] DBNet {cfg['profile']} params={count_params(model)/1e6:.1f}M size={size}")
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"],
                                  betas=tuple(opt_cfg["betas"]), weight_decay=opt_cfg["weight_decay"])
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    steps_per_epoch = math.ceil(len(train_loader) / cfg["grad_accum_steps"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = cfg["scheduler"]["warmup_steps"]
    clip = cfg.get("gradient_clip_norm", 0.0)
    accelerator.print(f"[sched] steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup}")

    gstep = 0
    t0 = time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            with accelerator.accumulate(model):
                batch["pixel_values"] = _normalize(batch["pixel_values"])
                out = model(batch["pixel_values"])
                losses = db_loss(out, batch)
                accelerator.backward(losses["loss"])
                if accelerator.sync_gradients and clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), clip)
                if accelerator.sync_gradients:
                    lr = cosine_lr(gstep, opt_cfg["lr"], warmup, total_steps)
                    for g in optimizer.param_groups:
                        g["lr"] = lr
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                gstep += 1
                if gstep % cfg["log_every_steps"] == 0:
                    rate = gstep / max(time.time() - t0, 1e-9)
                    accelerator.print(f"epoch {epoch} step {gstep}/{total_steps} "
                                      f"loss {losses['loss'].item():.4f} "
                                      f"(ls {losses['ls'].item():.3f} lb {losses['lb'].item():.3f} "
                                      f"lt {losses['lt'].item():.3f} lc {losses['lc'].item():.3f}) "
                                      f"lr {lr:.2e} {rate:.2f} st/s")
                if gstep % cfg["eval_every_steps"] == 0:
                    m = evaluate(model, accelerator, cfg, val_rows, size, cfg["eval_max_pages"])
                    if m:
                        accelerator.print(f"  [eval] step {gstep} F1@.5 {m['f1_50']:.3f} "
                                          f"F1@.7 {m['f1_70']:.3f} mIoU {m['iou']:.3f} "
                                          f"P {m['prec_50']:.3f} R {m['recall_50']:.3f}")
                if gstep % cfg["save_every_steps"] == 0:
                    save_ckpt(accelerator, model, gstep, cfg)

    m = evaluate(model, accelerator, cfg, val_rows, size, cfg["eval_max_pages"])
    if m:
        accelerator.print(f"[final] F1@.5 {m['f1_50']:.3f} F1@.7 {m['f1_70']:.3f} mIoU {m['iou']:.3f}")
    save_ckpt(accelerator, model, gstep, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the DBNet text-line detector")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--single", action="store_true")
    g.add_argument("--parallel", action="store_true")
    ap.add_argument("--smoke", type=int, default=0, help="cap train rows for a quick smoke test")
    args = ap.parse_args()
    cfg = load_config("single" if args.single else "parallel")
    if args.smoke:
        cfg["_smoke"] = args.smoke
    train(cfg)


if __name__ == "__main__":
    main()
