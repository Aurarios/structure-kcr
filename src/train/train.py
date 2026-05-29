"""From-scratch Khmer OCR trainer.

  # single GPU (4070 Ti 12GB)
  python -m src.train.train --single

  # multi-GPU DDP (3x A5000) — launch via accelerate, not python directly
  accelerate launch --multi_gpu --num_processes 3 -m src.train.train --parallel

The two flags select a YAML profile. Everything else (model size, image size, batch size, lr,
grad-accum, etc.) lives in src/train/configs/{single,parallel}.yaml.
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

from src.corpus.common import DATA, PROJECT_ROOT
from src.train.data.dataset import Collator, OcrDataset, letterbox
from src.train.data.label_encoder import LabelEncoder
from src.train.eval.metrics import page_metrics
from src.train.eval.predict import parse_grounded
from src.train.model.khmer_ocr import build_model, count_params

CONFIG_DIR = Path(__file__).parent / "configs"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_config(profile: str) -> dict:
    path = CONFIG_DIR / f"{profile}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine_lr(step: int, base_lr: float, warmup: int, total: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _normalize(pixels: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(pixels.device, pixels.dtype)
    std = IMAGENET_STD.to(pixels.device, pixels.dtype)
    return (pixels - mean) / std


@torch.no_grad()
def evaluate(model, loader, accelerator) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for batch in loader:
        pixels = _normalize(batch["pixel_values"])
        out = model(pixel_values=pixels, labels=batch["labels"])
        # accelerator.gather handles cross-process averaging in DDP
        loss = accelerator.gather(out.loss.detach()).mean()
        total_loss += loss.item()
        n += 1
        if n >= 100:                  # cap eval at 100 batches per call
            break
    model.train()
    return total_loss / max(1, n)


@torch.no_grad()
def generation_eval(
    model,
    accelerator: Accelerator,
    enc: LabelEncoder,
    val_rows: list[dict],
    image_size: tuple[int, int],
    max_new_tokens: int,
    sample_n: int,
    iou_thresh: float = 0.5,
) -> dict[str, float] | None:
    """Decode ``sample_n`` val pages and report block-level metrics (CER, SyllER, box P/R/F1).

    Only the main process runs this — the work is small (a few dozen pages) and trivially
    serializable. Slower than the loss-only eval because each page is autoregressively decoded.
    Returns averaged metric dict, or None on non-main ranks.
    """
    if not accelerator.is_main_process:
        return None

    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    device = accelerator.device
    dtype = next(unwrapped.parameters()).dtype
    mean = IMAGENET_MEAN.to(device, dtype)
    std = IMAGENET_STD.to(device, dtype)

    rows = val_rows[:sample_n]
    aggregates: list[dict[str, float]] = []
    for row in rows:
        img_path = PROJECT_ROOT / row["image"].replace("\\", "/")
        label_path = PROJECT_ROOT / row["label"].replace("\\", "/")
        if not img_path.exists() or not label_path.exists():
            continue
        arr = letterbox(Image.open(img_path), image_size[0], image_size[1])
        px = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        px = ((px.to(device, dtype) - mean) / std)
        out_ids = unwrapped.generate(
            pixel_values=px,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=enc.pad,
            bos_token_id=enc.bos,
            eos_token_id=enc.eos,
            decoder_start_token_id=enc.bos,
        )
        decoded = enc.decode(out_ids[0].tolist())
        pred_blocks = parse_grounded(decoded)
        with open(label_path, encoding="utf-8") as f:
            gt = json.load(f)
        gt_blocks = [
            {"text": (b.get("text") or "").strip(), "bbox_norm": b["bbox_norm"]}
            for b in gt.get("blocks", []) if b.get("bbox_norm")
        ]
        aggregates.append(page_metrics(pred_blocks, gt_blocks, iou_thresh=iou_thresh))

    model.train()
    if not aggregates:
        return None
    out: dict[str, float] = {}
    for key in aggregates[0]:
        vals = [m[key] for m in aggregates
                if isinstance(m[key], (int, float)) and m[key] == m[key]]  # drop NaN
        out[key] = sum(vals) / len(vals) if vals else float("nan")
    return out


def save_ckpt(accelerator, model, optimizer, step: int, cfg: dict) -> None:
    if not accelerator.is_main_process:
        return
    ckpt_dir = PROJECT_ROOT / cfg["ckpt_dir"] / f"step_{step:08d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(ckpt_dir, safe_serialization=True)
    torch.save({"optimizer": optimizer.state_dict(), "step": step}, ckpt_dir / "optim.pt")
    # prune older ckpts beyond keep_last_n_ckpts
    keep = int(cfg.get("keep_last_n_ckpts", 3))
    parent = ckpt_dir.parent
    siblings = sorted(parent.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    for old in siblings[:-keep]:
        for f in old.rglob("*"):
            if f.is_file():
                f.unlink()
        old.rmdir()
    accelerator.print(f"[ckpt] saved -> {ckpt_dir}")


def load_ckpt(model, optimizer, accelerator: Accelerator, path: Path) -> int:
    """Resume model + optimizer state. Returns the global_step we resumed at.

    ``path`` may be either a specific ``step_NNNNNNNN`` directory or the parent ckpt dir, in
    which case the latest step dir is picked automatically. Loading happens after
    ``accelerator.prepare()`` so the optimizer instance matches.
    """
    path = Path(path)
    if path.is_dir() and not (path / "config.json").exists():
        siblings = sorted(path.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
        if not siblings:
            raise FileNotFoundError(f"no step_* dirs found in {path}")
        path = siblings[-1]
    accelerator.print(f"[resume] loading from {path}")
    unwrapped = accelerator.unwrap_model(model)
    src_model = type(unwrapped).from_pretrained(path)
    unwrapped.load_state_dict(src_model.state_dict())
    del src_model
    optim_file = path / "optim.pt"
    step = 0
    if optim_file.exists():
        blob = torch.load(optim_file, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(blob["optimizer"])
        step = int(blob.get("step", 0))
    accelerator.print(f"[resume] resumed at global_step={step}")
    return step


def train(cfg: dict) -> None:
    set_seed(cfg["seed"])
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"])

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["grad_accum_steps"],
        mixed_precision=cfg["mixed_precision"],
    )
    device = accelerator.device

    # tokenizer + datasets
    enc = LabelEncoder(PROJECT_ROOT / cfg["tokenizer_model"])
    collator = Collator(pad_id=enc.pad)
    train_ds = OcrDataset(PROJECT_ROOT / cfg["manifests"]["train"], enc,
                          tuple(cfg["image_size"]), cfg["max_label_len"], train=True)
    val_ds = OcrDataset(PROJECT_ROOT / cfg["manifests"]["val"], enc,
                        tuple(cfg["image_size"]), cfg["max_label_len"], train=False)
    # Keep raw val manifest rows for generation-eval (the prepared loader yields tokenized
    # batches; we still need image + label paths to run model.generate + compute metrics).
    val_rows_raw = list(val_ds.rows)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size_per_device"], shuffle=True,
        num_workers=cfg["num_workers"], collate_fn=collator, pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size_per_device"], shuffle=False,
        num_workers=max(1, cfg["num_workers"] // 2), collate_fn=collator, pin_memory=True,
    )

    # model
    model = build_model(
        profile_name=cfg["profile"], vocab_size=enc.vocab_size,
        pad_token_id=enc.pad, bos_token_id=enc.bos, eos_token_id=enc.eos,
    )
    if cfg.get("gradient_checkpointing"):
        model.encoder.gradient_checkpointing_enable()
        model.decoder.gradient_checkpointing_enable()

    n_params = count_params(model)
    accelerator.print(f"[model] profile={cfg['profile']} params={n_params/1e6:.1f}M "
                      f"image_size={cfg['image_size']} vocab={enc.vocab_size}")

    # optimizer
    opt_cfg = cfg["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt_cfg["lr"], betas=tuple(opt_cfg["betas"]), weight_decay=opt_cfg["weight_decay"],
    )

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # scheduler bookkeeping
    steps_per_epoch = math.ceil(len(train_loader) / cfg["grad_accum_steps"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = cfg["scheduler"]["warmup_steps"]
    accelerator.print(f"[sched] steps_per_epoch={steps_per_epoch} total_steps={total_steps} "
                      f"warmup={warmup}")

    label_smooth = cfg.get("label_smoothing", 0.0)
    clip_norm = cfg.get("gradient_clip_norm", 0.0)

    global_step = 0
    if cfg.get("resume_from"):
        global_step = load_ckpt(model, optimizer, accelerator, Path(cfg["resume_from"]))
    t0 = time.time()
    model.train()
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            with accelerator.accumulate(model):
                pixels = _normalize(batch["pixel_values"])
                out = model(pixel_values=pixels, labels=batch["labels"])
                loss = out.loss
                # transformers BartForCausalLM doesn't apply label smoothing internally; do it
                # the simple HF way via logits reshape:
                if label_smooth > 0.0:
                    logits = out.logits           # [B, T, V]
                    targets = batch["labels"]      # [B, T] with -100 = ignore
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-100,
                        label_smoothing=label_smooth,
                    )
                accelerator.backward(loss)
                if accelerator.sync_gradients and clip_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), clip_norm)
                # manual cosine LR with warmup
                if accelerator.sync_gradients:
                    lr = cosine_lr(global_step, opt_cfg["lr"], warmup, total_steps)
                    for g in optimizer.param_groups:
                        g["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % cfg["log_every_steps"] == 0:
                    elapsed = time.time() - t0
                    rate = global_step / max(elapsed, 1e-9)
                    accelerator.print(
                        f"epoch {epoch} step {global_step}/{total_steps} "
                        f"loss {loss.item():.4f} lr {lr:.2e} "
                        f"{rate:.2f} step/s"
                    )
                if global_step % cfg["eval_every_steps"] == 0:
                    val_loss = evaluate(model, val_loader, accelerator)
                    accelerator.print(f"  [eval] step {global_step} val_loss {val_loss:.4f}")
                    gen_n = int(cfg.get("gen_eval_samples", 0))
                    if gen_n > 0:
                        m = generation_eval(
                            model, accelerator, enc, val_rows_raw,
                            image_size=tuple(cfg["image_size"]),
                            max_new_tokens=cfg.get("gen_eval_max_new_tokens", 1024),
                            sample_n=gen_n,
                        )
                        if m is not None:
                            accelerator.print(
                                f"  [gen-eval] n={gen_n} cer={m['cer']:.4f} "
                                f"syllER={m['syllable_er']:.4f} "
                                f"P/R/F1={m['precision']:.3f}/{m['recall']:.3f}/{m['f1']:.3f} "
                                f"mIoU={m['mean_iou']:.3f}"
                            )
                if global_step % cfg["save_every_steps"] == 0:
                    save_ckpt(accelerator, model, optimizer, global_step, cfg)

    # final eval + ckpt
    val_loss = evaluate(model, val_loader, accelerator)
    accelerator.print(f"[final] val_loss {val_loss:.4f}")
    gen_n_final = int(cfg.get("gen_eval_samples_final", cfg.get("gen_eval_samples", 0)))
    if gen_n_final > 0:
        m = generation_eval(
            model, accelerator, enc, val_rows_raw,
            image_size=tuple(cfg["image_size"]),
            max_new_tokens=cfg.get("gen_eval_max_new_tokens", 1024),
            sample_n=gen_n_final,
        )
        if m is not None:
            accelerator.print(
                f"[final gen-eval] n={gen_n_final} cer={m['cer']:.4f} "
                f"syllER={m['syllable_er']:.4f} "
                f"P/R/F1={m['precision']:.3f}/{m['recall']:.3f}/{m['f1']:.3f} "
                f"mIoU={m['mean_iou']:.3f}"
            )
    save_ckpt(accelerator, model, optimizer, global_step, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the from-scratch Khmer OCR model")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--single", action="store_true",
                   help="single-GPU profile (4070 Ti 12GB, ~50M model)")
    g.add_argument("--parallel", action="store_true",
                   help="multi-GPU DDP profile (3x A5000 24GB, ~250M model). "
                        "Launch via `accelerate launch --multi_gpu --num_processes 3`.")
    ap.add_argument("--resume", type=str, default=None,
                    help="path to checkpoint dir to resume from. May be a specific "
                         "step_NNNNNNNN dir or the parent ckpt dir (latest is auto-picked).")
    args = ap.parse_args()

    profile = "single" if args.single else "parallel"
    cfg = load_config(profile)
    if args.resume:
        cfg["resume_from"] = args.resume
    train(cfg)


if __name__ == "__main__":
    main()
