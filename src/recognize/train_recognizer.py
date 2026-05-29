"""Train the Khmer text-line recognizer (SVTR-flavored encoder + attention decoder + CTC aux).

  python -m src.recognize.train_recognizer --single
  accelerate launch --multi_gpu --num_processes 3 -m src.recognize.train_recognizer --parallel

Loss = (1 - ctc_weight) * CE(attn, label-smoothed) + ctc_weight * CTC. Transcription is decoded
from the attention head; CTC is an auxiliary alignment regularizer. Eval = CER / syllable-ER via
src/train/eval/metrics.py on greedy-decoded val crops.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from src.corpus.common import PROJECT_ROOT
from src.recognize.data.line_dataset import LineDataset, RecCollator
from src.recognize.data.text_encoder import TextEncoder
from src.recognize.model.svtr import build_recognizer, count_params
from src.train.eval.metrics import cer, syllable_er

CONFIG_DIR = Path(__file__).parent / "configs"
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_config(profile: str) -> dict:
    with open(CONFIG_DIR / f"{profile}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine_lr(step, base, warmup, total):
    if step < warmup:
        return base * step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * base * (1.0 + math.cos(math.pi * min(1.0, prog)))


def _norm(px):
    return (px - IMAGENET_MEAN.to(px.device, px.dtype)) / IMAGENET_STD.to(px.device, px.dtype)


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
def evaluate(model, loader, accelerator, enc, max_lines):
    model.eval()
    unwrapped = accelerator.unwrap_model(model)
    cers, syls, n = [], [], 0
    for batch in loader:
        imgs = _norm(batch["images"])
        preds = unwrapped.greedy(imgs, enc.bos, enc.eos)
        # rebuild reference strings from attn targets
        for b in range(len(preds)):
            hyp = enc.decode(preds[b])
            ref_ids = [int(t) for t in batch["attn"][b].tolist()]
            ref = enc.decode(ref_ids)
            cers.append(cer(ref, hyp)); syls.append(syllable_er(ref, hyp)); n += 1
            if n >= max_lines:
                break
        if n >= max_lines:
            break
    model.train()
    return {"cer": float(np.mean(cers)) if cers else 1.0,
            "syllER": float(np.mean(syls)) if syls else 1.0, "n": n}


def train(cfg: dict) -> None:
    set_seed(cfg["seed"]); random.seed(cfg["seed"]); np.random.seed(cfg["seed"])
    accelerator = Accelerator(gradient_accumulation_steps=cfg["grad_accum_steps"],
                              mixed_precision=cfg["mixed_precision"])

    enc = TextEncoder(PROJECT_ROOT / cfg["tokenizer_model"])
    collate = RecCollator(pad_id=enc.pad)
    train_ds = LineDataset(cfg["manifests"]["train"], cfg["lines_root"], enc, train=True,
                           max_w=cfg["max_w"])
    val_ds = LineDataset(cfg["manifests"]["val"], cfg["lines_root"], enc, train=False,
                         max_w=cfg["max_w"])
    accelerator.print(f"[data] train lines {len(train_ds)} val {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size_per_device"], shuffle=True,
                              num_workers=cfg["num_workers"], collate_fn=collate, pin_memory=True,
                              drop_last=True, persistent_workers=cfg["num_workers"] > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size_per_device"], shuffle=False,
                            num_workers=2, collate_fn=collate)

    model = build_recognizer(cfg["profile"], enc.vocab_size, enc.pad)
    accelerator.print(f"[model] recognizer {cfg['profile']} params={count_params(model)/1e6:.1f}M "
                      f"vocab={enc.vocab_size}")
    oc = cfg["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=oc["lr"], betas=tuple(oc["betas"]),
                                  weight_decay=oc["weight_decay"])
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader)

    steps_per_epoch = math.ceil(len(train_loader) / cfg["grad_accum_steps"])
    total_steps = steps_per_epoch * cfg["epochs"]
    warmup = cfg["scheduler"]["warmup_steps"]
    ls = cfg.get("label_smoothing", 0.0)
    cw = cfg.get("ctc_weight", 0.3)
    clip = cfg.get("gradient_clip_norm", 0.0)
    accelerator.print(f"[sched] steps/epoch={steps_per_epoch} total={total_steps} warmup={warmup}")

    gstep = 0; t0 = time.time(); model.train()
    for epoch in range(cfg["epochs"]):
        for batch in train_loader:
            with accelerator.accumulate(model):
                imgs = _norm(batch["images"])
                dec_in = batch["attn"][:, :-1]
                target = batch["attn"][:, 1:]
                ctc_logits, attn_logits = model(imgs, dec_in)
                ce = F.cross_entropy(attn_logits.reshape(-1, attn_logits.size(-1)),
                                     target.reshape(-1), ignore_index=enc.pad, label_smoothing=ls)
                logp = ctc_logits.log_softmax(-1).transpose(0, 1)   # (T,B,V)
                ctc = F.ctc_loss(logp, batch["ctc_targets"], batch["input_lengths"],
                                 batch["ctc_lengths"], blank=enc.blank, zero_infinity=True)
                loss = (1 - cw) * ce + cw * ctc
                accelerator.backward(loss)
                if accelerator.sync_gradients and clip > 0:
                    accelerator.clip_grad_norm_(model.parameters(), clip)
                if accelerator.sync_gradients:
                    lr = cosine_lr(gstep, oc["lr"], warmup, total_steps)
                    for g in optimizer.param_groups:
                        g["lr"] = lr
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                gstep += 1
                if gstep % cfg["log_every_steps"] == 0:
                    rate = gstep / max(time.time() - t0, 1e-9)
                    accelerator.print(f"epoch {epoch} step {gstep}/{total_steps} "
                                      f"loss {loss.item():.4f} (ce {ce.item():.3f} ctc {ctc.item():.3f}) "
                                      f"lr {lr:.2e} {rate:.2f} st/s")
                if gstep % cfg["eval_every_steps"] == 0:
                    m = evaluate(model, val_loader, accelerator, enc, cfg["eval_max_lines"])
                    accelerator.print(f"  [eval] step {gstep} CER {m['cer']:.4f} "
                                      f"syllER {m['syllER']:.4f} (n={m['n']})")
                if gstep % cfg["save_every_steps"] == 0:
                    save_ckpt(accelerator, model, gstep, cfg)

    m = evaluate(model, val_loader, accelerator, enc, cfg["eval_max_lines"])
    accelerator.print(f"[final] CER {m['cer']:.4f} syllER {m['syllER']:.4f}")
    save_ckpt(accelerator, model, gstep, cfg)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Khmer line recognizer")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--single", action="store_true")
    g.add_argument("--parallel", action="store_true")
    args = ap.parse_args()
    train(load_config("single" if args.single else "parallel"))


if __name__ == "__main__":
    main()
