"""Recognizer inference: line crop image -> Khmer text.

Two modes:
  # recognize a single line-crop image
  python -m src.recognize.infer_recognizer --ckpt ... --image crop.jpg --device cpu
  # sample N validation crops and show prediction vs ground truth + CER
  python -m src.recognize.infer_recognizer --ckpt ... --sample 20 --device cpu
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from src.corpus.common import PROJECT_ROOT
from src.recognize.data.text_encoder import TextEncoder
from src.recognize.model.svtr import build_recognizer
from src.train.eval.metrics import cer, syllable_er

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CROP_H = 48
CROP_MAX_W = 1200
VAL_MANIFEST = "E:/kcr-recognize/lines/lines_val.jsonl"
LINES_ROOT = "E:/kcr-recognize/lines"


def load(ckpt, profile, device):
    enc = TextEncoder(PROJECT_ROOT / "data" / "tokenizer" / "khmer_ocr.model")
    model = build_recognizer(profile, enc.vocab_size, enc.pad)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    model.eval().to(device)
    return model, enc


def _prep(img_bgr, device, dtype):
    if img_bgr.shape[0] != CROP_H:
        s = CROP_H / img_bgr.shape[0]
        img_bgr = cv2.resize(img_bgr, (max(8, int(img_bgr.shape[1] * s)), CROP_H))
    if img_bgr.shape[1] > CROP_MAX_W:
        img_bgr = cv2.resize(img_bgr, (CROP_MAX_W, CROP_H))
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return ((t - IMAGENET_MEAN) / IMAGENET_STD).to(device, dtype)


@torch.no_grad()
def recognize(model, enc, img_bgr, device):
    px = _prep(img_bgr, device, next(model.parameters()).dtype)
    ids = model.greedy(px, enc.bos, enc.eos)[0]
    return enc.decode(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description="Khmer line recognizer inference")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--profile", default="single")
    ap.add_argument("--image", type=Path, default=None, help="single line-crop image")
    ap.add_argument("--sample", type=int, default=0, help="sample N val crops, show pred vs GT + CER")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, enc = load(args.ckpt, args.profile, args.device)

    if args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            raise SystemExit(f"could not read {args.image}")
        print("PRED:", recognize(model, enc, img, args.device))
        return

    if args.sample:
        rows = [json.loads(l) for l in open(VAL_MANIFEST, encoding="utf-8")]
        random.shuffle(rows)
        rows = rows[: args.sample]
        cers, syls = [], []
        print(f"{'CER':>6}  {'PRED  /  GROUND TRUTH'}")
        print("-" * 70)
        for r in rows:
            img = cv2.imread(str(Path(LINES_ROOT) / r["crop"]))
            if img is None:
                continue
            pred = recognize(model, enc, img, args.device)
            gt = r["text"]
            c = cer(gt, pred); cers.append(c); syls.append(syllable_er(gt, pred))
            mark = "✓" if c == 0 else f"{c:.2f}"
            print(f"{mark:>6}  {pred}")
            if c > 0:
                print(f"{'':>6}  {gt}   <- GT")
        print("-" * 70)
        print(f"mean CER {np.mean(cers):.4f}  syllER {np.mean(syls):.4f}  (n={len(cers)})")
        return

    raise SystemExit("provide --image or --sample N")


if __name__ == "__main__":
    main()
