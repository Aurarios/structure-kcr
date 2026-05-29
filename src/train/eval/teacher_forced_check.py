"""Teacher-forced diagnostic: does the model actually condition on the image?

Feeds the GROUND-TRUTH tokens as decoder input and takes argmax of the logits (no free-running
generation). If the teacher-forced predictions VARY by image and track the page content
(articles -> prose, forms -> field labels), the model is reading the image correctly and any bad
free-generation output is exposure bias, not model collapse.

  python -m src.train.eval.teacher_forced_check --ckpt data/checkpoints/single/step_00043750 \
      --indices 1 100 5000 --device cpu

Compare with src.train.eval.predict (free-running generation) on the same checkpoint to see the
exposure-bias gap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from src.corpus.common import DATA
from src.train.data.dataset import letterbox
from src.train.eval.predict import load_model_and_tokenizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
LABELS = DATA / "synthetic" / "labels"
IMAGES = DATA / "synthetic" / "images"


def main() -> None:
    ap = argparse.ArgumentParser(description="teacher-forced image-conditioning check")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--indices", type=int, nargs="+", default=[1, 100, 5000],
                    help="syn_NNNNNNN indices to test")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--n-tokens", type=int, default=40, help="how many predicted tokens to show")
    args = ap.parse_args()

    model, enc = load_model_and_tokenizer(args.ckpt)
    model.eval().to(args.device)
    size = model.config.encoder.image_size
    size = (size, size) if isinstance(size, int) else tuple(size)
    mean = IMAGENET_MEAN.to(args.device)
    std = IMAGENET_STD.to(args.device)

    for idx in args.indices:
        lab = json.loads((LABELS / f"syn_{idx:07d}.json").read_text(encoding="utf-8"))
        arr = letterbox(Image.open(IMAGES / f"syn_{idx:07d}.jpg"), size[0], size[1])
        px = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        px = ((px - mean) / std).to(args.device)
        ids = enc.encode_label(lab, max_len=args.n_tokens + 8)
        inp = torch.tensor(ids[:-1]).unsqueeze(0).to(args.device)
        with torch.no_grad():
            out = model(pixel_values=px, decoder_input_ids=inp)
        pred = out.logits.argmax(-1)[0].tolist()
        tf_text = enc.decode(pred[:args.n_tokens])
        gt_text = lab["blocks"][0]["text"][:50] if lab.get("blocks") else "(none)"
        print(f"syn_{idx:07d} [{lab['template']:16}]")
        print(f"    GT  first block : {gt_text!r}")
        print(f"    TF  prediction  : {tf_text[:80]!r}")
        print()


if __name__ == "__main__":
    main()
