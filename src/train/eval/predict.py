"""Inference CLI: image -> grounded markdown (and optional metrics vs. a ground-truth label).

  # decode one image, print grounded text
  python -m src.train.eval.predict --ckpt data/checkpoints/single/step_00010000 --image path/to.jpg

  # also evaluate against a ground-truth label JSON (CER, syllable ER, box IoU)
  python -m src.train.eval.predict --ckpt ... --image ... --label path/to/syn_NNN.json

The script parses the model's `<|ref|>{text}<|/ref|><|det|>[[<x1><y1><x2><y2>]]<|/det|>` blocks
back into a structured list, so the metrics in src/train/eval/metrics.py can run directly.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel

from src.corpus.common import PROJECT_ROOT
from src.train.data.dataset import letterbox
from src.train.data.label_encoder import LabelEncoder
from src.train.eval.metrics import page_metrics

# Pattern for one grounded block; non-greedy on the inner text to allow nested punctuation.
# Coordinate tokens decode back to "<NNN>" strings, so we parse them with a digit-captured regex.
_BLOCK_RE = re.compile(
    r"<\|ref\|>(?P<text>.*?)<\|/ref\|>\s*"
    r"<\|det\|>\s*\[\[\s*<(?P<x1>\d{1,3})>\s*<(?P<y1>\d{1,3})>\s*"
    r"<(?P<x2>\d{1,3})>\s*<(?P<y2>\d{1,3})>\s*\]\]\s*<\|/det\|>",
    re.DOTALL,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def parse_grounded(decoded: str) -> list[dict[str, Any]]:
    """Pull (text, bbox_norm) pairs out of the model's free-form output."""
    blocks: list[dict[str, Any]] = []
    for m in _BLOCK_RE.finditer(decoded):
        blocks.append({
            "text": m.group("text"),
            "bbox_norm": [int(m.group("x1")), int(m.group("y1")),
                          int(m.group("x2")), int(m.group("y2"))],
        })
    return blocks


def load_model_and_tokenizer(ckpt_dir: Path):
    model = VisionEncoderDecoderModel.from_pretrained(ckpt_dir)
    model.eval()
    # tokenizer model path is fixed by convention; allow override via ckpt-side artifact if present
    tok_path = PROJECT_ROOT / "data" / "tokenizer" / "khmer_ocr.model"
    enc = LabelEncoder(tok_path)
    return model, enc


def preprocess(image_path: Path, size: tuple[int, int]) -> torch.Tensor:
    img = Image.open(image_path)
    arr = letterbox(img, size[0], size[1])
    px = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return (px - IMAGENET_MEAN) / IMAGENET_STD


@torch.no_grad()
def predict(
    model: VisionEncoderDecoderModel,
    enc: LabelEncoder,
    image_path: Path,
    max_new_tokens: int = 1024,
    num_beams: int = 1,
    device: str = "cuda",
) -> tuple[str, list[dict[str, Any]]]:
    enc_size = model.config.encoder.image_size
    size = (enc_size, enc_size) if isinstance(enc_size, int) else tuple(enc_size)
    pixels = preprocess(image_path, size).to(device, dtype=next(model.parameters()).dtype)
    model = model.to(device)
    out_ids = model.generate(
        pixel_values=pixels,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        pad_token_id=enc.pad,
        bos_token_id=enc.bos,
        eos_token_id=enc.eos,
        decoder_start_token_id=enc.bos,
    )
    decoded = enc.decode(out_ids[0].tolist())
    blocks = parse_grounded(decoded)
    return decoded, blocks


def main() -> None:
    ap = argparse.ArgumentParser(description="Khmer OCR inference")
    ap.add_argument("--ckpt", type=Path, required=True, help="checkpoint dir (saved by train.py)")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--label", type=Path, default=None,
                    help="optional ground-truth label JSON; prints CER + box P/R if given")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU threshold for box matching")
    args = ap.parse_args()

    model, enc = load_model_and_tokenizer(args.ckpt)
    decoded, blocks = predict(model, enc, args.image,
                              args.max_new_tokens, args.num_beams, args.device)

    print("=" * 60)
    print("GROUNDED OUTPUT (raw decoded)")
    print("=" * 60)
    print(decoded)
    print()
    print("=" * 60)
    print(f"PARSED BLOCKS ({len(blocks)})")
    print("=" * 60)
    for i, b in enumerate(blocks):
        print(f"[{i:02d}] bbox={b['bbox_norm']}  text={b['text'][:80]!r}"
              f"{'...' if len(b['text']) > 80 else ''}")

    if args.label is not None:
        with open(args.label, encoding="utf-8") as f:
            gt = json.load(f)
        gt_blocks = [
            {"text": (g.get("text") or "").strip(), "bbox_norm": g["bbox_norm"]}
            for g in gt.get("blocks", []) if g.get("bbox_norm")
        ]
        m = page_metrics(blocks, gt_blocks, iou_thresh=args.iou)
        print()
        print("=" * 60)
        print(f"METRICS  (iou_thresh={args.iou})")
        print("=" * 60)
        print(f"  precision     {m['precision']:.3f}")
        print(f"  recall        {m['recall']:.3f}")
        print(f"  f1            {m['f1']:.3f}")
        print(f"  mean_iou      {m['mean_iou']:.3f}  (matched={int(m['matched_blocks'])})")
        print(f"  cer           {m['cer']:.4f}")
        print(f"  syllable_er   {m['syllable_er']:.4f}")
        print(f"  tp/fp/fn      {int(m['tp'])}/{int(m['fp'])}/{int(m['fn'])}")


if __name__ == "__main__":
    main()
