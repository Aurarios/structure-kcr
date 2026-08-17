"""End-to-end two-stage OCR: image -> detector -> line crops -> recognizer -> assembled grounded md.

  python -m src.pipeline.run_ocr --det-ckpt ... --rec-ckpt ... --image page.jpg [--out boxed.jpg]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from pathlib import Path as _P
PROJECT_ROOT = _P(__file__).resolve().parent.parent  # bundle root (inference_v1)
from .build_det_targets import CLASSES, NONTEXT_CLASSES
from .infer_detector import detect_page
from .dbnet import load_detector
from .assemble import assemble
from .text_encoder import TextEncoder
from .svtr import build_recognizer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
CROP_H = 48
# MUST match the recognizer's training max_w (configs/single.yaml). Training squished any line
# wider than this to max_w, so the stem only ever produced (max_w // 8) encoder time-steps and the
# attention decoder only learned to attend over that many positions. Feeding wider crops at
# inference pushes the line-end into unseen positional range -> decoder emits <eos> early and drops
# the right side of the line. Keep these equal.
CROP_MAX_W = 800


def load_models(det_ckpt, rec_ckpt, det_profile, rec_profile, device):
    det, n_cls = load_detector(det_ckpt, det_profile, device)   # loads V1 (9-cls) or V2 (11-cls)
    print(f"[run_ocr] detector loaded with {n_cls} classes")
    enc = TextEncoder(PROJECT_ROOT / "assets" / "tokenizer" / "khmer_ocr.model")
    rec = build_recognizer(rec_profile, enc.vocab_size, enc.pad)
    rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu")["model"])
    rec.eval().to(device)
    return det, rec, enc


def _prep_crops(img_bgr, boxes):
    H, W = img_bgr.shape[:2]
    crops, valid, kept_idx = [], [], []
    for bi, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        # Khmer pre-base vowels ( េ ែ ៃ ោ ៅ ើ …) render to the LEFT of their consonant and post-base
        # signs to the right, so a box tightened to the ink clips edge glyphs -> the first/last word
        # gets cut off. Pad the CROP horizontally so the recognizer sees the whole edge glyph; the
        # reported layout box (appended to `valid`) stays exact, so assembly coords are unaffected.
        pad = max(2, int((y2 - y1) * 0.18))
        cx1, cx2 = max(0, x1 - pad), min(W, x2 + pad)
        c = img_bgr[max(0, y1):y2, cx1:cx2]
        if c.size == 0:
            continue
        nw = min(CROP_MAX_W, max(8, int(c.shape[1] * CROP_H / c.shape[0])))
        c = cv2.resize(c, (nw, CROP_H))
        crops.append(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
        valid.append((x1, y1, x2, y2))
        kept_idx.append(bi)
    return crops, valid, kept_idx


# keep Khmer block + ASCII + Latin-1 + whitespace; drop emoji / symbols / other-script junk that
# the recognizer hallucinates on hard crops (seals, photos, low contrast).
_JUNK = re.compile(r"[^ក-៿ -~ -ÿ\s]")


def _clean_line(t: str) -> str:
    return _JUNK.sub("", t).strip()


@torch.no_grad()
def _recognize(rec, enc, crops, device, batch=64):
    texts = []
    dtype = next(rec.parameters()).dtype
    for i in range(0, len(crops), batch):
        chunk = crops[i:i + batch]
        maxw = ((max(c.shape[1] for c in chunk) + 7) // 8) * 8
        t = torch.ones(len(chunk), 3, CROP_H, maxw)
        for j, c in enumerate(chunk):
            t[j, :, :, : c.shape[1]] = torch.from_numpy(c).permute(2, 0, 1).float() / 255.0
        t = ((t - IMAGENET_MEAN) / IMAGENET_STD).to(device, dtype)
        for ids in rec.greedy(t, enc.bos, enc.eos):
            texts.append(_clean_line(enc.decode(ids)))
    return texts


def normalize_dpi(pil_img, target_width: int = 2400):
    """Upscale a low-DPI input so text lines match the recognizer's training scale.

    The recognizer trained on crops DOWNSCALED to 48px from high-DPI (dsf=2, ~2480px-wide) renders.
    A small/low-DPI scan has ~25px lines that get UPSCALED to 48px (blurry) -> out of distribution.
    Upscaling the page to ~training width makes crops get downscaled like training -> the model reads
    it. This was the dominant real-document failure cause (verified by experiment).
    """
    from PIL import Image as _Image
    w, h = pil_img.size
    if w >= target_width:
        return pil_img
    nh = int(h * target_width / w)
    return pil_img.resize((target_width, nh), _Image.LANCZOS)


@torch.no_grad()
def run_ocr(pil_img, det, rec, enc, det_cfg, device, merge_blocks=True):
    norm_img = normalize_dpi(pil_img, det_cfg.get("upscale_width", 2400))
    detections = detect_page(det, norm_img, det_cfg.get("image_size", 1280), device, det_cfg,
                             return_class=True)
    img_bgr = cv2.cvtColor(np.asarray(norm_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    # split detections: text regions -> recognizer; image/formula regions -> crop (no recognition)
    text_dets = [(b, c) for b, c in detections if CLASSES[c] not in NONTEXT_CLASSES]
    nontext_dets = [(b, c) for b, c in detections if CLASSES[c] in NONTEXT_CLASSES]

    crops, valid, kept_idx = _prep_crops(img_bgr, [b for b, _ in text_dets])
    texts = _recognize(rec, enc, crops, device) if crops else []
    units = [{"box": list(b), "text": t, "cls": text_dets[kept_idx[j]][1]}
             for j, (b, t) in enumerate(zip(valid, texts)) if t.strip()]

    # crop the non-text regions "along the way" — figures/photos/formulas, kept as structure units
    figures = []
    for b, c in nontext_dets:
        x1, y1, x2, y2 = [int(v) for v in b]
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        fi = len(figures)
        figures.append({"index": fi, "box": [x1, y1, x2, y2], "cls": c,
                        "class_name": CLASSES[c], "crop": img_bgr[max(0, y1):y2, max(0, x1):x2]})
        units.append({"box": [x1, y1, x2, y2], "text": "", "cls": c, "fig_index": fi})

    w0, h0 = norm_img.size
    res = assemble(units, w0, h0, merge_blocks=merge_blocks)
    res["norm_image"] = norm_img        # the DPI-normalized image the boxes are aligned to
    res["figures"] = figures            # cropped image/formula regions (BGR arrays) + their boxes
    return res


def main() -> None:
    _ASSETS = Path(__file__).resolve().parent.parent / "assets" / "checkpoints"
    ap = argparse.ArgumentParser(description="two-stage Khmer OCR")
    ap.add_argument("--det-ckpt", type=Path, default=_ASSETS / "detector.pt")
    ap.add_argument("--rec-ckpt", type=Path, default=_ASSETS / "recognizer.pt")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--det-profile", default="single")
    ap.add_argument("--rec-profile", default="single")
    ap.add_argument("--det-size", type=int, default=1536)
    ap.add_argument("--upscale-width", type=int, default=2400,
                    help="upscale inputs narrower than this to match training DPI (0 disables)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--audit-dir", type=Path, default=None,
                    help="dump normalized image, detection-boxes image, and recognized text for audit")
    args = ap.parse_args()

    det, rec, enc = load_models(args.det_ckpt, args.rec_ckpt, args.det_profile,
                                args.rec_profile, args.device)
    det_cfg = {"image_size": args.det_size, "prob_thresh": 0.3,
               "box_unclip_ratio": 2.0, "min_box_size": 4, "min_confidence": 0.5,
               "upscale_width": args.upscale_width}
    img = Image.open(args.image)
    res = run_ocr(img, det, rec, enc, det_cfg, args.device)
    print("=" * 60); print(f"ASSEMBLED ({len(res['block_units'])} blocks, "
                            f"{len(res['line_units'])} lines)"); print("=" * 60)
    print(res["grounded_blocks"])

    norm_bgr = cv2.cvtColor(np.asarray(res["norm_image"].convert("RGB")), cv2.COLOR_RGB2BGR)

    def _draw(units):
        im = norm_bgr.copy()
        for i, u in enumerate(units):
            x1, y1, x2, y2 = [int(v) for v in u["box"]]
            cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(im, str(i), (x1, max(0, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 1, cv2.LINE_AA)
        return im

    if args.out:
        cv2.imwrite(str(args.out), _draw(res["line_units"])); print(f"-> {args.out}")

    if args.audit_dir:
        d = args.audit_dir
        d.mkdir(parents=True, exist_ok=True)
        # 1) the DPI-normalized (converted) image the model actually saw
        cv2.imwrite(str(d / "normalized.jpg"), norm_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        # 2) detection image with numbered line boxes
        cv2.imwrite(str(d / "detection.jpg"), _draw(res["line_units"]))
        # 3) recognized text: per-line (numbered, matches boxes) + assembled blocks + grounded
        lines_txt = "\n".join(f"[{i:02d}] {u['text']}" for i, u in enumerate(res["line_units"]))
        blocks_txt = "\n\n".join(u["text"] for u in res["block_units"])
        out = (f"SOURCE: {args.image}\n"
               f"normalized size: {res['norm_image'].size[0]}x{res['norm_image'].size[1]}\n"
               f"lines: {len(res['line_units'])}  blocks: {len(res['block_units'])}\n"
               f"{'='*60}\nPER-LINE (index matches boxes in detection.jpg)\n{'='*60}\n{lines_txt}\n\n"
               f"{'='*60}\nSTRUCTURED MARKDOWN (headings / lists / tables)\n{'='*60}\n"
               f"{res.get('markdown','')}\n\n"
               f"{'='*60}\nASSEMBLED BLOCKS (reading order, plain)\n{'='*60}\n{blocks_txt}\n\n"
               f"{'='*60}\nGROUNDED\n{'='*60}\n{res['grounded_blocks']}\n")
        (d / "recognition.txt").write_text(out, encoding="utf-8")
        print(f"-> audit artifacts in {d}\\ (normalized.jpg, detection.jpg, recognition.txt)")


if __name__ == "__main__":
    main()
