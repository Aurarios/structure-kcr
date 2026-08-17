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

from src.corpus.common import PROJECT_ROOT
from src.detect.data.build_det_targets import CLASSES, NONTEXT_CLASSES
from src.detect.data.build_obj_targets import (CLASSES_V4, CLASSES_V6, NONTEXT_V4,
                                               NONTEXT_V6)
from src.detect.infer_detector import detect_page
from src.detect.infer_obj import detect_page_obj
from src.detect.model.dbnet import load_detector
from src.detect.model.fcos import build_obj_detector
from src.detect.train_obj_detector import load_config as load_det_config
from src.pipeline.assemble import assemble
from src.pipeline.line_split import lines_for_detection
from src.recognize.data.text_encoder import TextEncoder
from src.recognize.model.svtr import build_recognizer

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
    enc = TextEncoder(PROJECT_ROOT / "data" / "tokenizer" / "khmer_ocr.model")
    rec = build_recognizer(rec_profile, enc.vocab_size, enc.pad)
    rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu")["model"])
    rec.eval().to(device)
    return det, rec, enc


def _prep_crops(img_bgr, boxes, max_w=CROP_MAX_W, gray=True):
    """gray=True desaturates each crop to luminance (3-ch) before recognition. The recognizer trained
    on black-on-light text; real docs with COLORED text (blue/red exam papers) are out-of-distribution
    and produce repetition garbage. Grayscale maps colored glyphs -> dark-on-light (blue/red have low
    luminance) so they read like black text. No-op for already-black text, so it's safe by default."""
    H, W = img_bgr.shape[:2]
    crops, valid, kept_idx = [], [], []
    for bi, (x1, y1, x2, y2) in enumerate(boxes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        # Khmer pre-base vowels ( េ ែ ៃ ោ ៅ ើ …) render to the LEFT of their consonant and post-base
        # signs to the right, so a box tightened to the ink clips edge glyphs -> the first/last word
        # gets cut off. Pad the CROP horizontally so the recognizer sees the whole edge glyph; this
        # is height-preserving (resize keeps aspect) so it doesn't shrink the text. Vertical pad is
        # kept small to MATCH build_line_crops (PAD=2): training crops are tight to the DOM line box,
        # and a large vertical pad would shrink the glyph after resize-to-48 -> train/infer mismatch.
        # The reported layout box (`valid`) stays exact, so assembly coords are unaffected.
        h = y2 - y1
        pad = max(2, int(h * 0.18))            # horizontal: recover edge pre-base vowel / post sign
        vpad = 3                               # vertical: match training's PAD=2 (tight crop)
        cx1, cx2 = max(0, x1 - pad), min(W, x2 + pad)
        cy1, cy2 = max(0, y1 - vpad), min(H, y2 + vpad)
        c = img_bgr[cy1:cy2, cx1:cx2]
        if c.size == 0:
            continue
        if gray:                                  # colored text -> dark-on-light via MIN channel:
            # min(R,G,B) is ~0 for ANY saturated color (red/blue/green) but stays high on light bg,
            # so red text (luminance only ~76, too light) also goes dark. Beats BGR2GRAY on colored docs.
            c = cv2.cvtColor(c.min(axis=2), cv2.COLOR_GRAY2BGR)
        nw = min(max_w, max(8, int(c.shape[1] * CROP_H / c.shape[0])))
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
def _recognize(rec, enc, crops, device, batch=64, decode="attn"):
    """decode='attn' (autoregressive attention head, default — best on dense lines) or 'ctc'
    (single-shot CTC head — cannot over-generate, fixes hallucination on wide/sparse crops)."""
    texts = []
    dtype = next(rec.parameters()).dtype
    for i in range(0, len(crops), batch):
        chunk = crops[i:i + batch]
        maxw = ((max(c.shape[1] for c in chunk) + 7) // 8) * 8
        t = torch.ones(len(chunk), 3, CROP_H, maxw)
        for j, c in enumerate(chunk):
            t[j, :, :, : c.shape[1]] = torch.from_numpy(c).permute(2, 0, 1).float() / 255.0
        t = ((t - IMAGENET_MEAN) / IMAGENET_STD).to(device, dtype)
        if decode == "ctc":
            for ids in rec.ctc_greedy(t):
                texts.append(_clean_line(enc.ctc_collapse(ids)))
        else:
            for ids in rec.greedy(t, enc.bos, enc.eos):
                texts.append(_clean_line(enc.decode(ids)))
    return texts


def refine_boxes_ink(img_bgr, dets, classes, nontext, pad: int = 3, min_ink_frac: float = 0.002):
    """Snap detected TEXT boxes to their ink extent (min-channel + Otsu).

    SHRINK: detector boxes on sparse regions (word banks, centered headings) carry empty margins
    that starve the recognizer of glyph resolution after the width-squish.
    EXTEND: the detector's edge regression is only ~3-6px accurate at input resolution, which at
    page scale can cut INTO a leading numeral ('១.' losing its left stroke). If ink is continuous
    AT a left/right edge, the box is cutting through a glyph — walk outward until a real whitespace
    gap (>=0.3*h) so the whole glyph is recovered. Capped at ~0.9*h so neighbors are never grabbed;
    edges with no ink at the boundary (true gaps) are never extended.
    Non-text classes and `table` pass through untouched."""
    gray_min = img_bgr.min(axis=2)                      # min-channel: colored ink stays dark
    H, W = gray_min.shape
    out = []
    for box, cid, score in dets:
        name = classes[cid]
        if name in nontext or name == "table":
            out.append((box, cid, score))
            continue
        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(W, int(box[2])); y2 = min(H, int(box[3]))
        crop = gray_min[y1:y2, x1:x2]
        if crop.size < 16:
            out.append((box, cid, score))
            continue
        thr, ink = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.nonzero(ink)
        if len(xs) < min_ink_frac * crop.size:
            out.append((box, cid, score))
            continue

        h = y2 - y1
        ext = int(h * 0.9)                              # max outward rescue distance
        gap_tol = max(3, int(h * 0.3))                  # whitespace run that ends a glyph

        def _extend(edge_x: int, direction: int) -> int:
            """Walk outward from a box edge through contiguous ink columns (direction -1 = left)."""
            cols = gray_min[y1:y2, max(0, edge_x - ext) if direction < 0 else edge_x:
                            edge_x if direction < 0 else min(W, edge_x + ext)]
            if cols.size == 0:
                return edge_x
            ink_cols = (cols < thr).sum(axis=0) > 0
            seq = ink_cols[::-1] if direction < 0 else ink_cols
            if not seq[0]:                              # no ink AT the edge -> true gap, no rescue
                return edge_x
            run, last_ink = 0, 0
            for k, v in enumerate(seq):
                if v:
                    last_ink, run = k, 0
                else:
                    run += 1
                    if run >= gap_tol:
                        break
            return edge_x + direction * (last_ink + 1)

        ex1, ex2 = _extend(x1, -1), _extend(x2, +1)
        region = gray_min[y1:y2, ex1:ex2]
        ink2 = region < thr
        ys2, xs2 = np.nonzero(ink2)
        if len(xs2) < min_ink_frac * region.size:
            out.append((box, cid, score))
            continue
        nb = [max(0.0, ex1 + xs2.min() - pad), max(float(box[1]), y1 + ys2.min() - pad),
              min(float(W), ex1 + xs2.max() + 1 + pad), min(float(box[3]), y1 + ys2.max() + 1 + pad)]
        out.append((nb if nb[2] - nb[0] >= 4 and nb[3] - nb[1] >= 4 else box, cid, score))
    return out


def drop_blank_regions(img_bgr, dets, classes, nontext, min_edge: float = 0.003):
    """Drop FIGURE-class boxes whose interior carries no structure (blank desk / wall / paper).

    A phone photo of a document puts a large featureless background around the page, and the
    detector — trained only on synthetic pages that ARE the document — happily calls it `image`.
    Real figures always carry edge energy: across v5 val GT the weakest class floor is `formula`
    at 0.0066 and `chart` at 0.0078 Canny density, whereas background boxes measure 0.0000. Text
    classes are untouched (a sparse text line legitimately has little ink).
    """
    if not dets:
        return dets
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    out = []
    for box, cid, score in dets:
        if classes[cid] not in nontext:
            out.append((box, cid, score))
            continue
        x1 = max(0, int(box[0])); y1 = max(0, int(box[1]))
        x2 = min(W, int(box[2])); y2 = min(H, int(box[3]))
        crop = gray[y1:y2, x1:x2]
        if crop.size < 64:
            out.append((box, cid, score))
            continue
        edges = cv2.Canny(cv2.GaussianBlur(crop, (3, 3), 0), 50, 150)
        if edges.mean() / 255.0 >= min_edge:
            out.append((box, cid, score))
    return out


def drop_nested_text(dets, classes, nontext, contain_frac: float = 0.7):
    """Drop a TEXT box that sits almost entirely inside a higher-scoring TEXT box.

    On real pages the detector emits partially-overlapping variants of one line (on the ID card:
    'លេខបណ្ណ', 'លេខបណ្ណ CardCode', 'លេខបណ្ណ CardCode B.PP...' all at once). They survive NMS because
    nested boxes of very different size have low pairwise IoU, and they survive suppress_row_merges
    because that needs the container to hold >=2 others. Each one is then recognized separately, so
    the assembled text repeats.

    Class-aware ON PURPOSE: infer_obj's generic `containment` option is class-blind and destroys
    tables (a `table` region contains every one of its cells) — measured -1.9 meanF1 on v5 val.
    Region classes and `table` are therefore never considered on either side of the comparison.
    """
    def _is_text(cid) -> bool:
        return classes[cid] not in nontext and classes[cid] != "table"

    order = sorted(range(len(dets)), key=lambda i: -dets[i][2])   # high score first
    keep: list[int] = []
    for i in order:
        box, cid, _ = dets[i]
        if not _is_text(cid):
            keep.append(i)
            continue
        area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
        nested = False
        for j in keep:
            kb, kcid, _ = dets[j]
            if not _is_text(kcid):
                continue
            inter = (max(0.0, min(box[2], kb[2]) - max(box[0], kb[0]))
                     * max(0.0, min(box[3], kb[3]) - max(box[1], kb[1])))
            if inter / area > contain_frac:
                nested = True
                break
        if not nested:
            keep.append(i)
    return [dets[i] for i in sorted(keep)]


def drop_small_fragments(dets, classes, nontext, min_rel_h: float = 0.55, max_score: float = 0.32):
    """Drop tiny LOW-CONFIDENCE text-class boxes (descender/border fragments under bubbles etc.).
    Confident small boxes (real short captions) are always kept via the score condition."""
    hs = sorted(b[3] - b[1] for b, c, s in dets
                if classes[c] not in nontext and classes[c] != "table")
    if not hs:
        return dets
    med = hs[len(hs) // 2]
    return [(b, c, s) for b, c, s in dets
            if classes[c] in nontext or classes[c] == "table"
            or (b[3] - b[1]) >= min_rel_h * med or s >= max_score]


def suppress_row_merges(dets, classes, nontext, contain_frac: float = 0.7):
    """Drop a TEXT-class box that mostly CONTAINS >=2 other text boxes and scores below them.

    Per-class/agnostic NMS can't kill these 'whole row as one weak detection' duplicates (a big box
    overlapping two small ones has low pairwise IoU with each), so a row of bubbles or two adjacent
    captions gets recognized twice. True line boxes never contain 2+ other lines, so this is safe.
    """
    drop = set()
    for i, (box, cid, sc) in enumerate(dets):
        if classes[cid] in nontext or classes[cid] == "table":
            continue
        bx1, by1, bx2, by2 = box
        contained_better = 0
        for j, (b2, c2, s2) in enumerate(dets):
            if j == i or classes[c2] in nontext or classes[c2] == "table":
                continue
            a2 = max(1.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
            iw = max(0.0, min(bx2, b2[2]) - max(bx1, b2[0]))
            ih = max(0.0, min(by2, b2[3]) - max(by1, b2[1]))
            if iw * ih / a2 > contain_frac and s2 > sc:
                contained_better += 1
        if contained_better >= 2:
            drop.add(i)
    return [d for i, d in enumerate(dets) if i not in drop]


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def deskew_document(pil_img, min_area_frac: float = 0.20, max_area_frac: float = 0.98,
                    min_skew_px: float = 12.0):
    """Perspective-correct a PHOTOGRAPHED document (page/card lying on a desk).

    Every training page IS the document, edge to edge and axis-aligned. A phone photo instead shows
    a rotated, perspective-warped card inside a background, which is out of distribution for both
    stages. Finding the document quad and warping it to a rectangle puts the input back in
    distribution.

    Deliberately conservative — a flat scan must pass through untouched. The warp is applied only
    when a convex 4-gon is found covering `min_area_frac`..`max_area_frac` of the frame AND its
    corners sit at least `min_skew_px` from the frame's own corners (otherwise the "document" is
    just the whole image and warping would be a no-op that only costs resampling).
    Returns (image, did_warp).
    """
    bgr = cv2.cvtColor(np.asarray(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    H, W = bgr.shape[:2]
    scale = 900.0 / max(H, W)
    small = cv2.resize(bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) \
        if scale < 1.0 else bgr.copy()
    sh, sw = small.shape[:2]

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # low Canny thresholds + generous dilation: a document edge against a similarly-lit desk is a
    # soft gradient, and at 40/120 with a 3x3 kernel the border fragments into 10-point contours
    # that approxPolyDP never reduces to a quad.
    edges = cv2.Canny(gray, 20, 80)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return pil_img, False

    frame_area = float(sh * sw)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(c)
        if not (min_area_frac * frame_area <= area <= max_area_frac * frame_area):
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        quad = _order_quad(approx.reshape(4, 2).astype(np.float32)) / scale
        corners = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
        if float(np.abs(quad - corners).max()) < min_skew_px:
            return pil_img, False                  # already axis-aligned and full-frame
        wA = np.linalg.norm(quad[2] - quad[3]); wB = np.linalg.norm(quad[1] - quad[0])
        hA = np.linalg.norm(quad[1] - quad[2]); hB = np.linalg.norm(quad[0] - quad[3])
        ow, oh = int(round(max(wA, wB))), int(round(max(hA, hB)))
        if ow < 64 or oh < 64:
            continue
        dst = np.array([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]], dtype=np.float32)
        warped = cv2.warpPerspective(bgr, cv2.getPerspectiveTransform(quad, dst), (ow, oh),
                                     flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)), True
    return pil_img, False


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

    crops, valid, kept_idx = _prep_crops(img_bgr, [b for b, _ in text_dets],
                                         det_cfg.get("rec_max_w", CROP_MAX_W),
                                         gray=det_cfg.get("gray_crops", True))
    texts = _recognize(rec, enc, crops, device, decode=det_cfg.get("decode", "attn")) if crops else []
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


def load_obj_models(det_ckpt, rec_ckpt, det_profile, rec_profile, device):
    """Load the V4 FCOS object-detection layout detector + the recognizer."""
    state = torch.load(det_ckpt, map_location="cpu")
    det = build_obj_detector(det_profile)
    det.load_state_dict(state.get("model", state))
    det.eval().to(device)
    print(f"[run_ocr] V4 FCOS detector loaded ({det.num_classes} classes)")
    enc = TextEncoder(PROJECT_ROOT / "data" / "tokenizer" / "khmer_ocr.model")
    rec = build_recognizer(rec_profile, enc.vocab_size, enc.pad)
    rec.load_state_dict(torch.load(rec_ckpt, map_location="cpu")["model"])
    rec.eval().to(device)
    return det, rec, enc


def _per_class_thresholds(config_name: str | None) -> dict | None:
    """Swept per-class score thresholds from a detector config, or None to use the flat threshold.

    A flat score_thresh is wrong in both directions on real pages: text classes score low (0.33-0.38
    on scans, vs near-1.0 on synthetic) so 0.3 clips real lines, while figure classes need a HIGHER
    bar or they fire on blank/watermarked background. See `det_eval --sweep`.
    """
    if not config_name:
        return None
    try:
        return load_det_config(config_name).get("score_thresh_per_class") or None
    except FileNotFoundError:
        print(f"[run_ocr] no detector config '{config_name}.yaml'; using flat score_thresh")
        return None


@torch.no_grad()
def run_ocr_v4(pil_img, det, rec, enc, det_cfg, device, merge_blocks=True):
    """V4 pipeline: region object-detection -> per-block line split -> recognize -> assemble.

    Region classes feed structure; multi-line text blocks are split into recognizer line crops;
    image/chart/signature/hand_drawing/formula are cropped as figures; the `table` region is dropped
    here (assembly rebuilds the table grid from its detected cells).
    """
    # taxonomy follows the CHECKPOINT, not a constant: a v6 detector emits class 16 (`word_bank`)
    # and indexing CLASSES_V4 with it would IndexError.
    n_cls = getattr(det, "num_classes", len(CLASSES_V4))
    CLASSES = CLASSES_V6 if n_cls == len(CLASSES_V6) else CLASSES_V4
    NONTEXT = NONTEXT_V6 if n_cls == len(CLASSES_V6) else NONTEXT_V4
    if det_cfg.get("deskew", False):                    # photographed docs -> flatten first
        pil_img, warped = deskew_document(pil_img)
        if warped:
            print(f"[run_ocr] deskewed to {pil_img.size[0]}x{pil_img.size[1]}")
    norm_img = normalize_dpi(pil_img, det_cfg.get("upscale_width", 2400))
    dets = detect_page_obj(det, norm_img, det_cfg.get("image_size", 1024), device, det_cfg)
    if det_cfg.get("suppress_row_merges", True):        # kill weak whole-row duplicate boxes (V5)
        dets = suppress_row_merges(dets, CLASSES, NONTEXT)
    if det_cfg.get("drop_nested_text", True):           # kill nested same-line duplicate boxes
        dets = drop_nested_text(dets, CLASSES, NONTEXT,
                                det_cfg.get("nested_contain_frac", 0.7))
    if det_cfg.get("drop_fragments", True):             # kill tiny low-conf fragment boxes (V5)
        dets = drop_small_fragments(dets, CLASSES, NONTEXT)
    img_bgr = cv2.cvtColor(np.asarray(norm_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    if det_cfg.get("drop_blank_regions", True):         # kill background `image` boxes on photos
        dets = drop_blank_regions(img_bgr, dets, CLASSES, NONTEXT,
                                  det_cfg.get("min_region_edge", 0.003))
    if det_cfg.get("ink_snap", True):                   # tighten text boxes to ink extent (V5)
        dets = refine_boxes_ink(img_bgr, dets, CLASSES, NONTEXT)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    text_lines = []                                   # (box, cls_id) line boxes to recognize
    figures, units = [], []
    for box, cid, score in dets:
        name = CLASSES[cid]
        if name == "table":
            continue                                  # structural region; rebuilt from cells
        if name in NONTEXT:
            x1, y1, x2, y2 = [int(v) for v in box]
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            fi = len(figures)
            figures.append({"index": fi, "box": [x1, y1, x2, y2], "cls": cid,
                            "class_name": name, "crop": img_bgr[max(0, y1):y2, max(0, x1):x2]})
            units.append({"box": [x1, y1, x2, y2], "text": "", "cls": cid, "fig_index": fi})
            continue
        for lb in lines_for_detection(gray, box, name):
            text_lines.append((lb, cid))

    crops, valid, kept_idx = _prep_crops(img_bgr, [b for b, _ in text_lines],
                                         det_cfg.get("rec_max_w", CROP_MAX_W),
                                         gray=det_cfg.get("gray_crops", True))
    texts = _recognize(rec, enc, crops, device, decode=det_cfg.get("decode", "attn")) if crops else []
    for j, (b, t) in enumerate(zip(valid, texts)):
        if t.strip():
            units.append({"box": list(b), "text": t, "cls": text_lines[kept_idx[j]][1]})

    w0, h0 = norm_img.size
    res = assemble(units, w0, h0, merge_blocks=merge_blocks, classes=CLASSES)
    res["norm_image"] = norm_img
    res["figures"] = figures
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="two-stage Khmer OCR")
    ap.add_argument("--det-ckpt", type=Path, required=True)
    ap.add_argument("--rec-ckpt", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--det-profile", default="single")
    ap.add_argument("--rec-profile", default="single")
    ap.add_argument("--rec-max-w", type=int, default=0,
                    help="recognizer crop max width; MUST match training max_w "
                         "(single=800, parallel_v3=1200). 0 = auto from --rec-profile")
    ap.add_argument("--decode", choices=["attn", "ctc"], default="attn",
                    help="recognizer decode: attn (best on dense lines) or ctc (no over-generation)")
    ap.add_argument("--det-size", type=int, default=960)
    ap.add_argument("--v4", action="store_true",
                    help="use the V4 FCOS object-detection layout detector (region-level)")
    ap.add_argument("--score-thresh", type=float, default=0.3,
                    help="V4 detection score threshold (fallback for classes absent from "
                         "--det-config's score_thresh_per_class)")
    ap.add_argument("--deskew", action="store_true",
                    help="perspective-correct a photographed page/card before OCR (no-op on flat "
                         "scans; recommended for phone photos)")
    ap.add_argument("--det-config", default="obj_single_v5",
                    help="detector config in src/detect/configs/ supplying swept per-class score "
                         "thresholds; '' disables and uses the flat --score-thresh")
    ap.add_argument("--upscale-width", type=int, default=2400,
                    help="upscale inputs narrower than this to match training DPI (0 disables)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--audit-dir", type=Path, default=None,
                    help="dump normalized image, detection-boxes image, and recognized text for audit")
    args = ap.parse_args()

    rec_max_w = args.rec_max_w or (1200 if args.rec_profile == "parallel" else 800)
    img = Image.open(args.image)
    if args.v4:
        det, rec, enc = load_obj_models(args.det_ckpt, args.rec_ckpt, args.det_profile,
                                        args.rec_profile, args.device)
        det_cfg = {"image_size": args.det_size, "score_thresh": args.score_thresh,
                   "nms_iou": 0.5, "max_det": 300, "upscale_width": args.upscale_width,
                   "rec_max_w": rec_max_w, "decode": args.decode, "deskew": args.deskew,
                   "score_thresh_per_class": _per_class_thresholds(args.det_config)}
        res = run_ocr_v4(img, det, rec, enc, det_cfg, args.device)
    else:
        det, rec, enc = load_models(args.det_ckpt, args.rec_ckpt, args.det_profile,
                                    args.rec_profile, args.device)
        det_cfg = {"image_size": args.det_size, "prob_thresh": 0.3,
                   "box_unclip_ratio": 2.0, "min_box_size": 4, "min_confidence": 0.5,
                   "upscale_width": args.upscale_width, "rec_max_w": rec_max_w, "decode": args.decode}
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
