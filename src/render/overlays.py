"""Procedural overlay generators: seals, stamps, signatures, highlights.

Each generator returns a PIL RGBA image. ``apply_overlays(img_bgr, rng, fonts)`` composites
1-2 random overlays onto a numpy BGR image (the rendered page) at random position, rotation,
and opacity. Labels are NOT changed — overlays are visual noise the model learns to read through.

Why procedural and not stock PNGs:
  - unlimited variety (every page gets a unique seal)
  - no asset licensing issues
  - parameterized via rng so reproducible per-seed
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Khmer stamp/seal vocabulary — short institutional phrases that look plausible on a real seal.
_SEAL_TEXTS_KM = [
    "ច្បាប់ដើម", "ច្បាប់ចម្លង", "បានទទួល", "សម្ងាត់", "បន្ទាន់",
    "ក្រសួងយុត្តិធម៌", "តុលាការ", "រាជរដ្ឋាភិបាល", "ការផ្តល់សិទ្ធិ",
    "ឯកសារដើម", "បានពិនិត្យ", "ផ្តល់សិទ្ធិ", "ច្បាប់ត្រឹមត្រូវ",
]
_SEAL_TEXTS_EN = [
    "ORIGINAL", "COPY", "DRAFT", "CONFIDENTIAL", "URGENT",
    "RECEIVED", "APPROVED", "MINISTRY OF JUSTICE", "OFFICIAL",
    "VERIFIED", "CERTIFIED", "VOID", "PAID",
]

# Inky colors — slightly desaturated so they look stamped, not painted
_SEAL_COLORS = [
    (180, 30, 30),    # red ink
    (30, 60, 160),    # blue ink
    (90, 30, 130),    # purple ink
    (20, 100, 50),    # dark green
    (50, 50, 50),     # dark gray / faded black
]

_HIGHLIGHT_COLORS = [
    (255, 240, 80, 90),   # yellow
    (255, 150, 200, 80),  # pink
    (140, 255, 160, 80),  # green
    (180, 220, 255, 80),  # light blue
]


# ---------------------------------------------------------------------------
# Helpers

def _load_random_font(fonts: list[dict], rng, size: int) -> ImageFont.FreeTypeFont:
    """Pick a usable Khmer TTF from the project fonts dir at the requested px size."""
    font_info = rng.choice(fonts) if fonts else None
    if font_info and "path" in font_info and Path(font_info["path"]).exists():
        return ImageFont.truetype(font_info["path"], size=size)
    return ImageFont.load_default()


def _try_bilevel_alpha(rgba: Image.Image, sigma: float) -> Image.Image:
    """Slight blur to fake the irregular ink texture of a real stamp."""
    if sigma > 0:
        rgba = rgba.filter(ImageFilter.GaussianBlur(sigma))
    return rgba


# ---------------------------------------------------------------------------
# Round seal — two concentric circles, text inside

def make_seal(rng, fonts: list[dict]) -> Image.Image:
    radius = rng.randint(80, 160)
    size = radius * 2 + 20
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = rng.choice(_SEAL_COLORS)
    rgba = (*color, 235)
    cx, cy = size // 2, size // 2

    # outer circle (thick border)
    outer_w = max(3, radius // 18)
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        outline=rgba, width=outer_w,
    )
    # inner ring (thin)
    inner_r = radius - max(8, radius // 10)
    draw.ellipse(
        [cx - inner_r, cy - inner_r, cx + inner_r, cy + inner_r],
        outline=rgba, width=max(1, outer_w // 2),
    )

    # text inside — pick Khmer or English with 50/50 weight
    use_km = rng.random() < 0.6
    pool = _SEAL_TEXTS_KM if use_km else _SEAL_TEXTS_EN
    txt = rng.choice(pool)
    font_size = max(12, radius // 4)
    font = _load_random_font(fonts, rng, font_size)

    # rough centering using textbbox
    bbox = draw.textbbox((0, 0), txt, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((cx - tw // 2 - bbox[0], cy - th // 2 - bbox[1]), txt, font=font, fill=rgba)

    # optional star/dot in center bottom for that seal look
    if rng.random() < 0.5:
        dot_r = max(3, radius // 16)
        draw.ellipse([cx - dot_r, cy + inner_r // 2 - dot_r,
                      cx + dot_r, cy + inner_r // 2 + dot_r], fill=rgba)

    # rotate slightly
    img = img.rotate(rng.uniform(-15, 15), resample=Image.BILINEAR, expand=True)
    return _try_bilevel_alpha(img, sigma=rng.uniform(0.6, 1.2))


# ---------------------------------------------------------------------------
# Rectangular stamp — border + bold text

def make_rect_stamp(rng, fonts: list[dict]) -> Image.Image:
    use_km = rng.random() < 0.5
    pool = _SEAL_TEXTS_KM if use_km else _SEAL_TEXTS_EN
    txt = rng.choice(pool)
    color = rng.choice(_SEAL_COLORS)
    rgba = (*color, 230)

    font_size = rng.randint(28, 56)
    font = _load_random_font(fonts, rng, font_size)
    tmp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), txt, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad_x, pad_y = rng.randint(14, 28), rng.randint(8, 18)
    w, h = tw + 2 * pad_x, th + 2 * pad_y

    img = Image.new("RGBA", (w + 6, h + 6), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    bw = rng.randint(3, 6)
    draw.rectangle([3, 3, w + 2, h + 2], outline=rgba, width=bw)
    draw.text((3 + pad_x - bbox[0], 3 + pad_y - bbox[1]), txt, font=font, fill=rgba)

    img = img.rotate(rng.uniform(-12, 12), resample=Image.BILINEAR, expand=True)
    return _try_bilevel_alpha(img, sigma=rng.uniform(0.5, 1.0))


# ---------------------------------------------------------------------------
# Signature — bezier scribble with ink color

def make_signature(rng) -> Image.Image:
    w = rng.randint(200, 400)
    h = rng.randint(60, 110)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = rng.choice([(20, 20, 30), (30, 30, 110), (40, 50, 80)])
    rgba = (*color, 235)
    pen = rng.randint(2, 4)

    # 3-5 cursive strokes built from quadratic beziers
    n_strokes = rng.randint(3, 5)
    x = rng.randint(10, w // 4)
    for _ in range(n_strokes):
        cx = x + rng.randint(20, 60)
        cy = rng.randint(10, h - 10)
        ex = cx + rng.randint(20, 80)
        ey = rng.randint(10, h - 10)
        # discretize the bezier
        pts = []
        for t in range(20):
            t1 = t / 19.0
            bx = (1 - t1) ** 2 * x + 2 * (1 - t1) * t1 * cx + t1 ** 2 * ex
            by = (1 - t1) ** 2 * rng.randint(10, h - 10) + 2 * (1 - t1) * t1 * cy + t1 ** 2 * ey
            pts.append((bx, by))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=rgba, width=pen)
        x = ex
        if x > w - 30:
            break

    img = img.rotate(rng.uniform(-8, 8), resample=Image.BILINEAR, expand=True)
    return img


# ---------------------------------------------------------------------------
# Translucent highlight stripe — covers a few lines of text

def make_highlight(rng) -> Image.Image:
    w = rng.randint(300, 700)
    h = rng.randint(28, 55)
    color = rng.choice(_HIGHLIGHT_COLORS)
    img = Image.new("RGBA", (w, h), color)
    img = img.rotate(rng.uniform(-3, 3), resample=Image.BILINEAR, expand=True)
    return _try_bilevel_alpha(img, sigma=0.8)


# ---------------------------------------------------------------------------
# Compositor: choose 1-2 overlays and paste them onto the page

_GENERATORS = [
    ("seal",      make_seal,      0.45),
    ("rect",      make_rect_stamp, 0.25),
    ("signature", lambda rng, fonts: make_signature(rng), 0.20),
    ("highlight", lambda rng, fonts: make_highlight(rng), 0.10),
]


def _pick_generator(rng):
    weights = [w for _, _, w in _GENERATORS]
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for name, fn, w in _GENERATORS:
        acc += w
        if pick <= acc:
            return name, fn
    return _GENERATORS[-1][0], _GENERATORS[-1][1]


def apply_overlays(img_bgr: np.ndarray, rng, fonts: list[dict]) -> np.ndarray:
    """Composite 1-2 random overlays onto a BGR numpy image. Returns the modified image.

    Overlays use random position/opacity. Highlights are biased toward the body area;
    seals/stamps are biased toward corners + center bottom (like real stamped documents).
    """
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    pw, ph = pil.size

    n_overlays = 1 if rng.random() < 0.7 else 2
    for _ in range(n_overlays):
        name, gen = _pick_generator(rng)
        overlay = gen(rng, fonts)
        ow, oh = overlay.size

        # opacity jitter (multiply alpha)
        opacity = rng.uniform(0.55, 0.95)
        if opacity < 0.99:
            r, g, b, a = overlay.split()
            a = a.point(lambda v: int(v * opacity))
            overlay = Image.merge("RGBA", (r, g, b, a))

        # position bias by overlay type
        if name == "highlight":
            # mid-page horizontal stripe
            x = rng.randint(40, max(41, pw - ow - 40))
            y = rng.randint(int(ph * 0.15), max(int(ph * 0.15) + 1, ph - oh - 40))
        elif name == "signature":
            # bottom-right quadrant
            x = rng.randint(max(40, pw // 2), max(pw // 2 + 1, pw - ow - 20))
            y = rng.randint(max(20, int(ph * 0.6)), max(int(ph * 0.6) + 1, ph - oh - 20))
        else:
            # seals/stamps — corners or center-bottom
            corner = rng.randint(0, 3)
            margin = 30
            if corner == 0:    # bottom-left
                x = rng.randint(margin, max(margin + 1, pw // 3))
                y = rng.randint(max(margin, ph - oh - 200), max(ph - oh - 200 + 1, ph - oh - margin))
            elif corner == 1:  # bottom-right
                x = rng.randint(max(margin, 2 * pw // 3), max(2 * pw // 3 + 1, pw - ow - margin))
                y = rng.randint(max(margin, ph - oh - 200), max(ph - oh - 200 + 1, ph - oh - margin))
            elif corner == 2:  # top-right
                x = rng.randint(max(margin, 2 * pw // 3), max(2 * pw // 3 + 1, pw - ow - margin))
                y = rng.randint(margin, max(margin + 1, ph // 4))
            else:              # center
                x = rng.randint(max(margin, pw // 4), max(pw // 4 + 1, 3 * pw // 4 - ow))
                y = rng.randint(max(margin, ph // 3), max(ph // 3 + 1, 2 * ph // 3 - oh))

        pil.alpha_composite(overlay, dest=(x, y))

    out_rgb = np.asarray(pil.convert("RGB"))
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
