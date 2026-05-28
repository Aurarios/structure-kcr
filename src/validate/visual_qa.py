"""Visual QA gate: render a fixed tricky-Khmer sample in EVERY usable font, overlay detected boxes,
and build an HTML contact sheet for human inspection.

A human checks two things the automated gates can't: (1) correct shaping of stacked/subscript
clusters (coeng), and (2) absence of .notdef boxes (□) indicating missing glyphs. Fonts that look
wrong here should be removed/auto-disabled in fonts.yaml regardless of codepoint coverage.

    python -m src.validate.visual_qa
    open data/manifests/contact_sheet.html
"""
from __future__ import annotations

import base64
import json

import cv2
import numpy as np

from ..corpus.common import DATA, PROJECT_ROOT, load_yaml
from ..fonts.validate_coverage import load_usable_fonts
from ..render.render_playwright import PageRenderer

QA_DIR = DATA / "manifests" / "qa"
SHEET = DATA / "manifests" / "contact_sheet.html"

# tricky clusters: triple stacks, subscripts, coeng-ro, vowel combinations
SAMPLE = (
    "ស្ត្រីខ្មែរ សាស្ត្រាចារ្យ បណ្ឌិត កុម្ភៈ សន្តិភាព "
    "ភ្នំពេញ ប្រទេសកម្ពុជា អក្សរសាស្ត្រ វិទ្យាស្ថាន គ្រួសារ"
)
DIGITS = "០១២៣៤៥៦៧៨៩"


def _sample_page(font: dict, layouts: dict) -> dict:
    pg = layouts["page"]
    return {
        "template": "article_single",
        "font_name": font["name"],
        "font_path": str((PROJECT_ROOT / font["path"]).resolve()),
        "width": 900,
        "margin": 50,
        "bg": "#ffffff",
        "page_bg": "#ffffff",
        "body_px": 30,
        "title_px": 44,
        "header_px": 36,
        "line_height": 1.9,
        "letter_spacing": 0.0,
        "color": "#101010",
        "align": "left",
        "columns": 1,
        "blocks": [
            {"type": "title", "text": "ការសាកល្បងពុម្ពអក្សរ " + font["name"]},
            {"type": "text", "text": SAMPLE},
            {"type": "text", "text": "លេខ៖ " + DIGITS},
        ],
    }


def _overlay(png: bytes, leaves: list[dict]) -> bytes:
    img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    for lf in leaves:
        x1, y1, x2, y2 = [int(v) for v in lf["bbox"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 120, 220), 2)
        for ln in lf.get("lines", []):
            a, b, c, d = [int(v) for v in ln]
            cv2.rectangle(img, (a, b), (c, d), (80, 200, 80), 1)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def run() -> dict:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    layouts = load_yaml("layouts.yaml")
    fonts = load_usable_fonts()
    if not fonts:
        print("No usable fonts. Run fetch_fonts + validate_coverage first.")
        return {}

    cards = []
    with PageRenderer(device_scale_factor=2) as r:
        for font in fonts:
            try:
                res = r.render(_sample_page(font, layouts))
                ih = None
                # scale boxes css->image px
                tmp = cv2.imdecode(np.frombuffer(res.image_png, np.uint8), cv2.IMREAD_COLOR)
                ih, iw = tmp.shape[:2]
                sx, sy = iw / res.css_width, ih / res.css_height
                for lf in res.leaves:
                    x1, y1, x2, y2 = lf["bbox"]
                    lf["bbox"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
                    lf["lines"] = [[a * sx, b * sy, c * sx, d * sy] for a, b, c, d in lf.get("lines", [])]
                out = _overlay(res.image_png, res.leaves)
                (QA_DIR / f"{font['name']}.png").write_bytes(out)
                cards.append((font, base64.b64encode(out).decode("ascii"), len(res.leaves)))
            except Exception as e:
                cards.append((font, None, f"render error: {e}"))
    _write_sheet(cards)
    print(f"Rendered {len(cards)} font samples.\n-> open {SHEET}")
    return {"fonts": len(cards)}


def _write_sheet(cards) -> None:
    parts = [
        "<!doctype html><meta charset='utf-8'><title>Khmer font QA</title>",
        "<style>body{font-family:sans-serif;background:#f5f4ef;margin:0;padding:24px;}"
        "h1{font-size:20px}.card{background:#fff;border:1px solid #e0ddd3;border-radius:10px;"
        "margin:14px 0;padding:14px;}.meta{font-size:13px;color:#555;margin-bottom:8px}"
        ".hard{color:#a8442f;font-weight:700}.bad{color:#b00}img{max-width:100%;border:1px solid #ddd}"
        "</style>",
        "<h1>Khmer font QA contact sheet</h1>",
        "<p>Check each sample for correct stacked/subscript shaping and no □ (.notdef) boxes. "
        "Blue = block box, green = line boxes.</p>",
    ]
    for font, b64, info in cards:
        diff = font.get("difficulty", "?")
        cls = "hard" if diff == "hard" else ""
        cov = font.get("coverage")
        cov_s = f"{cov*100:.0f}%" if isinstance(cov, (int, float)) else "n/a"
        parts.append("<div class='card'>")
        parts.append(f"<div class='meta'><b>{font['name']}</b> · "
                     f"<span class='{cls}'>difficulty={diff}</span> · coverage={cov_s} · boxes={info}</div>")
        if b64:
            parts.append(f"<img src='data:image/png;base64,{b64}'>")
        else:
            parts.append(f"<div class='bad'>{info}</div>")
        parts.append("</div>")
    SHEET.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    run()
