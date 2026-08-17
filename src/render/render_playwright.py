"""Render a page model to a PNG + DOM-extracted line/block boxes via Chromium (Playwright).

Chromium shapes Khmer correctly (HarfBuzz), and reading boxes straight from the DOM gives perfect
image-to-label alignment with zero manual annotation. The exact font is embedded as a base64
@font-face data URL so the requested font is guaranteed to be used regardless of page origin.

Coordinates are returned in CSS pixels plus the page's CSS width/height and the device pixel ratio;
callers normalize to [0,999] (resolution-independent) and/or to image pixels (css * dpr).
"""
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

# block types that carry a real raster asset (data-block-type == detector class)
_IMG_BLOCK_TYPES = {"image", "chart", "signature", "hand_drawing"}

_TEMPLATE_DIR = Path(__file__).parent / "templates"
# autoescape MUST be on: text can contain '<' (e.g. passport/ID MRZ "KHM<ACWU<<..."), which Chromium
# would otherwise parse as an HTML tag and silently drop the leaf. select_autoescape(["html"]) does
# NOT cover our "page.html.j2" name (suffix is ".j2"), so enable it explicitly.
_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)),
                   autoescape=select_autoescape(enabled_extensions=("html", "j2", "htm"),
                                                default=True))

_EXTRACT_JS = r"""
() => {
  // non-text regions are detected/cropped as a single box (no text lines)
  const NONTEXT = new Set(['image', 'chart', 'signature', 'hand_drawing', 'figure', 'photo', 'logo', 'stamp', 'formula_img']);
  const out = [];
  for (const el of document.querySelectorAll('[data-block-type]')) {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    const bt = el.getAttribute('data-block-type');
    const text = (el.textContent || '').trim();
    const isNonText = NONTEXT.has(bt);
    if (!text && !isNonText) continue;          // genuinely empty text element -> skip
    const lines = [];
    if (isNonText) {
      lines.push([r.left, r.top, r.right, r.bottom]);   // whole region = one box
    } else {
      try {
        const range = document.createRange();
        range.selectNodeContents(el);
        // getClientRects returns one rect PER INLINE SEGMENT (and duplicates for elements like
        // <b>/<mark>), not per visual line — a styled span fragments the line and trains the
        // detector to split mid-line. Union rects that overlap vertically into one rect per line.
        const frags = [];
        for (const lr of range.getClientRects()) {
          if (lr.width >= 1 && lr.height >= 1) frags.push([lr.left, lr.top, lr.right, lr.bottom]);
        }
        frags.sort((a, b) => a[1] - b[1]);
        for (const f of frags) {
          const last = lines[lines.length - 1];
          if (last) {
            const ov = Math.min(last[3], f[3]) - Math.max(last[1], f[1]);
            const minH = Math.min(last[3] - last[1], f[3] - f[1]);
            if (ov > 0.5 * minH) {                       // same visual line -> union
              last[0] = Math.min(last[0], f[0]); last[1] = Math.min(last[1], f[1]);
              last[2] = Math.max(last[2], f[2]); last[3] = Math.max(last[3], f[3]);
              continue;
            }
          }
          lines.push([f[0], f[1], f[2], f[3]]);
        }
      } catch (e) {}
    }
    out.push({
      block_type: bt,
      text: text,
      bbox: [r.left, r.top, r.right, r.bottom],
      lines: lines,
      tbl: el.getAttribute('data-tbl'),
      row: el.getAttribute('data-row'),
      col: el.getAttribute('data-col'),
    });
  }
  return {
    dpr: window.devicePixelRatio,
    width: document.documentElement.scrollWidth,
    height: document.documentElement.scrollHeight,
    leaves: out,
  };
}
"""


@dataclass
class RenderResult:
    image_png: bytes
    css_width: float
    css_height: float
    dpr: float
    leaves: list[dict]


@lru_cache(maxsize=64)
def _font_b64(font_path: str) -> str:
    return base64.b64encode(Path(font_path).read_bytes()).decode("ascii")


@lru_cache(maxsize=512)
def _img_data_url(img_path: str, max_px: int = 512) -> str:
    """Encode a real image file as a base64 data URL (guaranteed-apply, same trick as @font-face).

    Downscaled to <=max_px so the embedded HTML stays small and Chromium lays out fast; the on-page
    box is set by the CSS width/height in the template, not by the image's intrinsic size. Cached so
    a reused pool image is encoded once per worker. Returns "" if the file can't be read."""
    from PIL import Image, ImageOps
    try:
        im = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    except Exception:
        return ""
    w, h = im.size
    if max(w, h) > max_px:
        s = max_px / max(w, h)
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_html(page: dict) -> str:
    tmpl = _env.get_template("page.html.j2")
    ctx = dict(page)
    ctx["font_b64"] = _font_b64(page["font_path"]) if page.get("font_path") else ""
    # V5 font pairing: independent title/heading font, embedded the same data-URL way
    tfp = page.get("title_font_path")
    ctx["title_font_b64"] = _font_b64(tfp) if tfp and tfp != page.get("font_path") else ""
    # attach a data URL to each real-image block (image / signature / hand_drawing) that has a src;
    # copy the block dicts so the page's own blocks (used for round-trip text) are not mutated.
    # Recurses into compose-engine 'band' columns (their children are ordinary blocks).
    def _attach(bs: list) -> list:
        out = []
        for b in bs:
            if b.get("type") == "band":
                b = dict(b)
                b["cols"] = [dict(c, blocks=_attach(c["blocks"])) for c in b["cols"]]
            elif b.get("type") in _IMG_BLOCK_TYPES and b.get("src"):
                b = dict(b)
                b["img_url"] = _img_data_url(str(b["src"]))
            out.append(b)
        return out

    ctx["blocks"] = _attach(page.get("blocks", []))
    return tmpl.render(**ctx)


class PageRenderer:
    """Reusable Chromium instance. Use as a context manager for batch rendering."""

    def __init__(self, device_scale_factor: int = 2):
        self.dsf = device_scale_factor
        self._pw = None
        self._browser = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(args=["--no-sandbox", "--force-color-profile=srgb"])
        return self

    def __exit__(self, *exc):
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def render(self, page: dict) -> RenderResult:
        html = build_html(page)
        ctx = self._browser.new_context(
            viewport={"width": int(page["width"]), "height": 1000},
            device_scale_factor=self.dsf,
        )
        pg = ctx.new_page()
        try:
            pg.set_content(html, wait_until="networkidle")
            pg.evaluate("document.fonts.ready")
            info = pg.evaluate(_EXTRACT_JS)
            png = pg.screenshot(full_page=True, type="png")
        finally:
            pg.close()
            ctx.close()
        return RenderResult(
            image_png=png,
            css_width=float(info["width"]),
            css_height=float(info["height"]),
            dpr=float(info["dpr"]),
            leaves=info["leaves"],
        )
