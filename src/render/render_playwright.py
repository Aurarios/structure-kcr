"""Render a page model to a PNG + DOM-extracted line/block boxes via Chromium (Playwright).

Chromium shapes Khmer correctly (HarfBuzz), and reading boxes straight from the DOM gives perfect
image-to-label alignment with zero manual annotation. The exact font is embedded as a base64
@font-face data URL so the requested font is guaranteed to be used regardless of page origin.

Coordinates are returned in CSS pixels plus the page's CSS width/height and the device pixel ratio;
callers normalize to [0,999] (resolution-independent) and/or to image pixels (css * dpr).
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)), autoescape=select_autoescape(["html"]))

_EXTRACT_JS = r"""
() => {
  const out = [];
  for (const el of document.querySelectorAll('[data-block-type]')) {
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    const text = (el.textContent || '').trim();
    if (!text) continue;
    const lines = [];
    try {
      const range = document.createRange();
      range.selectNodeContents(el);
      for (const lr of range.getClientRects()) {
        if (lr.width >= 1 && lr.height >= 1) lines.push([lr.left, lr.top, lr.right, lr.bottom]);
      }
    } catch (e) {}
    out.push({
      block_type: el.getAttribute('data-block-type'),
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


def build_html(page: dict) -> str:
    tmpl = _env.get_template("page.html.j2")
    ctx = dict(page)
    ctx["font_b64"] = _font_b64(page["font_path"]) if page.get("font_path") else ""
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
