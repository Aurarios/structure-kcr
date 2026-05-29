"""Render a page and extract per-line (text, box) pairs for recognizer training.

The page renderer (`render_playwright.py`) gives per-line BOXES via Range.getClientRects() but not
per-line TEXT. Here we wrap every character in a <span>, read each char's client rect, and group
chars by their vertical position into lines — yielding exact (line_text, line_box) pairs. Reuses the
same HTML build + @font-face embedding so Khmer shaping (HarfBuzz via Chromium) is identical.

Used by `src/recognize/data/build_line_crops.py` to produce real in-context line crops + labels.
The existing page renderer is left untouched (AR/dataset path depends on it).
"""
from __future__ import annotations

from dataclasses import dataclass

from .render_playwright import build_html

# Wrap each text-node character in a span, reflow, then group spans into visual lines by top.
_LINE_JS = r"""
() => {
  function wrapChars(el){
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
    const tns = [];
    while (walker.nextNode()) tns.push(walker.currentNode);
    for (const tn of tns){
      if (!tn.textContent) continue;
      const frag = document.createDocumentFragment();
      for (const ch of tn.textContent){
        const s = document.createElement('span');
        s.setAttribute('data-c','1');
        s.textContent = ch;
        frag.appendChild(s);
      }
      tn.parentNode.replaceChild(frag, tn);
    }
  }
  const blocks = Array.from(document.querySelectorAll('[data-block-type]'))
                      .filter(el => (el.textContent||'').trim().length);
  for (const el of blocks) wrapChars(el);
  document.body.offsetHeight;  // force reflow

  const out = [];
  for (const el of blocks){
    const spans = el.querySelectorAll('span[data-c]');
    let cur = null;
    for (const sp of spans){
      const r = sp.getBoundingClientRect();
      const ch = sp.textContent;
      if (r.width < 0.01 && r.height < 0.01){            // zero-size (e.g. trailing space)
        if (cur) cur.text += ch;
        continue;
      }
      const tol = Math.max(4, r.height * 0.5);
      if (cur === null || Math.abs(r.top - cur.top0) > tol){
        if (cur) out.push(cur);
        cur = {left:r.left, top:r.top, right:r.right, bottom:r.bottom, top0:r.top, text:ch};
      } else {
        cur.left = Math.min(cur.left, r.left);
        cur.right = Math.max(cur.right, r.right);
        cur.top = Math.min(cur.top, r.top);
        cur.bottom = Math.max(cur.bottom, r.bottom);
        cur.text += ch;
      }
    }
    if (cur) out.push(cur);
  }
  return {
    dpr: window.devicePixelRatio,
    width: document.documentElement.scrollWidth,
    height: document.documentElement.scrollHeight,
    lines: out.map(l => ({box:[l.left,l.top,l.right,l.bottom], text:l.text})),
  };
}
"""


@dataclass
class LineRenderResult:
    image_png: bytes
    css_width: float
    css_height: float
    dpr: float
    lines: list[dict]   # [{box:[x1,y1,x2,y2] css-px, text:str}]


class LineRenderer:
    """Chromium instance that returns per-line (text, box). Context-manager, reusable across pages."""

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

    def render(self, page: dict) -> LineRenderResult:
        html = build_html(page)
        ctx = self._browser.new_context(
            viewport={"width": int(page["width"]), "height": 1000},
            device_scale_factor=self.dsf,
        )
        pg = ctx.new_page()
        try:
            pg.set_content(html, wait_until="networkidle")
            pg.evaluate("document.fonts.ready")
            png = pg.screenshot(full_page=True, type="png")   # screenshot BEFORE char-wrapping
            info = pg.evaluate(_LINE_JS)
        finally:
            pg.close()
            ctx.close()
        return LineRenderResult(
            image_png=png,
            css_width=float(info["width"]),
            css_height=float(info["height"]),
            dpr=float(info["dpr"]),
            lines=info["lines"],
        )
