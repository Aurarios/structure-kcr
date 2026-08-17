"""V5 compositional layout engine — the "composed" pseudo-layout.

Replaces hand-coded page structure with PROCEDURAL composition, adapting DocLayout-YOLO's
"Mesh-candidate BestFit" idea (arXiv:2410.12628) to our HTML/DOM pipeline: instead of bin-packing
pre-rendered image crops, we pack *HTML block factories* into a sampled band scaffold and let
Chromium lay them out. Differences from the paper, by design:

- Their elements are fixed-size crops; ours are live text whose height is unknown until layout.
  True 2D packing of fixed cells would clip text (DOM text != visible text -> broken labels), so
  text always flows in auto-height flex columns ("bands") and only intrinsically-sized elements
  (figures) get fixed cells. The packer works on WIDTH budgets: column counts and figure cell sizes
  are chosen to maximize fill of the band width (the BestFit criterion that survives auto-height).
- Their diversity levers (stratified element sampling + min element count + >=N distinct
  categories) are kept: every page draws bands from a weighted pool with quotas that guarantee
  multiple distinct detector classes, including the rare ones (chart/formula/hand_drawing).

A page = header band? + 4..9 content bands + footer band?. Each band is a flex row of auto-height
columns holding ordinary template blocks, so rendering, DOM box extraction, round-trip and target
building are completely unchanged.
"""
from __future__ import annotations

import random


def _ls():
    # lazy import: layout_sampler imports this module at its bottom to register "composed";
    # importing it at our top level would be circular at load time.
    from src.render import layout_sampler as ls
    return ls


# ---------------------------------------------------------------- element factories
# each returns a list of blocks for ONE band column of width ~cw px

def _f_paragraphs(rng, corpus, assets, cw):
    # short paragraphs on purpose: det targets are per-LINE, so long multi-column text floods the
    # class distribution and starves the region classes (measured by the stats gate)
    ls = _ls()
    out = []
    if rng.random() < 0.60:
        out.append({"type": rng.choice(["heading", "heading", "subheading"]),
                    "text": ls._trim(corpus.sentence(), 34)})
        if rng.random() < 0.35:
            out.append({"type": "subheading", "text": ls._trim(corpus.sentence(), 28)})
    for _ in range(rng.randint(1, 2)):
        out.append({"type": "text", "text": corpus.paragraph(rng.randint(2, 4))})
    return out


def _f_list(rng, corpus, assets, cw):
    ls = _ls()
    items = [ls._trim(corpus.sentence(), rng.randint(30, 70)) for _ in range(rng.randint(3, 7))]
    return [{"type": "list", "items": ls._numbered(rng, items)}]


def _f_figure(rng, corpus, assets, cw, btype=None):
    ls = _ls()
    cats = {"image": ["photos"], "chart": ["charts"], "hand_drawing": ["drawings"]}
    # weighted mix (proportional-sampling truth: equal thirds starved image at 1.39% < 1.5%)
    r = rng.random()
    bt = btype or ("image" if r < 0.45 else "chart" if r < 0.75 else "hand_drawing")
    # BestFit on width: the figure fills 72..96% of its column
    w = int(cw * rng.uniform(0.72, 0.96))
    h = int(w * rng.uniform(0.55, 0.95))
    out = [ls._img_block(assets, rng, cats[bt], w, h, btype=bt)]
    if rng.random() < 0.75:
        out.append({"type": "caption", "text": ls._trim(corpus.sentence(), 60), "center": True})
    return out


def _f_table(rng, corpus, assets, cw):
    ls = _ls()
    # BestFit on width: column count budgeted by ~120..200px per table column
    max_cols = max(2, min(5, cw // rng.randint(120, 200)))
    return [ls._table(rng, corpus, rows_rng=(2, 6), cols_rng=(2, max_cols),
                      numeric_last=rng.random() < 0.4)]


def _f_form(rng, corpus, assets, cw):
    ls = _ls()
    n = rng.randint(3, 6)
    fields = [{"label": rng.choice(ls._LABELS), "value": rng.choice(ls._VALUES)}
              for _ in range(n)]
    return [{"type": "form", "fields": fields}]


def _f_bubbles(rng, corpus, assets, cw):
    ls = _ls()
    items = [ls._short(corpus, rng, rng.randint(6, 14)) for _ in range(rng.randint(4, 8))]
    blk = {"type": "bubbles", "items": items, "radius": rng.randint(8, 26),
           "color": rng.choice(["#cc3333", "#1a3a8f", "#207040"])}
    if rng.random() < 0.45:     # single-container word-bank row (real-worksheet pattern)
        blk.update(items=[" ".join(items)], justify="center", item_type="word_bank",
                   word_spacing=rng.randint(10, 30))
    return [blk]


def _f_formula(rng, corpus, assets, cw):
    ls = _ls()
    return [{"type": "formula", "text": ls._formula(rng)} for _ in range(rng.randint(1, 2))]


def _f_signatures(rng, corpus, assets, cw):
    ls = _ls()
    out = [ls._img_block(assets, rng, ["signatures"], rng.randint(140, 220),
                         rng.randint(60, 110), btype="signature")]
    if rng.random() < 0.6:
        out.append({"type": "caption", "text": ls._trim(corpus.sentence(), 30), "center": True})
    return out


# (factory, weight, size_class) — size_class drives stratified sampling: 'narrow' factories can
# live in 2-3 column bands, 'wide' ones get 1-2 columns so they keep a workable width budget.
_FACTORIES = [
    (_f_paragraphs, 2.4, "narrow"),
    (_f_list,       1.2, "narrow"),
    (_f_figure,     2.4, "narrow"),
    (_f_table,      1.5, "wide"),
    (_f_form,       1.0, "wide"),
    # raised 0.7 -> 1.6: word_bank vs list_item is a fine-grained visual discrimination (one
    # rounded container holding a word row, vs one rounded bubble per word) and at 0.7 only ~5% of
    # pages carried an example — too thin to learn the contrast. See CLASSES_V6.
    (_f_bubbles,    1.6, "wide"),
    (_f_formula,    1.2, "narrow"),
    (_f_signatures, 0.8, "narrow"),
]


def _pick_factory(rng, pool):
    ls = _ls()
    return ls._weighted_choice(rng, [f for f, _, _ in pool], [w for _, w, _ in pool])


def _band(rng, corpus, assets, content_w, ncols, pool):
    """One flex band: ncols auto-height columns, each filled by a sampled factory."""
    gap = rng.randint(22, 40)
    cw = int((content_w - gap * (ncols - 1)) / ncols)
    cols = []
    for _ in range(ncols):
        fac = _pick_factory(rng, pool)
        cols.append({"flex": 1, "blocks": fac(rng, corpus, assets, cw)})
    return {"type": "band", "gap": gap, "cols": cols, "rule": rng.random() < 0.12}


def _sidebar_band(rng, corpus, assets, content_w):
    """Asymmetric 2-column band: narrow rail (list/captions/figure) + wide body text."""
    gap = rng.randint(24, 40)
    narrow_w = int(content_w * 0.30)
    wide_w = content_w - gap - narrow_w
    rail_fac = rng.choice([_f_list, _f_figure, _f_form, _f_formula])
    rail = {"flex": 0.42, "blocks": rail_fac(rng, corpus, assets, narrow_w)}
    body = {"flex": 1.0, "blocks": _f_paragraphs(rng, corpus, assets, wide_w)}
    cols = [rail, body] if rng.random() < 0.5 else [body, rail]
    return {"type": "band", "gap": gap, "cols": cols, "rule": rng.random() < 0.12}


def _lay_composed(rng: random.Random, corpus, assets, page: dict) -> list[dict]:
    ls = _ls()
    page["engine"] = "composed"
    page["columns"] = 1                       # bands manage their own columns
    content_w = page["width"] - 2 * page["margin"]
    blocks: list[dict] = []

    # header: title (+ optional subtitle/caption), sometimes alongside a logo
    if rng.random() < 0.95:
        blocks.append({"type": "title", "text": ls._trim(corpus.sentence(), 60)})
        if rng.random() < 0.45:
            blocks.append({"type": "caption", "text": ls._trim(corpus.sentence(), 70),
                           "center": rng.random() < 0.5})

    n_bands = rng.randint(4, 8)
    # stratified band plan: guarantee >=1 multi-column text band and a rare-class quota
    # (the paper's ablation: element diversity AND layout diversity both matter)
    plan: list = ["text_multi"]
    if rng.random() < 0.70:
        plan.append("rare")                   # chart / formula / hand_drawing band
    if rng.random() < 0.40:
        plan.append("rare")
    if rng.random() < 0.40:
        plan.append("sidebar")
    while len(plan) < n_bands:
        plan.append("any")
    rng.shuffle(plan)

    for kind in plan:
        if kind == "sidebar":
            blocks.append(_sidebar_band(rng, corpus, assets, content_w))
            continue
        if kind == "rare":
            fac = _f_figure if rng.random() < 0.7 else _f_formula
            ncols = rng.randint(1, 2)
            gap = rng.randint(22, 40)
            cw = int((content_w - gap * (ncols - 1)) / ncols)
            cols = [{"flex": 1, "blocks": (fac(rng, corpus, assets, cw, btype="chart")
                                           if fac is _f_figure and rng.random() < 0.75
                                           else fac(rng, corpus, assets, cw))}
                    for _ in range(ncols)]
            blocks.append({"type": "band", "gap": gap, "cols": cols, "rule": False})
            continue
        if kind == "text_multi":
            ncols = rng.randint(2, 3)
            pool = [f for f in _FACTORIES if f[2] == "narrow"]
        else:                                  # "any"
            ncols = rng.choice([1, 1, 2, 2, 3])
            pool = _FACTORIES if ncols == 1 else [f for f in _FACTORIES
                                                  if ncols == 2 or f[2] == "narrow"]
        blocks.append(_band(rng, corpus, assets, content_w, ncols, pool))

    # footer: signatures row and/or page-number caption
    if rng.random() < 0.4:
        blocks.append(_band(rng, corpus, assets, content_w, rng.randint(1, 3),
                            [(_f_signatures, 1.0, "narrow")]))
    if rng.random() < 0.5:
        blocks.append({"type": "caption", "text": f"ទំព័រ {ls._kh_num(rng.randint(1, 99))}",
                       "center": True})
    return blocks
