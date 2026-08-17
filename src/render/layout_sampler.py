"""Sample a page: choose a LAYOUT TYPE, font, typography, and assemble corpus text + REAL images
into a page model consumed by render_playwright.

V3 redesign (vs V1/V2): layouts are spec-driven and diverse (one generator per documented layout in
docs/layouts/*.md), and figure/portrait/signature/stamp regions embed REAL images from an
`AssetPool` instead of CSS gradients — the gradient fakes were why the detector overfit. Each page
is tagged with `layout_type` so the dataset can be written to per-layout-type directories.

Text comes from the LM corpus (data/corpus/lm_corpus, else filtered/clean/raw); a built-in Khmer
pool keeps the render smoke test self-contained. Real images come from `AssetPool`; if a category is
empty the block falls back to a gradient so rendering still works.
"""
from __future__ import annotations

import random
from pathlib import Path

from ..corpus.common import (CLEAN_DIR, DATA, FILTERED_DIR, LM_DIR, RAW_DIR,
                             iter_jsonl_dir, load_yaml)
from .. import khmer_utils as ku

# fallback Khmer text so rendering works before any corpus is collected
_FALLBACK = [
    "ប្រទេសកម្ពុជាមានទីតាំងស្ថិតនៅភូមិភាគអាស៊ីអាគ្នេយ៍។",
    "ភាសាខ្មែរគឺជាភាសាផ្លូវការរបស់ប្រទេសកម្ពុជា។",
    "ទីក្រុងភ្នំពេញគឺជារាជធានីនៃព្រះរាជាណាចក្រកម្ពុជា។",
    "ប្រាសាទអង្គរវត្តគឺជាសំណង់ប្រវត្តិសាស្ត្រដ៏ល្បីល្បាញ។",
    "ការអប់រំមានសារៈសំខាន់សម្រាប់ការអភិវឌ្ឍន៍ប្រទេសជាតិ។",
    "សេដ្ឋកិច្ចកម្ពុជាកំពុងមានកំណើនជារៀងរាល់ឆ្នាំ។",
    "បច្ចេកវិទ្យាព័ត៌មានបានផ្លាស់ប្ដូររបៀបរស់នៅរបស់មនុស្ស។",
    "ទេសចរណ៍គឺជាវិស័យសំខាន់មួយក្នុងការរកចំណូលរបស់ប្រទេស។",
]

_LABELS = ["ឈ្មោះ", "អាសយដ្ឋាន", "ថ្ងៃខែឆ្នាំកំណើត", "លេខទូរស័ព្ទ", "ភេទ",
           "សញ្ជាតិ", "មុខរបរ", "លេខអត្តសញ្ញាណប័ណ្ណ"]
_VALUES = ["សុខ សំណាង", "ភ្នំពេញ", "០១/០១/២០០០", "០១២ ៣៤៥ ៦៧៨", "ប្រុស",
           "ខ្មែរ", "គ្រូបង្រៀន", "១២៣៤៥៦៧៨៩"]

_CARD_TITLES = ["ព្រះរាជាណាចក្រកម្ពុជា", "អត្តសញ្ញាណប័ណ្ណសញ្ជាតិខ្មែរ", "ប័ណ្ណបើកបរ",
                "លិខិតឆ្លងដែន", "ប័ណ្ណសម្គាល់និស្សិត", "ប័ណ្ណធានារ៉ាប់រង"]
_CARD_LABELS_BI = [("គោត្តនាម និងនាម", "Surname & Name"), ("ភេទ", "Sex"),
                   ("ថ្ងៃខែឆ្នាំកំណើត", "Date of Birth"), ("ទីកន្លែងកំណើត", "Place of Birth"),
                   ("កម្ពស់", "Height"), ("សញ្ជាតិ", "Nationality"), ("លេខសម្គាល់", "ID No"),
                   ("ថ្ងៃផុតកំណត់", "Expiry Date"), ("អាសយដ្ឋាន", "Address"),
                   ("ប្រភេទ", "Categories"), ("លេខបណ្ណ", "Card Code")]
_CARD_VALUES_EN = ["Cambodian", "AUTO", "B", "26-04-1999", "02-10-2023", "M", "B.PP.00323031",
                   "Phnom Penh", "165 cm"]
_KH_DIGITS = "០១២៣៤៥៦៧៨៩"
_KH_LETTERS = ["ក", "ខ", "គ", "ឃ", "ង", "ច", "ឆ", "ជ", "ឈ", "ញ"]
_DOT_LEADERS = ["........................", "…………………", ".............", "______________"]


def _kh_num(n: int) -> str:
    return "".join(_KH_DIGITS[int(c)] for c in str(n))


def _numbered(rng: random.Random, items: list[str]) -> list[str]:
    """Bake list numbering INTO the item text. CSS <ol>/<ul> markers sit OUTSIDE the Range line
    rects we label with, so marker-based numbering trained the detector to EXCLUDE leading
    numerals (observed on real worksheets: '១.' left out of every question box). The template now
    renders markerless lists and every numbering style lives in the text itself."""
    style = rng.random()
    if style < 0.40:
        return [f"{_kh_num(k + 1)}. {it}" for k, it in enumerate(items)]
    if style < 0.62:
        return [f"{k + 1}. {it}" for k, it in enumerate(items)]
    if style < 0.77:
        return [f"{_KH_LETTERS[k % len(_KH_LETTERS)]}) {it}" for k, it in enumerate(items)]
    if style < 0.88:
        # exam style: marker + parenthesised year, e.g. '១.(២០១៥)'. Real Khmer exam papers put a
        # wide hanging indent after this WHOLE group; see _NUM_PREFIX.
        yr = rng.randint(2010, 2025)
        return [f"{_kh_num(k + 1)}.({_kh_num(yr)}) {it}" for k, it in enumerate(items)]
    return [f"- {it}" for it in items]


def _trim(s: str, n: int) -> str:
    return s[:n].strip()


_PUNCT = set(" .,:;!?()[]{}<>«»\"'“”…-–—៚។៕៙‹›/\\|")


def _short(corpus: "CorpusText", rng: random.Random, n: int) -> str:
    """A short cell/value string with >=2 meaningful (non-punctuation) chars, so table cells/values
    aren't tiny pure-punctuation fragments (which made unrealistic 1px-wide boxes)."""
    for _ in range(5):
        s = _trim(corpus.sentence(), n)
        if sum(1 for ch in s if ch not in _PUNCT and not ch.isspace()) >= 2:
            return s
    return s or "ខ្មែរ"


def _mrz(rng: random.Random) -> str:
    a = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(rng.randint(3, 6)))
    b = "".join(rng.choice("0123456789") for _ in range(rng.randint(6, 9)))
    return f"KHM<{a}<<{b}<<<{rng.randint(0,9)}{rng.randint(0,9)}"


def _formula(rng: random.Random) -> str:
    v = rng.choice("xyzabnt")
    w = rng.choice("ijkmpq")
    forms = ["E = mc²", "a² + b² = c²",
             f"f({v}) = {rng.randint(2,9)}{v}² + {rng.randint(1,9)}{v} − {rng.randint(1,9)}",
             f"∑({v}={rng.randint(0,3)} → {rng.randint(4,9)}) {v}ᵢ",
             f"{v} = (−{w} ± √({w}² − 4ac)) / 2a",
             f"∫₀¹ {v}² d{v} = 1/3", f"P({v}) = {rng.randint(1,9)}/{rng.randint(10,99)}"]
    return rng.choice(forms)


def _fallback_bg(rng: random.Random) -> str:
    h1, h2 = rng.randint(0, 360), rng.randint(0, 360)
    return f"linear-gradient({rng.choice([0,45,90,135])}deg, hsl({h1},40%,72%), hsl({h2},40%,46%))"


def _img_block(assets, rng: random.Random, categories: list[str], w: int, h: int,
               btype: str = "image", float: str | None = None, round: bool = False) -> dict:
    """A real-image block (image/signature/hand_drawing). Falls back to a gradient if the asset
    pool has nothing for the requested categories (keeps rendering working before a fetch)."""
    blk: dict = {"type": btype, "w": int(w), "h": int(h)}
    src = assets.get_any(categories) if assets is not None else None
    if src is not None:
        blk["src"] = str(src)
    else:
        blk["bg"] = _fallback_bg(rng)
    if float:
        blk["float"] = float
    if round:
        blk["round"] = True
    return blk


class CorpusText:
    """Yields normalized Khmer sentences/paragraphs from the best available corpus stage."""

    def __init__(self, seed: int = 0, quiet: bool = False, max_docs: int = 20000):
        self.rng = random.Random(seed)
        self.pool: list[str] = []
        for d in (LM_DIR, FILTERED_DIR, CLEAN_DIR, RAW_DIR):
            if d.exists() and any(d.glob("*.jsonl")):
                for row in iter_jsonl_dir(d):
                    t = row.get("text", "")
                    if t:
                        self.pool.append(t)
                    if len(self.pool) >= max_docs:
                        break
                if self.pool:
                    if not quiet:
                        print(f"[sampler] text pool from {d.name}: {len(self.pool)} docs")
                    break
        if not self.pool:
            if not quiet:
                print("[sampler] no corpus found; using built-in fallback Khmer pool")
            self.pool = list(_FALLBACK)

    def sentence(self) -> str:
        doc = self.rng.choice(self.pool)
        sents = ku.split_sentences(doc) or [doc]
        return self.rng.choice(sents)

    def paragraph(self, n_sent: int) -> str:
        return "".join(self.sentence() for _ in range(n_sent))


def _weighted_choice(rng: random.Random, items: list, weights: list[float]):
    return rng.choices(items, weights=weights, k=1)[0]


# ----------------------------------------------------------------- shared block builders

def _content_width(page: dict) -> int:
    cols = page.get("columns", 1)
    inner = page["width"] - 2 * page["margin"]
    return int(inner / cols) if cols > 1 else inner


def _figure(rng, corpus, assets, cw, cats=("photos", "charts"), btype="image",
            caption=True, cap_prefix="រូបភាព៖ ") -> list[dict]:
    # split chart out from generic image (its own detector class) when a chart asset is available
    if btype == "image" and "charts" in cats and assets is not None and assets.has("charts") \
            and rng.random() < 0.5:
        btype, cats = "chart", ["charts"]
    # Only SMALL images float (text wraps beside a thumbnail, like real articles); wide figures are
    # block-level (centered, text above/below) so adjacent text/headings never squeeze into a tiny
    # column beside a big picture.
    flt = rng.choice([None, None, None, "right", "left"])
    if flt:
        w = int(cw * rng.uniform(0.26, 0.42)); h = rng.randint(140, 280)
    else:
        w = int(cw * rng.uniform(0.5, 0.95)); h = rng.randint(160, 360)
    out = [_img_block(assets, rng, list(cats), w, h, btype=btype, float=flt)]
    if caption:
        out.append({"type": "caption", "text": cap_prefix + _trim(corpus.sentence(), 40),
                    "center": True})
    return out


def _table(rng, corpus, tid=0, rows_rng=(3, 8), cols_rng=(2, 5), header=True,
           numeric_last=False) -> dict:
    n_cols = rng.randint(*cols_rng)
    n_rows = rng.randint(*rows_rng)
    hdr = [_short(corpus, rng, 12) for _ in range(n_cols)] if header else None

    def cell(ci):
        if numeric_last and ci == n_cols - 1:
            return _kh_num(rng.randint(1, 9999))
        return _short(corpus, rng, 14)
    rows = [[cell(ci) for ci in range(n_cols)] for _ in range(n_rows)]
    return {"type": "table", "tid": tid, "header": hdr, "rows": rows}


def _body(rng, corpus, assets, cw, n_par_rng=(3, 9), fig_prob=0.25, sub_prob=0.35,
          fig_cats=("photos", "charts")) -> list[dict]:
    blocks: list[dict] = []
    for i in range(rng.randint(*n_par_rng)):
        if i > 0 and rng.random() < sub_prob:
            blocks.append({"type": "heading", "text": _trim(corpus.sentence(), 34)})
            if rng.random() < 0.4:
                blocks.append({"type": "subheading", "text": _trim(corpus.sentence(), 28)})
        blocks.append({"type": "text", "text": corpus.paragraph(rng.randint(2, 6))})
        if rng.random() < fig_prob:
            blocks += _figure(rng, corpus, assets, cw, cats=fig_cats)
    return blocks


# ----------------------------------------------------------------- layout generators
# each returns a block list and may set page geometry overrides on `page`.

def _lay_news_article(rng, corpus, assets, page):
    page["columns"] = 2 if rng.random() < 0.3 else 1
    cw = _content_width(page)
    b = [{"type": "title", "text": _trim(corpus.sentence(), 60)},
         {"type": "caption", "text": "ដោយ អ្នកនិពន្ធ · " + rng.choice(["ភ្នំពេញប៉ុស្តិ៍", "ព័ត៌មាន", "សារព័ត៌មាន"])}]
    b += _body(rng, corpus, assets, cw, n_par_rng=(3, 9), fig_prob=0.28)
    if rng.random() < 0.3:
        b.append({"type": "list", "items": _numbered(rng, [_trim(corpus.sentence(), 40) for _ in range(rng.randint(3, 5))])})
    return b


def _lay_magazine_multicol(rng, corpus, assets, page):
    page["columns"] = rng.choice([2, 2, 3])
    page["margin"] = rng.randint(50, 90)
    cw = _content_width(page)
    b = [{"type": "title", "text": _trim(corpus.sentence(), 50)},
         {"type": "caption", "text": rng.choice(["ព័ត៌មានពិសេស", "អត្ថបទ", "បទយកការណ៍"])}]
    b += _body(rng, corpus, assets, cw, n_par_rng=(5, 11), fig_prob=0.35, sub_prob=0.3)
    return b


def _lay_scientific_paper(rng, corpus, assets, page):
    page["columns"] = 2 if rng.random() < 0.4 else 1
    cw = _content_width(page)
    b = [{"type": "title", "text": _trim(corpus.sentence(), 70), },
         {"type": "caption", "text": "អ្នកនិពន្ធ · " + rng.choice(["សាកលវិទ្យាល័យ", "វិទ្យាស្ថាន"])},
         {"type": "heading", "text": "អរូបី"},
         {"type": "text", "text": corpus.paragraph(rng.randint(3, 5))}]
    for s in range(rng.randint(2, 4)):
        b.append({"type": "heading", "text": f"{_kh_num(s + 1)}. " + _trim(corpus.sentence(), 30)})
        for ss in range(rng.randint(1, 3)):
            if rng.random() < 0.5:
                b.append({"type": "subheading", "text": f"{_kh_num(s + 1)}.{_kh_num(ss + 1)} " + _trim(corpus.sentence(), 24)})
            b.append({"type": "text", "text": corpus.paragraph(rng.randint(2, 4))})
            if rng.random() < 0.4:
                b.append({"type": "formula", "text": _formula(rng)})
        if rng.random() < 0.45:
            b += _figure(rng, corpus, assets, cw, cats=("charts", "drawings", "photos"),
                         cap_prefix="រូបទី " + _kh_num(s + 1) + "៖ ")
    if rng.random() < 0.5:
        b.append(_table(rng, corpus, numeric_last=True))
    b.append({"type": "heading", "text": "ឯកសារយោង"})
    b.append({"type": "list", "items": _numbered(rng, [_trim(corpus.sentence(), 60) for _ in range(rng.randint(3, 6))])})
    return b


def _lay_business_report(rng, corpus, assets, page):
    cw = _content_width(page)
    b = [{"type": "heading", "text": _trim(corpus.sentence(), 40)}]
    for _ in range(rng.randint(2, 4)):
        if rng.random() < 0.5:
            b.append({"type": "subheading", "text": _trim(corpus.sentence(), 28)})
        b.append({"type": "text", "text": corpus.paragraph(rng.randint(2, 5))})
    for _ in range(rng.randint(1, 2)):
        b += _figure(rng, corpus, assets, cw, cats=("charts", "photos"), cap_prefix="តារាង/ក្រាហ្វិក៖ ")
    for _ in range(rng.randint(1, 2)):
        b.append(_table(rng, corpus, numeric_last=True, rows_rng=(3, 7)))
    if rng.random() < 0.5:
        b.append({"type": "form", "fields": [{"label": "ចំណូលសរុប", "value": _kh_num(rng.randint(1000, 999999)) + " ៛"}]})
    return b


def _lay_financial_statement(rng, corpus, assets, page):
    page["margin"] = rng.randint(50, 90)
    page["body_px"] = rng.randint(13, 17)
    b = [{"type": "title", "text": _trim(corpus.sentence(), 40)},
         {"type": "caption", "text": "សម្រាប់ការិយបរិច្ឆេទ " + _kh_num(rng.randint(2018, 2026))}]
    for t in range(rng.randint(1, 3)):
        if rng.random() < 0.6:
            b.append({"type": "subheading", "text": _trim(corpus.sentence(), 26)})
        b.append(_table(rng, corpus, tid=t, rows_rng=(4, 14), cols_rng=(2, 6), numeric_last=True))
        if rng.random() < 0.4:
            b.append({"type": "text", "text": corpus.paragraph(rng.randint(1, 2))})
    return b


def _lay_form_generic(rng, corpus, assets, page):
    cw = _content_width(page)
    page["dotted_values"] = True
    b = []
    if rng.random() < 0.5:
        b.append(_img_block(assets, rng, ["logos"], rng.randint(70, 130), rng.randint(50, 90), float="left"))
    b.append({"type": "title", "text": rng.choice(["ពាក្យសុំ", "ទម្រង់ព័ត៌មាន", "សំណើ"]) + " " + _trim(corpus.sentence(), 20)})
    if rng.random() < 0.5:
        b.append({"type": "text", "text": corpus.paragraph(1)})
    idx = list(range(len(_LABELS))); rng.shuffle(idx)
    for i in idx[:rng.randint(5, 8)]:
        b.append({"type": "form", "fields": [{"label": _LABELS[i % len(_LABELS)], "value": rng.choice(_VALUES + _CARD_VALUES_EN)}]})
    if rng.random() < 0.4:
        b.append(_table(rng, corpus, rows_rng=(2, 4)))
    for _ in range(rng.randint(0, 3)):
        b.append({"type": "list", "items": ["☐ " + _trim(corpus.sentence(), 24)]})
    b.append({"type": "form", "fields": [{"label": "ហត្ថលេខា", "value": ""}]})
    b.append(_img_block(assets, rng, ["signatures"], rng.randint(150, 240), rng.randint(50, 90), btype="signature", float="right"))
    if rng.random() < 0.4:
        b.append(_img_block(assets, rng, ["stamps", "logos"], rng.randint(110, 160), rng.randint(110, 160)))
    return b


def _lay_id_card(rng, corpus, assets, page):
    page["width"] = rng.randint(980, 1080)
    page["margin"] = rng.randint(24, 48)
    page["columns"] = 1
    page["page_bg"] = rng.choice(["#eaf2fb", "#f3eee4", "#e9f5ee", "#fdeeee", "#ffffff"])
    b = []
    if rng.random() < 0.9:
        b.append(_img_block(assets, rng, ["logos"], rng.randint(60, 110), rng.randint(45, 80), float="left"))
    km_t = rng.choice(_CARD_TITLES)
    b.append({"type": "title", "text": km_t + (" KINGDOM OF CAMBODIA" if rng.random() < 0.5 else "")})
    b.append(_img_block(assets, rng, ["portraits"], rng.randint(150, 230), rng.randint(190, 280), float="left"))
    idx = list(range(len(_CARD_LABELS_BI))); rng.shuffle(idx)
    for i in idx[:rng.randint(5, 9)]:
        km, en = _CARD_LABELS_BI[i]
        lab = f"{km} {en}" if rng.random() < 0.7 else km
        val = rng.choice(_CARD_VALUES_EN + _VALUES + [_kh_num(rng.randint(100000, 999999))])
        b.append({"type": "form", "fields": [{"label": lab, "value": val}]})
    if rng.random() < 0.75:
        b.append(_img_block(assets, rng, ["signatures"], rng.randint(120, 200), rng.randint(40, 70), btype="signature"))
    if rng.random() < 0.7:
        s = rng.randint(90, 150)
        b.append(_img_block(assets, rng, ["charts", "logos"], s, s, float="right"))  # QR-ish
    b.append({"type": "text", "text": "ID: " + _kh_num(rng.randint(100000000, 999999999))})
    b.append({"type": "text", "text": _mrz(rng) + " " + _mrz(rng)})
    return b


def _lay_passport_mrz(rng, corpus, assets, page):
    page["width"] = rng.randint(980, 1060)
    page["margin"] = rng.randint(30, 60)
    page["page_bg"] = rng.choice(["#eef2f7", "#f5f0e6", "#eef5f0"])
    b = [{"type": "title", "text": "លិខិតឆ្លងដែន / PASSPORT — " + rng.choice(["KHM", "CAMBODIA"])}]
    b.append(_img_block(assets, rng, ["portraits"], rng.randint(160, 230), rng.randint(200, 290), float="left"))
    pp_labels = [("ប្រភេទ", "Type"), ("លេខលិខិត", "Passport No"), ("គោត្តនាម", "Surname"),
                 ("នាម", "Given names"), ("សញ្ជាតិ", "Nationality"), ("ថ្ងៃខែឆ្នាំកំណើត", "Date of birth"),
                 ("ភេទ", "Sex"), ("ទីកន្លែងកំណើត", "Place of birth"), ("ថ្ងៃផុតកំណត់", "Date of expiry")]
    idx = list(range(len(pp_labels))); rng.shuffle(idx)
    for i in idx[:rng.randint(6, 9)]:
        km, en = pp_labels[i]
        b.append({"type": "form", "fields": [{"label": f"{km} {en}", "value": rng.choice(_CARD_VALUES_EN + _VALUES)}]})
    if rng.random() < 0.7:
        b.append(_img_block(assets, rng, ["signatures"], rng.randint(140, 200), rng.randint(40, 70), btype="signature"))
    b.append({"type": "text", "text": "P<KHM" + "".join(rng.choice("ABCDEFGHIJKLMNOP") for _ in range(20)) + "<<<<<"})
    b.append({"type": "text", "text": "".join(rng.choice("0123456789KHM<") for _ in range(40))})
    return b


def _lay_book_page(rng, corpus, assets, page):
    page["width"] = rng.randint(1000, 1120)
    page["margin"] = rng.randint(90, 150)
    page["columns"] = 2 if rng.random() < 0.25 else 1
    page["align"] = rng.choice(["left", "justify"])
    cw = _content_width(page)
    b = []
    if rng.random() < 0.4:
        b.append({"type": "caption", "text": _trim(corpus.sentence(), 30)})  # running header
    if rng.random() < 0.85:
        b.append({"type": "heading", "text": "ជំពូកទី " + _kh_num(rng.randint(1, 12))})
        b.append({"type": "title", "text": _trim(corpus.sentence(), 40)})
    for i in range(rng.randint(4, 9)):
        if i > 0 and rng.random() < 0.22:
            b.append({"type": "subheading", "text": _trim(corpus.sentence(), 30)})
        b.append({"type": "text", "text": corpus.paragraph(rng.randint(3, 6))})
        if rng.random() < 0.18:
            b += _figure(rng, corpus, assets, cw, cats=("photos", "drawings"),
                         cap_prefix="រូបទី " + _kh_num(rng.randint(1, 20)) + "៖ ")
    b.append({"type": "text", "text": "− " + _kh_num(rng.randint(1, 350)) + " −"})
    return b


def _lay_textbook_figures(rng, corpus, assets, page):
    page["columns"] = 2 if rng.random() < 0.35 else 1
    cw = _content_width(page)
    b = [{"type": "heading", "text": _trim(corpus.sentence(), 36)}]
    if rng.random() < 0.5:
        b.append({"type": "subheading", "text": _trim(corpus.sentence(), 28)})
    for i in range(rng.randint(3, 6)):
        b.append({"type": "text", "text": corpus.paragraph(rng.randint(2, 4))})
        if rng.random() < 0.55:
            r = rng.random()
            if r < 0.4:
                bt, cats = "hand_drawing", ["drawings"]
            elif r < 0.7 and assets is not None and assets.has("charts"):
                bt, cats = "chart", ["charts"]
            else:
                bt, cats = "image", ["photos", "charts"]
            flt = rng.choice([None, None, "right", "left"])
            if flt:
                w = int(cw * rng.uniform(0.26, 0.42)); h = rng.randint(140, 250)
            else:
                w = int(cw * rng.uniform(0.5, 0.85)); h = rng.randint(150, 320)
            b.append(_img_block(assets, rng, cats, w, h, btype=bt, float=flt))
            b.append({"type": "caption", "text": "រូបទី " + _kh_num(i + 1) + "៖ " + _trim(corpus.sentence(), 30), "center": True})
        if rng.random() < 0.35:
            b.append({"type": "formula", "text": _formula(rng)})
    if rng.random() < 0.4:
        b.append(_table(rng, corpus, numeric_last=True))
    if rng.random() < 0.4:
        b.append({"type": "list", "items": [f"{_kh_num(k + 1)}. " + _trim(corpus.sentence(), 36) for k in range(rng.randint(2, 4))]})
    return b


def _lay_worksheet(rng, corpus, assets, page):
    page["body_px"] = rng.randint(18, 26)
    cw = _content_width(page)
    b = [{"type": "title", "text": rng.choice(["លំហាត់", "សំណួរ", "កិច្ចការផ្ទះ", "មេរៀនទី " + _kh_num(rng.randint(1, 30))])}]
    for li in range(rng.randint(2, 4)):
        b.append({"type": "heading", "text": _KH_LETTERS[li] + ". " + _trim(corpus.sentence(), 34)})
        if rng.random() < 0.55:
            words = [_trim(corpus.sentence(), 8) for _ in range(rng.randint(4, 9))]
            if rng.random() < 0.5:
                # single-container word bank (the real-worksheet pattern: one rounded rectangle
                # holding the whole word row) — wide word spacing, squarer corners
                b.append({"type": "bubbles", "items": [" ".join(words)], "justify": "center",
                          "item_type": "word_bank",
                          "word_spacing": rng.randint(10, 30), "radius": rng.randint(6, 26),
                          "border_px": round(rng.uniform(1.2, 2.4), 1)})
            else:
                b.append({"type": "bubbles", "items": words,
                          "radius": rng.randint(12, 26)})
        for q in range(rng.randint(2, 4)):
            b.append({"type": "text", "text": f"{_kh_num(q + 1)}. {_trim(corpus.sentence(), 48)} {rng.choice(_DOT_LEADERS)} ។"})
    if rng.random() < 0.55:
        for k in range(rng.randint(1, 2)):
            bt = "image" if rng.random() < 0.6 else "hand_drawing"
            cats = ["photos"] if bt == "image" else ["drawings"]
            b.append(_img_block(assets, rng, cats, rng.randint(160, 300), rng.randint(120, 230), btype=bt, float="left"))
            b.append({"type": "caption", "text": f"{_kh_num(k + 1)}. " + _trim(corpus.sentence(), 16) + " " + rng.choice(_DOT_LEADERS) + " ។"})
    return b


def _lay_exam_paper(rng, corpus, assets, page):
    cw = _content_width(page)
    b = [{"type": "title", "text": rng.choice(["ប្រឡងសញ្ញាបត្រ", "វិញ្ញាសា", "តេស្តពិន្ទុ"]) + " " + _trim(corpus.sentence(), 18)},
         {"type": "subheading", "text": "មុខវិជ្ជា៖ " + _trim(corpus.sentence(), 16) + " · រយៈពេល " + _kh_num(rng.randint(1, 3)) + " ម៉ោង · ពិន្ទុសរុប " + _kh_num(rng.choice([50, 100]))}]
    if rng.random() < 0.6:
        for lab in ("ឈ្មោះ", "លេខសម្គាល់", "ថ្នាក់"):
            b.append({"type": "form", "fields": [{"label": lab, "value": "".join(["."] * 18)}]})
    for s in range(rng.randint(1, 3)):
        b.append({"type": "heading", "text": "ផ្នែកទី " + _kh_num(s + 1) + " (" + _kh_num(rng.randint(10, 40)) + " ពិន្ទុ)"})
        for q in range(rng.randint(2, 4)):
            b.append({"type": "text", "text": f"{_kh_num(q + 1)}. {_trim(corpus.sentence(), 50)} ({_kh_num(rng.randint(1, 10))} ពិន្ទុ)"})
            if rng.random() < 0.5:
                b.append({"type": "list", "items": [f"{_KH_LETTERS[c]}) " + _trim(corpus.sentence(), 24) for c in range(rng.randint(3, 4))]})
            elif rng.random() < 0.3:
                b += _figure(rng, corpus, assets, cw, cats=("drawings", "charts"), cap_prefix="រូប៖ ")
    return b


def _lay_receipt_invoice(rng, corpus, assets, page):
    receipt = rng.random() < 0.5
    page["width"] = rng.randint(700, 900) if receipt else rng.randint(1040, 1140)
    page["margin"] = rng.randint(30, 70)
    page["align"] = "left"
    b = []
    if rng.random() < 0.6:
        b.append(_img_block(assets, rng, ["logos"], rng.randint(80, 150), rng.randint(60, 100), float="left"))
    b.append({"type": "title", "text": rng.choice(["វិក្កយបត្រ", "បង្កាន់ដៃ", "ហាងលក់ទំនិញ"]) + " " + _trim(corpus.sentence(), 14)})
    for lab in ("លេខវិក្កយបត្រ", "កាលបរិច្ឆេទ", "អតិថិជន"):
        b.append({"type": "form", "fields": [{"label": lab, "value": rng.choice(_VALUES)}]})
    n_rows = rng.randint(3, 9)
    header = ["ទំនិញ", "ចំនួន", "តម្លៃ"]
    rows = [[_short(corpus, rng, 12), _kh_num(rng.randint(1, 20)), _kh_num(rng.randint(1, 99)) + "០០ ៛"] for _ in range(n_rows)]
    b.append({"type": "table", "tid": 0, "header": header, "rows": rows})
    b.append({"type": "form", "fields": [{"label": "សរុប", "value": _kh_num(rng.randint(10, 999)) + "០០ ៛"}]})
    if rng.random() < 0.5:
        b.append(_img_block(assets, rng, ["stamps", "logos"], rng.randint(100, 150), rng.randint(100, 150), float="right"))
    b.append({"type": "caption", "text": "សូមអរគុណ · Thank you", "center": True})
    return b


def _lay_certificate(rng, corpus, assets, page):
    if rng.random() < 0.6:
        page["width"] = rng.randint(1240, 1360)
    page["margin"] = rng.randint(90, 160)
    page["align"] = "center"
    page["page_bg"] = rng.choice(["#fdfaf0", "#fbf6e8", "#f7f3ff", "#ffffff"])
    page["page_border"] = rng.choice(["10px double #b5933f", "8px solid #7a6a3a", "6px double #4a5a8a"])
    b = []
    if rng.random() < 0.6:
        b.append(_img_block(assets, rng, ["logos"], rng.randint(90, 150), rng.randint(90, 150)))
    b.append({"type": "title", "text": rng.choice(["វិញ្ញាបនបត្រ", "សញ្ញាបត្រ", "លិខិតសរសើរ"])})
    b.append({"type": "subheading", "text": _trim(corpus.sentence(), 36)})
    for _ in range(rng.randint(2, 3)):
        b.append({"type": "text", "text": corpus.paragraph(rng.randint(2, 3))})
    n_sig = rng.randint(1, 2)
    for k in range(n_sig):
        flt = "left" if k == 0 else "right"
        b.append(_img_block(assets, rng, ["signatures"], rng.randint(140, 210), rng.randint(50, 90), btype="signature", float=flt))
        b.append({"type": "caption", "text": _trim(corpus.sentence(), 20)})
    if rng.random() < 0.8:
        s = rng.randint(110, 170)
        b.append(_img_block(assets, rng, ["stamps", "logos"], s, s))
    return b


def _lay_letter_memo(rng, corpus, assets, page):
    page["align"] = "left"
    b = []
    if rng.random() < 0.7:
        b.append(_img_block(assets, rng, ["logos"], rng.randint(80, 140), rng.randint(60, 100), float="left"))
    b.append({"type": "title", "text": _trim(corpus.sentence(), 36)})
    b.append({"type": "caption", "text": _trim(corpus.sentence(), 40)})
    for lab in ("លេខ", "កាលបរិច្ឆេទ"):
        b.append({"type": "form", "fields": [{"label": lab, "value": rng.choice(_VALUES)}]})
    b.append({"type": "text", "text": "ស្ដីពី៖ " + _trim(corpus.sentence(), 40)})
    b.append({"type": "text", "text": "ជូនចំពោះ៖ " + _trim(corpus.sentence(), 30)})
    for _ in range(rng.randint(2, 5)):
        b.append({"type": "text", "text": corpus.paragraph(rng.randint(2, 4))})
    b.append({"type": "text", "text": "សូមគោរព"})
    b.append(_img_block(assets, rng, ["signatures"], rng.randint(150, 230), rng.randint(50, 90), btype="signature", float="right"))
    b.append({"type": "caption", "text": _trim(corpus.sentence(), 24)})
    if rng.random() < 0.4:
        b.append(_img_block(assets, rng, ["stamps"], rng.randint(100, 150), rng.randint(100, 150), float="right"))
    return b


def _lay_contract_legal(rng, corpus, assets, page):
    page["align"] = rng.choice(["left", "justify"])
    b = [{"type": "title", "text": _trim(corpus.sentence(), 50)},
         {"type": "caption", "text": "លេខ " + _kh_num(rng.randint(1, 999)) + " · " + _kh_num(rng.randint(2018, 2026))}]
    for a in range(rng.randint(2, 5)):
        b.append({"type": "heading", "text": "មាត្រា " + _kh_num(a + 1) + "៖ " + _trim(corpus.sentence(), 26)})
        if rng.random() < 0.6:
            b.append({"type": "text", "text": corpus.paragraph(rng.randint(1, 3))})
        if rng.random() < 0.6:
            b.append({"type": "list", "items": [f"{_kh_num(a + 1)}.{_kh_num(k + 1)} " + _trim(corpus.sentence(), 44) for k in range(rng.randint(2, 4))]})
    b.append({"type": "text", "text": "ធ្វើនៅ " + rng.choice(["ភ្នំពេញ", "សៀមរាប"]) + " ថ្ងៃទី " + _kh_num(rng.randint(1, 28))})
    n_sig = rng.randint(1, 3)
    for k in range(n_sig):
        b.append(_img_block(assets, rng, ["signatures"], rng.randint(140, 210), rng.randint(50, 90), btype="signature", float="left" if k % 2 == 0 else "right"))
        b.append({"type": "caption", "text": _trim(corpus.sentence(), 20)})
    if rng.random() < 0.7:
        s = rng.randint(110, 160)
        b.append(_img_block(assets, rng, ["stamps", "logos"], s, s, float="right"))
    return b


_LAYOUTS = {
    "news_article": _lay_news_article,
    "magazine_multicol": _lay_magazine_multicol,
    "scientific_paper": _lay_scientific_paper,
    "business_report": _lay_business_report,
    "financial_statement": _lay_financial_statement,
    "form_generic": _lay_form_generic,
    "id_card": _lay_id_card,
    "passport_mrz": _lay_passport_mrz,
    "book_page": _lay_book_page,
    "textbook_figures": _lay_textbook_figures,
    "worksheet": _lay_worksheet,
    "exam_paper": _lay_exam_paper,
    "receipt_invoice": _lay_receipt_invoice,
    "certificate": _lay_certificate,
    "letter_memo": _lay_letter_memo,
    "contract_legal": _lay_contract_legal,
}


# layouts whose geometry is too rigid for random rare-class block insertion ("composed" balances
# its own classes by construction — see compose_engine.py)
_NO_INSERT = {"id_card", "passport_mrz", "composed"}
# A numbering prefix is the numeral/letter marker PLUS an optional parenthetical qualifier —
# real exam papers write '១.(២០១៥)' (question number + the year it was set) and put the hanging
# indent after the WHOLE group. Matching only '១.' put the gap mid-group, so the detector never saw
# a wide-gap prefix longer than two glyphs and cut real ones off at the parenthesis.
_NUM_PREFIX = __import__("re").compile(
    r"^(?:(?:[០-៩]+|\d+)[.)]|[ក-អ][.)])(?:\s*\([^()]{1,14}\))?")
_LATIN_SNIPPETS = ["Tel: 012 345 678", "E-mail: info@example.com.kh", "USD 25.50",
                   "No. 124/25", "01/02/2024", "Lot 88, St. 271", "ID: KH-2024-0917"]


def _ruled_bg(rng: random.Random, base: str) -> str:
    """Exercise-book ruled (or grid) paper as a CSS background."""
    gap = rng.randint(30, 44)
    line = rng.choice(["#b8cce0", "#c4d4e4", "#d9c6c6"])
    ruled = (f"repeating-linear-gradient(transparent, transparent {gap - 1}px, "
             f"{line} {gap - 1}px, {line} {gap}px)")
    if rng.random() < 0.3:                       # grid (math notebook)
        ruled += (f", repeating-linear-gradient(90deg, transparent, transparent {gap - 1}px, "
                  f"{line} {gap - 1}px, {line} {gap}px)")
    return f"{ruled} {base}"


def _emph_partition(rng: random.Random, text: str) -> tuple[str, str, str] | None:
    """Split text into (pre, emph, post) at SPACE boundaries only — an element boundary inside a
    Khmer cluster breaks HarfBuzz shaping (dotted circles), so spaceless text is left alone.
    Concatenation is exactly `text`, keeping the round-trip byte-identical."""
    sp = [i for i, ch in enumerate(text) if ch == " "]
    if len(sp) < 2:
        return None
    i = rng.randrange(len(sp) - 1)
    j = rng.randrange(i + 1, min(len(sp), i + 4))
    a, b = sp[i] + 1, sp[j]
    if b - a < 3:
        return None
    return text[:a], text[a:b], text[b:]


def _apply_style_pass(rng: random.Random, page: dict, blocks: list[dict], corpus: CorpusText,
                      assets, style: dict, fonts: list[dict]) -> None:
    """V5 style generalization: emphasis runs, mixed Khmer/Latin content, rare-class inserts."""
    emph_p = style.get("emph_prob", 0.0)
    mixed_p = style.get("mixed_content_prob", 0.0)
    for b in iter_blocks(blocks):                  # walks into compose-engine bands too
        if b.get("type") != "text" or len(b.get("text", "")) < 40:
            continue
        if mixed_p and rng.random() < mixed_p and " " in b["text"]:
            snippet = rng.choice(_LATIN_SNIPPETS if rng.random() < 0.6
                                 else [f"ថ្ងៃទី{_kh_num(rng.randint(1, 28))} {_kh_num(rng.randint(2015, 2025))}",
                                       f"លេខ {_kh_num(rng.randint(100, 9999))}"])
            cut = b["text"].rindex(" ")
            b["text"] = f"{b['text'][:cut]} {snippet}{b['text'][cut:]}"
        # hanging-indent numbering: real worksheets/exams set a WIDE gap between '១.' and the
        # question text; single-space synthetic numbering made the detector cut numerals off.
        # Rendered as an empty inline spacer (textContent unchanged; rect-union keeps one line box).
        m = _NUM_PREFIX.match(b["text"])
        if m and rng.random() < 0.5:
            b["pre"], b["post"] = b["text"][:m.end()], b["text"][m.end():]
            b["numgap"] = rng.randint(10, 46)
            continue
        if emph_p and rng.random() < emph_p:
            part = _emph_partition(rng, b["text"])
            if part:
                b["pre"], b["emph"], b["post"] = part
                b["emph_style"] = rng.choice(["bold", "underline", "mark"])

    if page["layout_type"] in _NO_INSERT:
        return
    ins = style.get("insert", {})
    extras: list[dict] = []
    if rng.random() < ins.get("formula_prob", 0.0):
        for _ in range(rng.randint(1, 2)):
            extras.append({"type": "formula", "text": _formula(rng)})
    if rng.random() < ins.get("chart_prob", 0.0):
        extras.append(_img_block(assets, rng, ["charts"], rng.randint(360, 620),
                                 rng.randint(220, 380), btype="chart"))
        extras.append({"type": "caption", "text": _trim(corpus.sentence(), 90), "center": True})
    if rng.random() < ins.get("image_prob", 0.0):
        extras.append(_img_block(assets, rng, ["photos"], rng.randint(320, 560),
                                 rng.randint(220, 380), btype="image"))
    if rng.random() < ins.get("hand_drawing_prob", 0.0):
        extras.append(_img_block(assets, rng, ["drawings"], rng.randint(260, 480),
                                 rng.randint(200, 340), btype="hand_drawing"))
    if rng.random() < ins.get("signature_prob", 0.0):
        for _ in range(rng.randint(1, 2)):
            extras.append(_img_block(assets, rng, ["signatures"], rng.randint(150, 240),
                                     rng.randint(60, 110), btype="signature"))
    if rng.random() < ins.get("subheading_prob", 0.0):
        extras.append({"type": "subheading", "text": _trim(corpus.sentence(), 70)})
    for e in extras:
        blocks.insert(rng.randint(1, max(1, len(blocks) - 1)), e)


def sample_page(rng: random.Random, corpus: CorpusText, layouts: dict,
                fonts: list[dict], assets=None) -> dict:
    tmpls = layouts["templates"]
    names = [t["name"] for t in tmpls if t["name"] in _LAYOUTS]
    weights = [t["weight"] for t in tmpls if t["name"] in _LAYOUTS]
    layout_type = _weighted_choice(rng, names, weights)
    font = _weighted_choice(rng, fonts, [f.get("weight", 1.0) for f in fonts])
    style = layouts.get("style", {})

    typ = layouts["typography"]
    body_px = rng.randint(*typ["body_font_px"])
    scale = rng.uniform(*typ["heading_scale"])
    pg = layouts["page"]
    margin = rng.randint(*pg["margin_px"])
    white = rng.random() < pg["background"]["white_prob"]
    page_bg = "#ffffff" if white else rng.choice(["#fbf7ee", "#f4f4f4", "#fcfbf7"])
    if rng.random() < style.get("ruled_paper_prob", 0.0):
        page_bg = _ruled_bg(rng, page_bg)

    # page geometry: width jitter + occasional landscape + narrow receipts (V5)
    wj = style.get("width_jitter")
    width = rng.randint(*wj) if wj else pg["width_px"]
    if rng.random() < style.get("landscape_prob", 0.0):
        width = rng.randint(1500, 1760)
    if layout_type == "receipt_invoice" and rng.random() < style.get("receipt_narrow_prob", 0.0):
        width = rng.randint(560, 780)

    # ink color: mostly dark, sometimes colored (textbooks/exams print in blue/red/green)
    if rng.random() < style.get("colored_text_prob", 0.0) and style.get("text_palette"):
        color = rng.choice(style["text_palette"])
    else:
        color = rng.choice(typ["text_color_dark"])

    # font pairing: independent title/heading font (real Khmer docs pair Muol-style + body font)
    title_font = font
    if len(fonts) > 1 and rng.random() < style.get("title_font_prob", 0.0):
        others = [f for f in fonts if f["name"] != font["name"]]
        title_font = _weighted_choice(rng, others, [f.get("weight", 1.0) for f in others])

    page = {
        "layout_type": layout_type,
        "template": layout_type,                 # back-compat alias
        "font_name": font["name"],
        "font_path": str((DATA.parent / font["path"]).resolve()) if "path" in font else None,
        "title_font_name": title_font["name"],
        "title_font_path": (str((DATA.parent / title_font["path"]).resolve())
                            if "path" in title_font else None),
        "width": width,
        "margin": margin,
        "bg": "#dcdcdc" if not white else "#ffffff",
        "page_bg": page_bg,
        "page_border": None,
        "dotted_values": False,
        "ink_values": rng.random() < style.get("ink_values_prob", 0.0),
        "heading_color": (rng.choice(style["heading_palette"])
                          if style.get("heading_palette")
                          and rng.random() < style.get("heading_color_prob", 0.0) else None),
        "body_px": body_px,
        "title_px": int(body_px * scale),
        "header_px": int(body_px * (1 + (scale - 1) * 0.6)),
        "line_height": round(rng.uniform(*typ["line_height"]), 2),
        "letter_spacing": round(rng.uniform(*typ["letter_spacing_px"]), 2),
        "color": color,
        "align": rng.choice(typ["align"]),
        "columns": 1,
        "column_gap": rng.randint(24, 48),
        "blocks": [],
    }
    page["title_px"] = int(page["body_px"] * scale)
    page["header_px"] = int(page["body_px"] * (1 + (scale - 1) * 0.6))

    blocks = _LAYOUTS[layout_type](rng, corpus, assets, page)
    _apply_style_pass(rng, page, blocks, corpus, assets, style, fonts)
    # title_px/header_px may depend on a body_px the layout overrode
    page["title_px"] = int(page["body_px"] * scale)
    page["header_px"] = int(page["body_px"] * (1 + (scale - 1) * 0.6))
    # give every table a UNIQUE id per page so multiple tables aren't merged into one during
    # markdown/structure reconstruction (data-tbl groups cells; collisions corrupt the table).
    _tid = 0
    for b in iter_blocks(blocks):
        if b.get("type") == "table":
            b["tid"] = _tid
            _tid += 1
    page["blocks"] = blocks
    return page


def iter_blocks(blocks: list[dict]):
    """Depth-first walk in DOM order, descending into 'band' columns (compose engine)."""
    for b in blocks:
        if b.get("type") == "band":
            for col in b["cols"]:
                yield from iter_blocks(col["blocks"])
        else:
            yield b


def expected_leaf_texts(page: dict) -> list[str]:
    """Leaf texts in the exact order the template emits them (== DOM order)."""
    out: list[str] = []
    for b in iter_blocks(page["blocks"]):
        t = b["type"]
        if t in ("title", "heading", "subheading", "text", "caption", "formula"):
            out.append(b["text"])
        elif t in ("image", "signature", "hand_drawing", "chart"):
            continue                              # non-text region: nothing to round-trip
        elif t in ("list", "bubbles"):
            out.extend(b["items"])
        elif t == "table":
            if b.get("header"):
                out.extend(b["header"])
            for row in b["rows"]:
                out.extend(row)
        elif t == "form":
            for f in b["fields"]:
                out.append(f["label"])
                out.append(f["value"])
    return out


# register the V5 compositional engine ("composed" pseudo-layout). Imported at the bottom so
# compose_engine can lazily use this module's factories without a circular import at load time.
from .compose_engine import _lay_composed  # noqa: E402
_LAYOUTS["composed"] = _lay_composed
