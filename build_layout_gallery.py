"""Build layout_gallery.html — one real sample image per layout (16) with its spec, weight, and the
generator function that produced it. Reads E:/kcr-v3 + config/layouts.yaml."""
import base64, glob, json, os, random
import cv2

BASE = "E:/kcr-v3"
THUMB_W = 360
# name -> (weight, one-line spec) from config/layouts.yaml comments
LAY = [
 ("news_article",2.0,"single/2-col news: headline, byline, body, inline figure"),
 ("magazine_multicol",1.5,"2–3 column magazine, multiple figures"),
 ("scientific_paper",1.5,"abstract, numbered sections, figures, equations, references"),
 ("business_report",1.5,"headings, body, charts, tables, KPI callouts"),
 ("financial_statement",1.5,"dense multi-table financial grids, totals"),
 ("form_generic",1.5,"label:value fields, checkboxes, signature line"),
 ("id_card",2.5,"eKYC: portrait, logo/flag, bilingual fields, signature, MRZ"),
 ("passport_mrz",1.5,"passport page: portrait, fields, 2-line MRZ"),
 ("book_page",2.0,"chapter/heading, body, figure, page number/footer"),
 ("textbook_figures",1.5,"dense text + multiple captioned figures + formulas"),
 ("worksheet",2.0,"numbered questions, dotted fill-ins, word banks, picture prompts"),
 ("exam_paper",1.5,"header block, numbered questions, multiple-choice, marks"),
 ("receipt_invoice",1.5,"key:value header, items table, totals, stamp"),
 ("certificate",1.5,"centered titles, body, signature(s), official seal"),
 ("letter_memo",1.0,"letterhead/logo, date, salutation, body, signature"),
 ("contract_legal",1.5,"numbered clauses, sub-clauses, signature blocks, stamp"),
]

def thumb(layout):
    imgs = sorted(glob.glob(f"{BASE}/{layout}/images/*.jpg"))
    if not imgs: return "", None
    random.seed(7); p = random.choice(imgs[:200])
    im = cv2.imread(p)
    if im is None: return "", None
    h,w = im.shape[:2]
    im = cv2.resize(im,(THUMB_W,max(1,round(h*THUMB_W/w))),interpolation=cv2.INTER_AREA)
    ok,buf = cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,70])
    # classes present in that page's label (to show the block types)
    lid = os.path.splitext(os.path.basename(p))[0]
    cls=set()
    lf=f"{BASE}/{layout}/labels/{lid}.json"
    if os.path.exists(lf):
        d=json.load(open(lf,encoding="utf-8"))
        cls={b.get("block_type") for b in d.get("blocks",[])}
    return ("data:image/jpeg;base64,"+base64.b64encode(buf).decode() if ok else "",
            sorted(c for c in cls if c))

cards=[]
total_w=sum(w for _,w,_ in LAY)
for name,w,spec in LAY:
    img,cls = thumb(name)
    pct = 100*w/total_w
    chips="".join(f'<span class="chip">{c}</span>' for c in (cls or []))
    cards.append(f'''<div class="card">
      <div class="thumb">{'<img src="'+img+'">' if img else '<div class=no>no sample</div>'}</div>
      <div class="body">
        <div class="hd"><b>{name}</b><span class="w">w {w} · {pct:.0f}%</span></div>
        <div class="spec">{spec}</div>
        <div class="gen">generator: <code>_lay_{name}()</code></div>
        <div class="chips">{chips}</div>
      </div></div>''')

html=f'''<!doctype html><html><head><meta charset="utf-8"><title>KCR — 16 layouts</title>
<style>
 :root{{--bg:#262624;--panel:#1f1e1d;--line:#3a3733;--ink:#ece9e3;--mut:#a8a29a;--dim:#6f6a62;--accent:#d97757;--cyan:#7dcfff}}
 *{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);
   font-family:ui-sans-serif,system-ui,"Segoe UI",sans-serif;line-height:1.6}}
 header{{padding:30px 40px 6px;max-width:1320px;margin:0 auto}}
 h1{{margin:0 0 4px;font-size:26px}} .sub{{color:var(--mut);font-size:14px;max-width:820px}}
 code{{font-family:ui-monospace,Consolas,monospace;font-size:.86em;color:var(--cyan)}}
 .grid{{max-width:1320px;margin:14px auto 80px;padding:0 40px;display:grid;
   grid-template-columns:repeat(auto-fill,minmax(290px,1fr));gap:16px}}
 .card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden;display:flex;flex-direction:column}}
 .thumb{{background:#fbfaf7;height:300px;display:grid;place-items:center;overflow:hidden}}
 .thumb img{{width:100%;height:100%;object-fit:cover;object-position:top}}
 .no{{color:#999;font-size:13px}}
 .body{{padding:12px 14px}}
 .hd{{display:flex;justify-content:space-between;align-items:baseline;gap:8px}}
 .hd b{{font-size:15px}} .w{{color:var(--accent);font:600 11px ui-monospace,monospace}}
 .spec{{color:var(--mut);font-size:12.5px;margin:4px 0 6px}}
 .gen{{font-size:11.5px;color:var(--dim);margin-bottom:8px}}
 .chips{{display:flex;flex-wrap:wrap;gap:4px}}
 .chip{{font:600 10px ui-monospace,monospace;background:#2a2926;border:1px solid var(--line);
   color:var(--mut);padding:2px 6px;border-radius:10px}}
 .note{{max-width:1320px;margin:0 auto;padding:0 40px 50px;color:var(--dim);font-size:12.5px}}
</style></head><body>
<header><h1>The 16 KCR layouts</h1>
<div class="sub">One real generated sample per layout. Each is produced by a <code>_lay_*()</code> generator in
<code>src/render/layout_sampler.py</code>, selected by weight (in <code>config/layouts.yaml</code>) — higher
weight = more frequent. Chips show the block/detector classes present in that sampled page.</div></header>
<div class="grid">{''.join(cards)}</div>
<div class="note">Selection: <code>_weighted_choice(names, weights)</code> per page. Render: spec →
<code>templates/page.html.j2</code> → Chromium screenshot + DOM box read. See <code>data_formulation.html</code>
for the box/label deep-dive.</div>
</body></html>'''

open("layout_gallery.html","w",encoding="utf-8").write(html)
print(f"[done] layout_gallery.html ({len(html)//1024} KB, {len(LAY)} layouts)")
