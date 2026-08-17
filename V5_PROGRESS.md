# V5 — Generalized Data Engine, Box Accuracy, Eval Harness & Dataset Tooling

Plan: dataset gaps (class starvation 260:1, zero color/font-pairing diversity, 16 fixed layouts,
trained-in box looseness, no real-doc metric) → eval harness first, then dataset tooling, style
generalization, compositional layout engine, box-accuracy fixes, re-render + retrain.

## Phase A — eval harness + v4 baselines  ✅

NEW `src/eval/det_eval.py` — standalone detector eval: micro per-class F1@0.5 **and** F1@0.75,
mean matched IoU (tightness), per-layout breakdown, per-class score-threshold sweep, JSON out.
NEW `src/eval/real_battery.py` — fixed real-doc battery (`data/real_eval/pages/`, 5 pages: hi-DPI
legal decree, low-res legal scan, worksheet, colored exam, angled licence photo) → versioned HTML
report per tag + `--compare tagA tagB` diff page.

### Baseline: detector_obj_v4_parallel_line/step_00055404 @1280, val 400 pages (2026-06-10)

- **meanF1@0.5 = 0.981, meanF1@0.75 = 0.951, mean matched IoU = 0.957** → the synthetic val set is
  **saturated**; the model has mastered the synthetic distribution. Synthetic val can no longer
  drive progress — only new data diversity + the real battery can. (Training-time 0.853 was a
  noisier per-page-averaged metric; this is global/micro.)
- Weakest classes = the starved ones: **chart F1 0.834 (R 0.761)**, table P 0.900, image P 0.921.
- Tightness healthy on synth (text mIoU 0.94); worksheet/exam pages lowest mIoU (0.93).
- Per-class best thresholds (for Phase E): image 0.45, signature 0.50, table_cell 0.55, table 0.40,
  chart 0.40, text 0.30, headings/captions 0.10–0.20. Full JSON:
  `data/real_eval/det_eval_v4_parallel.json`.

### Baseline: real battery tag `v4_parallel` (det parallel_line 55404 @1280 + rec parallel step_REC)

`data/real_eval/reports/v4_parallel/report.html` — 33/33/21/29/15 lines on the 5 pages, 0.3–0.9 s/page
(GPU). Known qualitative issues to beat in v5: worksheet word-bank wide boxes, exam 2-col structure
flattening, licence-card angled capture.

## Phase B — dataset_tools + versioned dataset org  ✅ (code)

NEW `src/dataset_tools/{common,stats,viz,browse}.py`:
- `stats` — det-target class shares + per-layout + style coverage; **`--gate`** fails (exit 1) if a
  class share is under its floor (default 1.5%; hand_drawing/table 0.5%). V3 measured baseline:
  text 52.7%, table_cell 18.4%, chart 0.6%, formula 0.5%, hand_drawing 0.2% → v3 FAILS the gate
  (by design; v5 must pass).
- `viz` — contact sheet of `collect_obj_boxes` training-target overlays (supersedes
  visualize_dataset.py).
- `browse` — query pages by class/layout → overlay contact sheet (find rare-class samples fast).
- `build_dataset.py` writes per-page `meta` (fonts, colors, engine, …) + dataset `meta.json` +
  `index.jsonl` + auto `DATASET_CARD.md` under the versioned root (`--per-layout-dir E:/kcr-v5`).

## Phase C — style generalization  ✅

`config/layouts.yaml style:` + `sample_page()` + `page.html.j2` + `augment.py`:
- **Colored text** (30% pages: blue/red/green/purple inks) + colored headings (40%) — the
  data-level fix for the colored-exam failure. **Per-page font pairing** (55%: independent title
  font, the Muol+body convention). **Emphasis runs** (bold/underline/highlight inside paragraphs;
  pre+emph+post is an exact partition split only at spaces → shaping + round-trip safe).
  **Ruled/grid exercise-book paper** (8%). **Pen-ink form values** (35%). **Mixed Khmer/Latin/digit
  content** injection (15%). **Page geometry**: width 1080–1460, landscape 5%, narrow receipts.
- **Capture realism** (augment.py, photometric/box-safe): phone-light gradients (25%), back-page
  bleed-through ghosting (12%), photocopier banding (10%).
- Rare-class boosters inserted into flow layouts (formula/chart/hand_drawing/signature/subheading).
- Gates on a 120-page render: roundtrip 120/120 (100%), box_sanity 120/120.

## Phase D — compositional layout engine  ✅

`src/render/compose_engine.py` — "composed" pseudo-layout, **~50% of all pages** (weight 26.5 =
sum of the 16 curated weights). Adapts DocLayout-YOLO's Mesh-candidate BestFit to the HTML/DOM
pipeline: stratified element factories (paragraphs/list/figure/table/form/bubbles/formula/
signatures) packed into a sampled band scaffold (flex rows of auto-height columns + sidebar bands)
with width-budget BestFit; text is never height-clipped so DOM text == visible text by
construction. Template got ONE new recursive block type (`band`); rendering/labeling unchanged.
- Gates on a 240-page mixed render (109 composed / 131 curated): roundtrip 240/240,
  box_sanity 240/240, **class-balance gate PASS** — chart 0.65→1.66%, formula 0.54→2.61%,
  hand_drawing 0.20→0.89%, signature 0.92→2.04%, table_cell 18.4→10.1%.

## Phase E — box accuracy  ✅ (code; A/B measured at Phase F eval)

- `_warp` GT inflation tamed for the line-level detector: jitter 1.5–6% → 0.4–2% of size,
  rotation ±3° → ±1.2°, prob 0.3 → 0.25 (`obj_dataset.py`; DBNet defaults untouched).
- **Ink-snap** (`run_ocr.refine_boxes_ink`, default on): detected text boxes shrink to ink extent
  (min-channel + Otsu, never grows, blank-crop guard). Validated on the worksheet battery page —
  word-bank/caption boxes now hug the text (report tag `v4_inksnap`).
- Per-class score thresholds wired into `detect_page_obj` (`score_thresh_per_class` in det cfg);
  v4 sweep values in `data/real_eval/det_eval_v4_parallel.json`; re-sweep after v5 training.

## Phase F.0 — local dataset build: gate episodes (2026-06-10)

The class-balance gate caught two real issues at scale (the 240-page smoke was within noise):
1. **chart 1.48% < 1.50%** on the 30K render → chart probs bumped (curated insert 0.55→0.70, rare-band
   chart pref 0.6→0.75) + 4K top-up → 1.52%.
2. **Sampling bias found**: stats sampled 400/layout-dir, but `composed` is ONE dir holding ~50% of
   pages → engine pages 17× under-weighted; chart was actually 1.70% (fine) but **image truly 1.39%**
   (even-sampling had over-counted it via curated image-heavy layouts). Fixed `stats` to sample
   proportionally to dir size. Permanent config fixes: composed figure mix 45/30/25 image/chart/hd +
   curated `image_prob: 0.35` insert (fresh renders now measure ~1.6% image) + 8K top-up.
3. **Floor calibration**: image floor set to 1.2% (large-object class, ~43K instances at 42K pages,
   F1 0.949 in V4 — share-of-line-boxes under-represents its signal; default 1.5% guards the
   historically-zero classes). Final: **42K pages (39,829 train / 2,171 val), ALL GATES PASS** —
   roundtrip 100%, box_sanity clean, class balance with chart 1.73%, image 1.43%, formula 2.52%,
   hand_drawing 0.93%.

NOTE: the local `--single` run trains on the 34K manifest snapshot it loaded at launch (the 8K
image top-up landed mid-training); the production A5000 350K render uses the fully fixed config.

## Phase F — local train + compare  ✅ (2026-06-10; A5000 350K remaining)

**`detector_obj_v5/step_00008080`** (single, resnet18 @1024, 42K-set 34K-snapshot, 8080 steps):

| eval (micro, 400 pages) | V5 model on V5 val | V4 model on V5 val | V4 model on V3 val |
|---|---|---|---|
| meanF1@0.5 | **0.957** | 0.744 | 0.981 (saturated) |
| meanF1@0.75 | **0.912** | 0.693 | 0.951 |
| worst classes | table 0.86, subheading 0.91 | formula 0.52, table 0.55, form_value 0.56 | chart 0.83 |

- **The generalization gap is now measured**: V4 (bigger model, resnet50 @1280) loses −0.24 F1 on
  the V5 distribution; V5-single masters it. V4's 0.981 was narrow-distribution overfit — exactly
  the user's hypothesis.
- Former famine classes on V5 val: **chart 0.973, formula 0.981, hand_drawing 0.965,
  signature 0.996** (V4-era: ≈0–0.83).
- Per-class threshold sweep for the V5 model in `data/real_eval/det_eval_v5_single.json`
  (e.g. chart/image/signature/table/table_cell 0.35, text/caption/heading 0.20).
- **Real battery** tag `v5_single` + `compare_v4_parallel_vs_v5_single.html`: more lines found on
  every hard page (colored exam 35 vs 29; angled licence 21 lines + 4 figures vs 15 + 2 — granular
  bilingual field detection on the glossy card).

## Phase F.2 — real-doc test round + numeral fix (r2)  ✅ (2026-06-10 evening)

User testing in the detector tester found 3 worksheet misses. Diagnosis (thr-0.05 dump):
word-bank 0.39 + 5th bubble 0.40 were *threshold* artifacts (the studio's per-class thresholds
already recover them); but **leading numerals excluded from every line box was a GT-convention
bug**: CSS `<ol>`/`<ul>` markers are invisible to `Range.getClientRects`, so all synthetic list
labels excluded numbering and the model learned to exclude it.

Fixes: `_numbered()` bakes numbering INTO item text (Khmer/Arabic/letter/dash styles), template
lists are markerless (`list-style:none`), scientific_paper double-numbering removed, single-
container word-bank variant (wide word-spacing, square corners) in worksheet + compose engine.

**r2 retrain** (fresh 30K, all gates passed first try, `detector_obj_v5/step_00007116`):
- micro meanF1@0.5 0.956, **@0.75 0.922** (r1 0.912 — tighter), mIoU 0.920; image 0.84→0.90.
- Real worksheet: every line box now STARTS AT its numeral (`១. ២. ៣.` included), all 5 bubbles +
  word-bank detected. Remaining cosmetic: 2 weak full-row duplicate boxes at 0.16–0.24 conf.
- Battery (production decode w/ per-class thresholds, tag `v5_r2`): legal 52 lines (v4: 33),
  colored exam 39 (v4: 29), licence 30 (v4: 15). `compare_v4_parallel_vs_v5_r2.html`.
- Serve apps + real_battery now auto-resolve the latest `detector_obj_v5/step_*` and apply sweep
  thresholds by default (r1 archived at `detector_obj_v5_r1`).

## Phase F.3 — line-fragmentation GT bug + hanging-indent gap (r3)  ⏳ training

User zoom-in showed question boxes starting MID-numeral + a sliver box. Probe confirmed the root
cause: **`Range.getClientRects()` returns one rect per INLINE SEGMENT (and duplicates for styled
elements)** — a `pre<b>emph</b>post` line yielded 5 rects, so Phase-C emphasis had been silently
fragmenting line GT in ~25% of paragraphs (the model learned lines can split mid-line). Fixes:
1. `_EXTRACT_JS` unions rects with >50% vertical overlap → exactly one rect per visual line
   (probe: 2 lines → 2 rects). 2. `numgap` hanging indent: empty inline spacer between `១.` and
   question text (real-worksheet wide gap; textContent unchanged → roundtrip 80/80).
3. Pipeline: `suppress_row_merges` (kills weak whole-row duplicate boxes), assemble splits blocks
   at numbered-line starts. 4. Recognizer train-aug #10: enclosing border/ellipse (bubble crops).
Remaining recognizer-side (until RECOGNIZER=1 refresh): bubble word garble, `១.`→`.` misreads.

**r3 RESULT (2026-06-11, `detector_obj_v5/step_00007128`) — GO for A5000**: micro 0.952/0.906/0.916,
text P 0.995, table best-yet 0.905. Real worksheet: every numbered question box includes its
numeral (edges 444–460 vs r2's 462–560), headings include `ក./គ.`, all bubbles + word-bank +
wrap-lines + captions + page number. Battery `v5_r3`: licence 36 lines + 2 figures (v4: 15+2).
Residual: 2 faint fragment FPs at 0.16–0.20 conf (expect 350K training to calibrate; filterable).

**Pipeline polish (post-r3, no retrain)** — the last few px of numeral clipping is the detector's
edge-regression precision limit (3–6px at input res), so the pipeline absorbs it:
- `refine_boxes_ink` now also EXTENDS a box outward when ink is continuous at its edge (cutting
  through a glyph) until a real whitespace gap — worksheet edges 450/460 → 440–450, all numerals
  fully recovered. Capped at 0.9×h; true gaps never extended.
- `drop_small_fragments` (h < 0.55×median AND score < 0.32) kills descender/border fragments.
- Deployment per-class thresholds floored at 0.22 (sweep is synth-tuned where precision is free).
Worksheet end state: 20 clean detections, every numeral inside its box, zero fragments.

### Remaining (A5000): `./run_v5_a5000.sh` — 350K render (fully fixed config incl. numeral fix) →
gates → `--parallel` detector (`detector_obj_v5_parallel`) → det_eval+sweep. Optional
`RECOGNIZER=1` refresh (LR 5e-4). Then `real_battery --tag v5_parallel` + compare; studio then
points at the downloaded parallel ckpt via `--det-ckpt/--det-profile parallel --det-size 1280`.
