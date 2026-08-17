# KCR OCR — Progress & Findings

_Last updated: 2026-05-31. Two-stage Khmer document OCR. **V1 is baked; V2 is implemented but its
training is NOT finished** (paused to free the GPU)._

---

## 1. System overview

`image → DBNet detector (boxes + region class) → crop lines → SVTR+attn+CTC recognizer → deterministic
assembly → grounded-markdown / HTML / text`. Image/formula regions are cropped, not recognized.

---

## 2. V1 — BAKED (do **not** overwrite)

| Piece | Checkpoint | Quality |
|---|---|---|
| Detector | `data/checkpoints/detector_single/step_00207730` (9-class, 960px) | F1@.5 **0.952**, F1@.7 0.838, mIoU 0.812 |
| Recognizer | `data/checkpoints/recognizer_single/step_00050000` (best) | CER **2.00%**, syllER 2.36% |

- Recognizer training was **stopped early** at ~step 58k/124k because validation CER **plateaued ~2.1%**
  from epoch ~2.5 (oscillating 2.0–2.35%, no downtrend). Best saved = step_50000.
- V1 data: `data/synthetic/` (350k pages, `syn_0000000`–`syn_0349999`), `data/manifests/`,
  `E:/kcr-recognize/lines`. Configs: `single.yaml` / `parallel.yaml` (left byte-for-byte intact).

---

## 3. Key findings & fixes (the expensive debugging lessons — keep these)

- **DPI/scale mismatch was the dominant real-doc failure.** Recognizer trained on crops downscaled
  from high-DPI renders; low-DPI scans upscaled to 48px were OOD. Fix: `run_ocr.normalize_dpi()`
  upscales any input narrower than ~2400px. (Verified by experiment — this cracked real-doc reading.)
- **Right-edge truncation = train/inference crop-width mismatch.** Inference must clamp line crops to
  the recognizer's training `max_w` (800). `run_ocr.CROP_MAX_W` MUST equal config `max_w`.
- **Khmer pre-base vowels ( េ ែ ៃ ោ ៅ ើ) render LEFT of their consonant** → a box tight to the ink
  clips them ("front word cut off"). Fix: horizontal pad on the recognizer crop only (box stays exact).
- **Headings over-promoted to H1/H2** on worksheets → `refine_structure` now TRUSTS the learned class
  and only promotes a genuinely centred+short line that the head left as `text`.
- **Emoji / other-script hallucination** on hard crops (seals, photos) → `run_ocr._clean_line` strips
  anything outside Khmer + ASCII + Latin-1. Live for V1 and V2.
- **Detector merged tight lines** → vertical-squash augmentation (V1) taught line separation.
- **prob_thresh tradeoff (measured):** 0.30 = no line-merging but misses faint bubble-edge glyphs;
  ≤0.22 recovers them but merges tight prose lines. Kept 0.30; faint-edge recovery handled by V2 data.
- **Studio audit dump** (`data/manifests/studio_audit/{normalized,detection}.jpg`, `lines.txt`,
  `figures/`) turns "looks cut off" into exact box coordinates — use it to debug, not screenshots.

---

## 4. Detector audit → why V2

V1 detector weaknesses: (1) **data monoculture** — only 5 clean templates, no cards/books/worksheets/
figures/formulas; (2) **no image/formula** path (`_EXTRACT_JS` dropped non-text); (3) **weak, imbalanced
class head** (low loss weight); (4) **no blur/perspective aug** → faint edges on real photos missed;
(5) train 960 vs infer 1536 mismatch.

---

## 5. V2 — IMPLEMENTED (code complete, additive, V1-isolated)

- **Classes 9 → 11**: added `image` (figures/photos/logos/stamps/signatures — cropped, not read) and
  `formula`. `dbnet.load_detector()` infers class count from the checkpoint → **loads both V1 (9) and
  V2 (11)** with one codebase.
- **New layouts** (`layout_sampler.py` + `layouts.yaml` + `page.html.j2`): `id_card`/eKYC
  (flag/photo/QR as `image` + **bilingual KH/EN** fields + MRZ), `book_page`, `worksheet`
  (numbered + dotted leaders + **rounded-bubble word-banks** + captioned figure prompts),
  `article_with_figure`, `formula_doc`, `receipt_invoice`, `certificate`.
- **Photo-realistic image regions** (`build_dataset._paint_figures`): paints gradient + random shapes +
  noise into `image` boxes (replaces smooth CSS gradients) so the detector generalizes "figure" → real
  photos.
- **Rounded-bubble word-bank** (`bubbles` block → `list_item`): fixes the bubble overlap + class
  flip-flop (was trained as a plain `<ul>`).
- **Detector augmentation**: scan/blur/low-res/JPEG/lighting/noise (`dataset._scan_aug`) + box-aware
  **perspective/rotation** (`dataset._warp`) for angled card/phone capture.
- **Class reliability**: `db_loss` gamma 0.5 → 1.0; assembly trusts the learned class.
- **Resolution**: train + infer both 1280.
- **Inference**: `image`/`formula` cropped not recognized → `run_ocr` returns `figures`; assembly emits
  `![រូបភាព]` / `<img>` / `$$…$$`; Studio embeds the crops; emoji strip applied.
- **Recognizer V2**: trains on **English-rich** V2 line crops (cards/bilingual/mixed) + a
  **seal/stamp overlay aug** (`line_dataset._aug`, 15% of crops) so it reads English AND reads through
  official stamps.
- **Isolation**: V2 configs are `*_v2.yaml`, selected by the new `--config` flag on both trainers.
  Images → `E:/kcr-v2`; line crops → `E:/kcr-recognize-v2`; manifests → `data/manifests_v2/`;
  checkpoints → `detector_single_v2` + `recognizer_single_v2`. **No V1 file is touched.**

### V2 mid-training test (detector step 25k, on the real worksheet)
- ✅ Numbered items kept WHOLE (numeral + dotted leader in one box) — the core V1 failure, fixed.
- ✅ Word-bank captured full-width; bubbles individually boxed; title/sections classified.
- ❌ Photos not detected at 25k → diagnosed as gradient-vs-real-photo gap → **fixed** via
  `_paint_figures` (validated in render; awaiting retrain to confirm on the model).
- Detector box F1@.5 plateaued ~0.95 by step 18k. Plot saved: `data/manifests/v2_det_test.jpg`.

### Benchmark vs "Blizzer OCR Studio"
Their strength is **recognition accuracy** + clean reading order; they **skip figures entirely**. Our
detector already matches their segmentation; the gap to their text quality is purely our recognizer →
exactly what recognizer V2 targets. Figure cropping is a bonus beyond Blizzer.

---

## 6. CURRENT STATE (paused)

- V2 pipeline was **killed mid stage-1 re-render** (on the fixed data) to free the GPU for Unity.
  A partial render may exist under `E:/kcr-v2`; it is overwritten on resume (start-index 1,000,000).
- **V1 detector + recognizer + their data/manifests/configs are intact.**
- The OCR Studio may still be running on `http://localhost:8000` (CPU, V1, harmless).

### To RESUME V2 (when GPU is free again)
```powershell
.\run_v2_build_train.ps1      # 4 stages: render 120k → detector V2 → line crops → recognizer V2  (~27h)
```
Then relaunch the Studio on the V2 pair:
```powershell
python -m src.serve.app --device cpu `
  --det-ckpt data/checkpoints/detector_single_v2/<latest>/model.pt --det-size 1280 `
  --rec-ckpt data/checkpoints/recognizer_single_v2/<latest>/model.pt
```

### Re-test the detector mid-training on any image (CPU, no GPU needed)
`src.detect.model.load_detector("data/checkpoints/detector_single_v2/step_XXXXXXXX/model.pt","single","cpu")`
→ `detect_page(..., return_class=True)` → draw class-labeled boxes (see prior `v2_det_test` script).

---

## 7. Open items / next

- **After V2 trains**: re-audit the driving license, stamped legal doc, worksheet, table page. Confirm
  photos detected (now textured training), bubbles clean (no overlap/misclass), English reads, stamps
  read-through.
- Recognition parity with Blizzer.
- Optional `english_doc` template if full English **prose** is ever needed (current English = card-style
  labels/values/dates, which is what the data contains).
- Phase 5: `--parallel` (3× A5000) using `*_v2` configs with `--config single_v2`/`parallel_v2`.

## 8. Disk / locations
- V1 images `data/synthetic/` (C:), V1 line crops `E:/kcr-recognize/lines`.
- V2 images `E:/kcr-v2` (~52GB), V2 line crops `E:/kcr-recognize-v2` (~35GB), V2 manifests
  `data/manifests_v2/` (C:, store absolute E: paths).
- NOTE: recognizer training grows the **Windows pagefile ~50–60 GB** (14 DataLoader workers each hold a
  full in-RAM copy of the line manifest); it's reclaimed when training stops. Not data loss.
