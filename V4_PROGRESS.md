# V4 Progress / Resume Doc (last updated 2026-06-04)

Single reference to continue the work. Read this first when resuming.

## TL;DR — where we are

- **V4 = a from-scratch FCOS object-detection layout detector** that REPLACES the DBNet
  segmentation detector. It detects **region-level** boxes (overlap allowed), so it can do what
  DBNet structurally couldn't: a real `table` region enclosing its cells, `chart` vs `image`,
  `signature`, etc. Motivated by two SOTA papers the user provided (PP-DocLayout = RT-DETR;
  DocLayout-YOLO = YOLOv10 + GL-CRM + bin-packed synth). User approved a **full rewrite**.
- **STATUS: V4 detector DONE + validated on a mid-size 30K run. meanF1@0.5 = 0.875.** Bound to the
  OCR Studio web UI (`--v4` flag). The pipeline (detect → line-split → recognize → assemble →
  Markdown/HTML) works end-to-end.
- **THE ONE REMAINING PROBLEM: the recognizer.** It's still the frozen **V1** recognizer
  (`recognizer_single`), which HALLUCINATES on real scans (loops on frequent Khmer tokens like
  `បងប្អូន/មាន/បាន`). The detector is correct; the recognizer is the weak link. **Next step =
  retrain the recognizer on V3 data.**

## Key artifacts / locations

- **V4 detector checkpoint:** `data/checkpoints/detector_obj_v4/step_00007132/model.pt` (16 classes).
- **V4 code (NEW):**
  - `src/detect/model/fcos.py` — FCOS (reuses ResNet from dbnet.py) + FPN P3-P7 + CRM + head +
    `fcos_loss` + **ATSS assignment** + decode. `build_obj_detector(profile)`.
  - `src/detect/data/build_obj_targets.py` — `CLASSES_V4` (16), `collect_obj_boxes(label)`
    (leaf bbox per class + synthesized `table` region from cells), `NONTEXT_V4`.
  - `src/detect/data/obj_dataset.py` — `ObjDetDataset` + `collate_fn` (reuses letterbox/_scan_aug/_warp).
  - `src/detect/train_obj_detector.py` — trainer (Accelerator, cosine, per-class mAP@0.5 eval).
  - `src/detect/infer_obj.py` — `detect_page_obj()` decode + hand-rolled NMS (no torchvision).
  - `src/pipeline/line_split.py` — projection line-split of text blocks for the recognizer.
  - configs: `src/detect/configs/obj_single_v4.yaml` (img 1024, ckpt detector_obj_v4),
    `obj_parallel_v4.yaml`, `obj_smoke_v4.yaml` (overfit gate).
- **V4 pipeline wiring:** `src/pipeline/run_ocr.py` → `--v4` flag, `load_obj_models` + `run_ocr_v4`.
  `src/pipeline/assemble.py` → now parametrized with `classes=` arg (pass `CLASSES_V4`).
  `src/serve/app.py` (OCR Studio) → `--v4` flag.
- **Data:** 30K V3 pages at `E:\kcr-v3\{layout}\{images,labels}`; manifests `data/manifests_v3/`
  (28524 train / 1476 val). Real-image asset pool `E:\kcr-assets\` (photos/portraits/charts/
  drawings/logos 500 each + ~190 signatures; stamps only 4). Smoke/overlays `E:\kcr-v3-smoke\`.

## V4 class taxonomy (16) — `build_obj_targets.CLASSES_V4`
title, heading, subheading, text, list_item, caption, **table**, table_head, table_cell, image,
**chart**, signature, hand_drawing, formula, form_label, form_value.
Non-text (cropped, not recognized): image, chart, signature, hand_drawing, formula.

## Mid-size 30K result (final meanF1@0.5 = 0.875)
Excellent (≥0.96): table_head 1.0, table_cell 1.0, form_label 1.0, text/signature/caption/title
~0.98, heading/subheading/list/form_value ~0.96. Good: table region 0.85, image 0.84.
**WEAK: chart 0.63, hand_drawing 0.00** (rare + visually photo-like).

## Launch the OCR Studio (test the V4 detector)
```
.\.venv\Scripts\python.exe -m src.serve.app --v4 ^
  --det-ckpt data/checkpoints/detector_obj_v4/step_00007132/model.pt ^
  --rec-ckpt data/checkpoints/recognizer_single/step_00055000/model.pt ^
  --det-size 1024 --score-thresh 0.35 --device cuda --port 8000
```
Open http://localhost:8000 . Tabs: Markdown / HTML / Text / Lines. Drop `--v4` to compare DBNet.
(Detection looks correct; TEXT is garbage because rec = old V1 — see below.)

## NEXT STEPS (in priority order)

### 1. Retrain the recognizer on V3 (fixes the hallucination — biggest win)
The recognizer is the only reason end-to-end text is bad. Steps:
```
# (a) build V3 line crops (re-renders lines; ~30-45 min, CPU). Use 8 workers MAX (see gotcha).
.\.venv\Scripts\python.exe -m src.recognize.data.build_line_crops --pages 200000 --workers 8 ^
    --out-dir E:\kcr-recognize-v3\lines
# (b) train recognizer on them (~2-3h GPU)
.\.venv\Scripts\python.exe -m src.recognize.train_recognizer --single --config single_v3
```
`src/recognize/configs/single_v3.yaml` already exists (lines_root E:/kcr-recognize-v3, ckpt
recognizer_single_v3). The line_dataset already has scan-realistic aug + seal/stamp overlay aug.

### 2. Detector misdetect fixes (seen on real legal decree)
On a real scanned decree the V4 detector missed/mis-classified 3 things — all DATA gaps:
- **Official red seal not detected** → stamps are gradient-fallback (only 4 real). Need more real
  seals/stamps (HF had no croppable source; retry Wikimedia `--source wikimedia --only stamps` when
  un-throttled, or find another).
- **Page number "៣/៧" → `formula`** → add a `page_number` class (book/contract footers) so formula
  stops over-firing on slashed numerals.
- **Centered chapter heading "ជំពូកទី N" missed / →text** → add more centered standalone-heading
  variety in the sampler (currently `មាត្រា N` works as subheading; chapter titles underrepresented).

### 3. Fix weak detector classes chart (0.63) / hand_drawing (0.00)
Boost their sample frequency + visual distinctiveness in `layout_sampler.py` (chart already tagged;
hand_drawing is rare). Then re-render.

### 4. Full 350K + `--parallel`
After 1-3, render the full 350K (`run_v3_build_train.ps1` pattern but out E:\kcr-v3) and train
`detector_obj_v4` for the final model; then the `obj_parallel_v4` profile on the A5000 box.

## CRITICAL GOTCHAS (learned this session)
- **RENDER: use 8 workers MAX.** 12 over-subscribes the 32GB RAM → Chromium "Connection closed"
  crashes/stall. 8 is the stable max on this 20-core/32GB box. (build_dataset `--workers 8`.)
- **TRAIN GPU: batch 8 @1024px = GPU 99% / 10.7GB** (the max that fits on the 4070 Ti).
- **FCOS-on-documents: ATSS assignment is mandatory.** Vanilla FCOS (assign by max-side) starves
  extreme-aspect-ratio document regions (a 900x40 title → coarse level with no grid cell inside →
  F1 stuck ~0.07). The `_assign_image` in fcos.py uses ATSS (top-k center-nearest per level, inside
  box, area tie-break). The **overfit gate** (`--config obj_smoke_v4 --smoke 30`) caught this — ALWAYS
  run it before a big train.
- **Orphaned `spawn` workers**: PyTorch `persistent_workers=True` + the render ProcessPool leave
  spin-waiting orphan python.exe on Windows when the parent exits (drove CPU to 59%). Kill leftover
  `python.exe` with `*spawn_main*`/`*resource_tracker*` in the command line after big runs.
- Everything is ADDITIVE: DBNet (V1/V2/V3) + their checkpoints + the recognizer are frozen/intact.

## Memory files (auto-loaded each session)
`~/.claude/projects/.../memory/`: `v4-detector.md` (this work), `v3-redesign.md` (the dataset),
`v1-v2-split.md`. MEMORY.md indexes them.

## Background processes right now
OCR Studio running on :8000 (V4). No training running. GPU free. ~30K render + manifests_v3 on disk.
