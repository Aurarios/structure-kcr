# CPU-Inference Efficiency — findings + plan (2026-06-04)

**Goal:** OCR inference runs **CPU-only** (target: modest **4-core cloud CPU, 1–3 s/page**). The
3× A5000 box is a **training accelerator only** (DDP `--parallel`) — used to train leaner/retrained
models that make CPU inference viable. No GPU serving.

## Measured baseline (this box, `torch.set_num_threads(4)`)

End-to-end V4 pipeline (detector@1024 + recognizer attention-AR), per page:

| page | lines | 1024 | 768 |
|---|---|---|---|
| real legal (sparse) | 19–22 | 2.16 s | 2.39 s |
| scientific paper | 34–36 | 3.38 s | 3.23 s |
| book page (dense) | 40–61 | **4.83 s** | 2.85 s* |

\*768 also **drops lines** on dense pages (61→40) — an accuracy regression, not just speed.

**Split:** detector = fixed conv cost (1.40 s @1024); recognizer = **60–80 ms/line, linear** → it
**dominates dense pages**. The recognizer's cost is in the **shared encoder** (stem + 6-layer
transformer), NOT the decoder.

## What we tried (measured, 4 threads)

**Detector resolution sweep** (the real detector lever):
| size | s/page | vs 1024 |
|---|---|---|
| 1024 | 1.40 | 1.0× |
| 768 | 0.64 | **2.2×** |
| 640 | 0.43 | 3.3× |

**Recognizer decode** — attention-AR vs single-shot CTC (synthetic val, 136 crops):
| decoder | CER | exact | ms/crop |
|---|---|---|---|
| attention AR | 0.064 | 68.4% | 80.8 |
| CTC 1-shot | 0.067 | 59.6% | 60.4 |

→ CTC barely faster (1.3×, encoder dominates) and loses 9 pts exact-match. Not the win hoped.

**ONNX Runtime export** (`src/optimize/export_onnx.py`, `src/detect/infer_obj_onnx.py`):
- ONNX **fp32 ≈ 1.1–1.3×** over torch (PyTorch oneDNN CPU conv already well-tuned).
- ONNX **dynamic INT8 = 13–14× SLOWER on BOTH models.** Dynamic quant only helps large
  MatMul/Linear; on conv (detector) + small matmuls (recognizer) it adds per-call quantize/dequantize
  overhead that dwarfs savings. **Dynamic INT8 is the wrong tool here.**

## Conclusions

1. **No-retrain wins are modest:** detector@768 (2.2×, but costs accuracy on dense pages) + ONNX
   fp32 (~1.2×) + CTC decode (~1.3×). Gets dense pages ~4.8 → ~2.5–3 s on this box; borderline, and
   worse on a genuinely slow 4-core.
2. **Real INT8 needs STATIC (calibrated) quant** → QLinearConv/QLinearMatMul kernels (gated on the
   target CPU having VNNI/AVX-512). Dynamic quant must NOT be used.
3. **The reliable big levers require retraining — the A5000s' job:**
   - **Retrain detector @768** (config `image_size: 768`) → recover the accuracy lost downscaling →
     bank the 2.2× for free.
   - **Lean recognizer**: shrink the shared encoder (enc_layers 6→3/4, d_model 256→192) and/or
     distill from the current model. Directly cuts the per-line bottleneck.
   - Optional **QAT-INT8** for both.

## Artifacts built this session
- `src/recognize/model/svtr.py`: added `Recognizer.ctc_greedy()` (single-shot CTC decode); replaced
  `AdaptiveAvgPool2d((1,None))` with `x.mean(dim=2)` — numerically identical, ONNX-exportable.
- `src/optimize/export_onnx.py`: FCOS → ONNX (fp32) + flat cls/reg/ctr wrapper for fixed size.
- `src/detect/infer_obj_onnx.py`: `OnnxObjDetector` (precomputed locations, numpy decode+NMS;
  drop-in for `detect_page_obj`). fp32 model at `data/checkpoints/onnx/detector_single_1024.onnx`.

## Next steps (priority)
1. **[A5000]** Retrain detector @768 + a lean recognizer (task #55). Biggest, most reliable win.
2. **[CPU]** Static (calibrated) INT8 for detector + recognizer (task #53) — if target CPU has VNNI.
3. **[wire]** CPU-optimized `run_ocr` path: ONNX fp32 detector@768 + CTC/attention recognizer (#54).
4. Width-bucketed recognizer ONNX export (fixed-W buckets avoid the MultiheadAttention dynamic-seq
   reshape that blocks a single dynamic export).
