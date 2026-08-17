# KCR Khmer OCR — V1 Inference Bundle

Self-contained, two-stage **Khmer document OCR** you can run with no training setup:

```
image → text detector (DBNet) → crop lines → recognizer (SVTR+attention+CTC) → Markdown / HTML / Text
```

Everything needed is in this folder — model weights, tokenizer, code, and a web UI. **CPU-only by
default** (no GPU required). This is the **V1** model; it targets clean printed / scanned Khmer
documents (articles, legal docs, tables, forms).

---

## Quick start

### 1. Install dependencies (once)
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the web UI (easiest)
```bash
# Windows
./run_studio.ps1
# macOS / Linux
./run_studio.sh
```
Then open **http://localhost:8000**, click to upload a Khmer document image, and read the result in
the **Markdown / HTML / Text** tabs. Toggle **Detection overlay** to see the detected line boxes.

### 3. …or run on a single image from the command line
```bash
python -m kcr_ocr.pipeline --image path/to/page.jpg --out boxed.jpg --audit-dir out_audit
```
- `--out boxed.jpg` — saves the image with detected boxes drawn.
- `--audit-dir out_audit` — dumps the normalized image, numbered detection image, and the recognized
  text (per-line + assembled markdown). Great for debugging.
- `--device cuda` — use a GPU if you have one (default is `cpu`).

---

## What's in here
```
inference_v1/
├── kcr_ocr/                  # the inference package (no training code)
│   ├── app.py                # web UI (stdlib http.server, no framework)
│   ├── pipeline.py           # image → detect → recognize → assemble  (+ CLI)
│   ├── dbnet.py              # text detector model
│   ├── svtr.py               # line recognizer model
│   ├── infer_detector.py     # box decoding
│   ├── assemble.py           # reading order + Markdown/HTML/Text output
│   ├── text_encoder.py       # SentencePiece wrapper
│   └── khmer_utils.py        # Khmer Unicode normalization (correctness-critical)
├── assets/
│   ├── checkpoints/          # detector.pt (48MB) + recognizer.pt (74MB)
│   └── tokenizer/            # khmer_ocr.model (SentencePiece, 22k vocab)
├── requirements.txt
├── run_studio.ps1 / .sh      # one-command web UI launchers
└── README.md
```

## Tips & notes
- **Higher-DPI input reads better.** The pipeline auto-upscales small images to ~2400px wide; clear
  scans/photos give the best result.
- **Khmer + basic English** are supported (the tokenizer covers full Khmer + ASCII).
- **Known limits (V1):** struggles with heavy stamp/seal occlusion, decorative ID-card layouts, and
  long English prose. (A V2 model targeting eKYC/cards, books, worksheets, figures and stamps is in
  progress.)
- Runs comfortably on CPU; a typical page takes a few seconds.

## Troubleshooting
- **`ModuleNotFoundError`** → run commands from *inside* the `inference_v1/` folder (so `kcr_ocr` is
  importable), with the venv activated.
- **Port 8000 in use** → `python -m kcr_ocr.app --port 8080`.
- **Khmer text shows as boxes in the browser** → install a Khmer system font (e.g. *Khmer OS*); the
  OCR itself is unaffected.
