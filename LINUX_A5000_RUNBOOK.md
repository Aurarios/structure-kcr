# A5000 box — best-accuracy V4 run (render 350K + train --parallel detector & recognizer)

Target: **maximum accuracy**, 3× RTX A5000 (24 GB) via DDP. Everything runs on the Linux box
(`uc-Precision-7920-Tower`). CPU-efficiency work is parked (see `CPU_EFFICIENCY.md`).

## ⚠️ Storage rule
`/` is **97 % full (68 G left)** — keep code+venv there, but **all big data on `/mnt/DATA_1`**
(2.6 T free; the only big disk writable to user `pheng` — DATA_2/DATA_3 are root-only).
Data root used below: **`/mnt/DATA_1/pheng/kcr`**.

```
/mnt/DATA_1/pheng/kcr/v3/{layout}/{images,labels}   # 350K pages (~100–130 GB)
/mnt/DATA_1/pheng/kcr/assets/                        # real-image pool (re-fetched on Linux)
/mnt/DATA_1/pheng/kcr/recognize-v3/lines/            # line crops (~30 GB)
/mnt/DATA_1/pheng/kcr/checkpoints/                   # detector + recognizer checkpoints
```

## 0. Unpack the transfer zip
You were given `kcr_server_transfer.zip` (built by `package_for_server.py`). It contains the source
tree (`structure-kcr/`) + the data payload (lm_corpus, fonts, tokenizer model) + the asset pool
(`kcr-assets/`). On **Linux**:
```bash
cd ~ && unzip kcr_server_transfer.zip          # -> ~/structure-kcr/  and  ~/kcr-assets/
mkdir -p /mnt/DATA_1/pheng/kcr/{v3,recognize-v3/lines,checkpoints}
mv ~/kcr-assets /mnt/DATA_1/pheng/kcr/assets         # 204MB real-image pool (skip step 3 below)
cd ~/structure-kcr
```
The zip already includes `data/corpus/lm_corpus/`, `data/fonts/`, and `data/tokenizer/khmer_ocr.model`
— so **skip step 2** (transfer) and **step 3** (asset fetch) below.

## 1. Environment
```bash
cd ~/structure-kcr
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-train.txt
python -m playwright install --with-deps chromium      # --with-deps pulls the apt libs Chromium needs
```

## 2. Transfer the small data payload (Windows → Linux, ~1.4 GB)
Copy these from the Windows repo into the **same paths** in the Linux repo (scp/rsync/your choice):
```
data/corpus/lm_corpus/          # 1.4 GB — text source for rendering
data/fonts/                     # 2.8 MB — Khmer fonts (render needs these)
data/tokenizer/khmer_ocr.model  # ~2 MB  — recognizer tokenizer (skip the 717 MB train_text.txt)
```
Example from Windows (PowerShell, OpenSSH):
```powershell
scp -r data/corpus/lm_corpus  pheng@<linux-ip>:~/structure-kcr/data/corpus/
scp -r data/fonts             pheng@<linux-ip>:~/structure-kcr/data/
scp data/tokenizer/khmer_ocr.model pheng@<linux-ip>:~/structure-kcr/data/tokenizer/
```

## 3. Fetch the real-image asset pool (on Linux, no transfer needed)
```bash
python -m src.assets.fetch_assets --source hybrid --limit 500 --root /mnt/DATA_1/pheng/kcr/assets
```
(HF fills photos/portraits/logos/charts/drawings; Wikimedia attempts signatures/stamps. Re-runnable.)

## 4–7. One command: render 350K → train detector → line crops → train recognizer
```bash
chmod +x run_v4_a5000.sh
./run_v4_a5000.sh > v4_a5000.log 2>&1 &     # long-running; tail the log
tail -f v4_a5000.log
```
This runs the four stages below. To run them by hand instead, see the script — each stage is one line.

## Stage detail (what the script runs)
```bash
# [1] RENDER 350K  (CPU; tune --workers to ~nproc, watch RAM — Chromium ~1.5GB/worker)
python src/build_dataset.py --n 350000 --workers 16 --image-format jpg --dsf 1 \
    --out-dir /mnt/DATA_1/pheng/kcr/v3 --manifest-dir manifests_v3 \
    --assets-dir /mnt/DATA_1/pheng/kcr/assets --overlay-prob 0.12 --min-free-gb 20

# [1b] QA gates (fast; do NOT skip — catches shaping/roundtrip bugs before a long train)
python -m src.validate.roundtrip_check   # text fidelity (want ~100%)
python -m src.validate.box_sanity        # boxes in-bounds / ordered

# [2] DETECTOR --parallel (3x A5000 DDP, ResNet50 @1280, bf16)
accelerate launch --multi_gpu --num_processes 3 --mixed_precision bf16 \
    -m src.detect.train_obj_detector --parallel --config obj_parallel_v4

# [3] LINE CROPS 200K (CPU; re-renders lines with scan-realistic aug)
python -m src.recognize.data.build_line_crops --pages 200000 --workers 16 \
    --out-dir /mnt/DATA_1/pheng/kcr/recognize-v3/lines

# [4] RECOGNIZER --parallel on V3 crops (fixes the real-scan hallucination)
accelerate launch --multi_gpu --num_processes 3 --mixed_precision bf16 \
    -m src.recognize.train_recognizer --parallel --config parallel_v3
```

## Monitoring
```bash
watch -n5 nvidia-smi            # expect 3 GPUs busy during training
grep -E "meanF1|loss|ckpt|eval" v4_a5000.log | tail
```

## Gotchas
- **Render workers**: start ~16 (this box has way more RAM than the Windows one — 94 G shm). If you
  see Chromium `Connection closed while reading from the driver`, drop workers (RAM/Chromium pressure).
- **bf16**: A5000 (Ampere) supports it natively — both configs use it.
- **Overfit gate already passed** for the FCOS detector (ATSS assignment). No need to re-run it; if
  you change the model, run `--config obj_smoke_v4 --smoke 30` first.
- **Checkpoints land on `/mnt/DATA_3`** (configs already set this) — keeps `/` from filling.
- **Accuracy levers if you want more after this**: bump detector `batch_size_per_device` (24 GB can
  often take 6–8 @1280), raise `epochs`, or add the deferred data-quality fixes (page_number class,
  more seals, chart/hand_drawing boosts) — those need a re-render.

## Outputs
- Detector: `/mnt/DATA_1/pheng/kcr/checkpoints/detector_obj_v4_parallel/step_*/model.pt`
- Recognizer: `/mnt/DATA_1/pheng/kcr/checkpoints/recognizer_parallel_v3/step_*/model.pt`
- Test end-to-end: `python -m src.serve.app --v4 --det-ckpt <det> --rec-ckpt <rec> --det-size 1280 --device cuda --port 8000`
