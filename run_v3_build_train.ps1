# V3 pipeline: real-image diverse dataset (350K) + 15-class structured detector + recognizer.
# FULLY ADDITIVE — V1/V2 detectors, recognizers, their data, manifests, configs and checkpoints are
# never touched. V3 writes pages to E:\kcr-v3 (per-layout-type subdirs), line crops to
# E:\kcr-recognize-v3, manifests to data/manifests_v3, and trains detector_single_v3 +
# recognizer_single_v3 via the *_v3 configs.
#
# PREREQUISITE: a real-image pool at E:\kcr-assets (see src/assets/fetch_assets.py). If the pool is
# empty, image/signature/figure regions fall back to gradients (defeats the V3 goal) — fetch first:
#   $env:HF_HUB_DISABLE_SYMLINKS_WARNING=1
#   .\.venv\Scripts\python.exe -m src.assets.fetch_assets --source hybrid --limit 500
#   (HF fills photos/portraits/logos/charts/drawings; Wikimedia tries signatures/stamps.)
#
# Then run (after freeing the GPU):
#   .\run_v3_build_train.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$PY = ".\.venv\Scripts\python.exe"

Write-Output "=== [1/4] V3 PAGE RENDER (350K) START $(Get-Date -Format HH:mm:ss) ==="
# 350K pages -> E:\kcr-v3\{layout_type}\{images,labels}; manifests on C: with absolute E: paths.
# jpg dsf=1 keeps it disk-frugal (~100-130 GB). Real images pulled from E:\kcr-assets.
& $PY src/build_dataset.py --n 350000 --workers 6 --image-format jpg --dsf 1 `
    --out-dir E:\kcr-v3 --manifest-dir manifests_v3 --assets-dir E:\kcr-assets `
    --overlay-prob 0.12 --min-free-gb 12 *> v3_render.log
if ($LASTEXITCODE -ne 0) { Write-Output "V3 RENDER FAILED ($LASTEXITCODE)"; exit 1 }
Write-Output "=== V3 PAGE RENDER DONE $(Get-Date -Format HH:mm:ss) ==="

Write-Output "=== [2/4] V3 DETECTOR TRAIN START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.detect.train_detector --single --config single_v3 *> v3_detector.log
if ($LASTEXITCODE -ne 0) { Write-Output "V3 DETECTOR FAILED ($LASTEXITCODE)"; exit 1 }
Write-Output "=== V3 DETECTOR DONE $(Get-Date -Format HH:mm:ss) ==="

Write-Output "=== [3/4] V3 LINE-CROP BUILD START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.recognize.data.build_line_crops --pages 200000 --workers 6 `
    --out-dir E:\kcr-recognize-v3\lines *> v3_lines.log
if ($LASTEXITCODE -ne 0) { Write-Output "V3 LINE-CROP FAILED ($LASTEXITCODE)"; exit 1 }
Write-Output "=== V3 LINE-CROP BUILD DONE $(Get-Date -Format HH:mm:ss) ==="

Write-Output "=== [4/4] V3 RECOGNIZER TRAIN START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.recognize.train_recognizer --single --config single_v3 *> v3_recognizer.log
if ($LASTEXITCODE -ne 0) { Write-Output "V3 RECOGNIZER FAILED ($LASTEXITCODE)"; exit 1 }
Write-Output "=== V3 RECOGNIZER DONE $(Get-Date -Format HH:mm:ss) ==="

Write-Output "=== V3 PIPELINE COMPLETE $(Get-Date -Format HH:mm:ss) ==="
Write-Output "Relaunch Studio on V3 (15-class):"
Write-Output "  --det-ckpt data/checkpoints/detector_single_v3/<latest>/model.pt --det-size 1280 \"
Write-Output "  --rec-ckpt data/checkpoints/recognizer_single_v3/<latest>/model.pt"
