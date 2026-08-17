# V2 pipeline: diverse detector data + 11-class detector + English-rich recognizer.
# FULLY ADDITIVE — V1 detector/recognizer, their data, manifests, configs and checkpoints are never
# touched. V2 writes images to E:\kcr-v2, line crops to E:\kcr-recognize-v2, manifests to
# data/manifests_v2, and trains detector_single_v2 + recognizer_single_v2 via the *_v2 configs.
#
# Run AFTER the V1 recognizer finishes (so render + train don't fight it for CPU/GPU):
#   .\run_v2_build_train.ps1
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$PY = ".\.venv\Scripts\python.exe"

Write-Output "=== [1/4] V2 PAGE RENDER START $(Get-Date -Format HH:mm:ss) ==="
# detector pages -> E:\kcr-v2 (~52GB for 120k); manifests on C: with absolute E: paths
& $PY src/build_dataset.py --n 120000 --workers 6 --image-format jpg --dsf 1 `
    --start-index 1000000 --out-dir E:\kcr-v2 --manifest-dir manifests_v2 `
    --overlay-prob 0.15 --min-free-gb 8 *> v2_render.log
if ($LASTEXITCODE -ne 0) { Write-Output "V2 RENDER FAILED ($LASTEXITCODE)"; exit 1 }
Write-Output "=== V2 PAGE RENDER DONE $(Get-Date -Format HH:mm:ss) ==="

Write-Output "=== [2/4] V2 DETECTOR TRAIN START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.detect.train_detector --single --config single_v2 *> v2_detector.log
if ($LASTEXITCODE -ne 0) { Write-Output "V2 DETECTOR FAILED ($LASTEXITCODE)"; exit 1 }
Write-Output "=== V2 DETECTOR DONE $(Get-Date -Format HH:mm:ss) ==="

Write-Output "=== [3/4] V2 LINE-CROP BUILD START $(Get-Date -Format HH:mm:ss) ==="
# English-rich line crops (cards/bilingual/mixed) -> E:\kcr-recognize-v2\lines
& $PY -m src.recognize.data.build_line_crops --pages 120000 --workers 6 `
    --out-dir E:\kcr-recognize-v2\lines *> v2_lines.log
if ($LASTEXITCODE -ne 0) { Write-Output "V2 LINE-CROP FAILED ($LASTEXITCODE)"; exit 1 }
Write-Output "=== V2 LINE-CROP BUILD DONE $(Get-Date -Format HH:mm:ss) ==="

Write-Output "=== [4/4] V2 RECOGNIZER TRAIN START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.recognize.train_recognizer --single --config single_v2 *> v2_recognizer.log
if ($LASTEXITCODE -ne 0) { Write-Output "V2 RECOGNIZER FAILED ($LASTEXITCODE)"; exit 1 }
Write-Output "=== V2 RECOGNIZER DONE $(Get-Date -Format HH:mm:ss) ==="

Write-Output "=== V2 PIPELINE COMPLETE $(Get-Date -Format HH:mm:ss) ==="
Write-Output "Relaunch Studio on V2:"
Write-Output "  --det-ckpt data/checkpoints/detector_single_v2/<latest>/model.pt --det-size 1280 \"
Write-Output "  --rec-ckpt data/checkpoints/recognizer_single_v2/<latest>/model.pt"
