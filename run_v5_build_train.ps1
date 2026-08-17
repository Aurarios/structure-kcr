# V5 local pipeline (RTX 4070 Ti): render 30K generalized pages -> QA gates -> train --single
# detector -> standalone eval vs the V4 baseline. ~6-8h total; run in a detached shell:
#   powershell -File run_v5_build_train.ps1 *> v5_local.log
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$PY = ".\.venv\Scripts\python.exe"

Write-Output "=== [1/4] RENDER 30K -> E:/kcr-v5  $(Get-Date -F HH:mm:ss) ==="
& $PY src/build_dataset.py --n 30000 --workers 8 --image-format jpg --dsf 1 `
    --out-dir E:/kcr-v5 --manifest-dir manifests_v5 --min-free-gb 5 --overlay-prob 0.12
if ($LASTEXITCODE -ne 0) { throw "render failed" }

Write-Output "=== [2/4] QA GATES  $(Get-Date -F HH:mm:ss) ==="
& $PY -m src.validate.roundtrip_check --root E:/kcr-v5
if ($LASTEXITCODE -ne 0) { throw "roundtrip gate failed" }
& $PY -m src.validate.box_sanity --root E:/kcr-v5
if ($LASTEXITCODE -ne 0) { throw "box gate failed" }
& $PY -m src.dataset_tools.stats --root E:/kcr-v5 --per-layout 400 --gate
if ($LASTEXITCODE -ne 0) { throw "class-balance gate failed" }

Write-Output "=== [3/4] DETECTOR --single  $(Get-Date -F HH:mm:ss) ==="
& $PY -m src.detect.train_obj_detector --single --config obj_single_v5
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Output "=== [4/4] EVAL vs V4 baseline  $(Get-Date -F HH:mm:ss) ==="
$latest = Get-ChildItem data/checkpoints/detector_obj_v5/step_* | Sort-Object Name | Select-Object -Last 1
& $PY -m src.eval.det_eval --ckpt $latest.FullName --manifest data/manifests_v5/val.jsonl `
    --profile single --size 1024 --sweep --out data/real_eval/det_eval_v5_single.json

Write-Output "=== V5 LOCAL PIPELINE COMPLETE  $(Get-Date -F HH:mm:ss) ==="
Write-Output "compare: data/real_eval/det_eval_v5_single.json vs det_eval_v4_parallel.json"
