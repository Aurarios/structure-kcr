# V5 local stages 3-4 (resumed after the class-balance gate top-up): train --single detector on
# E:/kcr-v5 (34K, manifests_v5) then standalone eval + per-class threshold sweep.
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$PY = ".\.venv\Scripts\python.exe"

Write-Output "=== [3/4] DETECTOR --single  $(Get-Date -F HH:mm:ss) ==="
& $PY -m src.detect.train_obj_detector --single --config obj_single_v5
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Output "=== [4/4] EVAL vs V4 baseline  $(Get-Date -F HH:mm:ss) ==="
$latest = Get-ChildItem data/checkpoints/detector_obj_v5/step_* | Sort-Object Name | Select-Object -Last 1
& $PY -m src.eval.det_eval --ckpt $latest.FullName --manifest data/manifests_v5/val.jsonl `
    --profile single --size 1024 --sweep --out data/real_eval/det_eval_v5_single.json

Write-Output "=== V5 LOCAL TRAIN+EVAL COMPLETE  $(Get-Date -F HH:mm:ss) ==="
