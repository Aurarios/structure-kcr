# V6 local pipeline (RTX 4070 Ti): render 30K pages -> QA gates -> train --single detector
# -> sweep -> real battery vs the tuned V5 baseline. ~6-8h total; run detached:
#   powershell -File run_v6_build_train.ps1 *> v6_local.log
#
# What V6 changes vs V5 (all three motivated by measured real-document failures):
#   1. `word_bank` split from `list_item` (taxonomy v6, 17 classes). V5 gave the one-container
#      word row and the per-word bubble row the SAME label, so on a real worksheet the detector
#      hedged at 0.11-0.59 confidence and emitted competing whole-row boxes; one oval was
#      unreachable at any threshold.
#   2. Auxiliary page-layout head (multi-task, +72.5K params, ignored at inference).
#   3. Numbering prefixes absorb a parenthetical qualifier ('១.(២០១៥)') so the hanging indent
#      lands after the whole group; _f_bubbles weight raised 0.7 -> 1.6 to support the new class.
$ErrorActionPreference = "Stop"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$PY = ".\.venv\Scripts\python.exe"

Write-Output "=== [1/5] RENDER 30K -> E:/kcr-v6  $(Get-Date -F HH:mm:ss) ==="
& $PY src/build_dataset.py --n 30000 --workers 8 --image-format jpg --dsf 1 `
    --out-dir E:/kcr-v6 --manifest-dir manifests_v6 --min-free-gb 5 --overlay-prob 0.12
if ($LASTEXITCODE -ne 0) { throw "render failed" }

Write-Output "=== [2/5] QA GATES  $(Get-Date -F HH:mm:ss) ==="
& $PY -m src.validate.roundtrip_check --root E:/kcr-v6
if ($LASTEXITCODE -ne 0) { throw "roundtrip gate failed" }
& $PY -m src.validate.box_sanity --root E:/kcr-v6
if ($LASTEXITCODE -ne 0) { throw "box gate failed" }
& $PY -m src.dataset_tools.stats --root E:/kcr-v6 --per-layout 400 --gate
if ($LASTEXITCODE -ne 0) { throw "class-balance gate failed" }

Write-Output "=== [3/5] DETECTOR --single (taxonomy v6 + layout head)  $(Get-Date -F HH:mm:ss) ==="
& $PY -m src.detect.train_obj_detector --single --config obj_single_v6
if ($LASTEXITCODE -ne 0) { throw "training failed" }

Write-Output "=== [4/5] SWEEP per-class thresholds  $(Get-Date -F HH:mm:ss) ==="
$latest = Get-ChildItem data/checkpoints/detector_obj_v6/step_* | Sort-Object Name | Select-Object -Last 1
& $PY -m src.eval.det_eval --ckpt $latest.FullName --manifest data/manifests_v6/val.jsonl `
    --profile single --size 1024 --sweep --out data/real_eval/det_eval_v6_sweep.json
if ($LASTEXITCODE -ne 0) { throw "eval failed" }
Write-Output "NOTE: copy the swept thresholds into obj_single_v6.yaml:score_thresh_per_class"
Write-Output "      (it currently carries v5's table as a placeholder; word_bank has no v5 entry)"

Write-Output "=== [5/5] REAL BATTERY vs v5_tuned  $(Get-Date -F HH:mm:ss) ==="
& $PY -m src.eval.real_battery --tag v6 --deskew `
    --det-ckpt $latest.FullName --det-profile single --det-size 1024 `
    --rec-ckpt data/checkpoints/recognizer_parallel_v3/step_REC --rec-profile parallel `
    --thresholds data/real_eval/det_eval_v6_sweep.json
& $PY -m src.eval.real_battery --compare v5_tuned v6

Write-Output "=== V6 LOCAL PIPELINE COMPLETE  $(Get-Date -F HH:mm:ss) ==="
Write-Output "compare: data/real_eval/reports/compare_v5_tuned_vs_v6.html"
