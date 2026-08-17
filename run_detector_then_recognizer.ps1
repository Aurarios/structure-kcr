$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$PY = ".\.venv\Scripts\python.exe"

Write-Output "=== DETECTOR TRAIN START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.detect.train_detector --single *> detector_run.log
if ($LASTEXITCODE -ne 0) { Write-Output "DETECTOR FAILED (exit $LASTEXITCODE)"; exit 1 }
Write-Output "=== DETECTOR DONE $(Get-Date -Format HH:mm:ss) ==="

# Auto-cutover: detector finished -> GPU free -> train the scan-aug recognizer
Write-Output "=== RECOGNIZER TRAIN START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.recognize.train_recognizer --single *> recognizer_run.log
if ($LASTEXITCODE -ne 0) { Write-Output "RECOGNIZER FAILED (exit $LASTEXITCODE)"; exit 1 }
Write-Output "=== RECOGNIZER DONE $(Get-Date -Format HH:mm:ss) ==="
Write-Output "=== FULL PIPELINE RETRAIN COMPLETE $(Get-Date -Format HH:mm:ss) ==="
