$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$PY = ".\.venv\Scripts\python.exe"
Write-Output "=== RECOGNIZER TRAIN START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.recognize.train_recognizer --single
if ($LASTEXITCODE -ne 0) { Write-Output "RECOGNIZER TRAIN FAILED (exit $LASTEXITCODE)"; exit 1 }
Write-Output "=== RECOGNIZER TRAIN DONE $(Get-Date -Format HH:mm:ss) ==="
