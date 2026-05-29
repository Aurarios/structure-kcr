$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$PY = ".\.venv\Scripts\python.exe"
Write-Output "=== DETECTOR TRAIN START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.detect.train_detector --single
if ($LASTEXITCODE -ne 0) { Write-Output "DETECTOR TRAIN FAILED (exit $LASTEXITCODE)"; exit 1 }
Write-Output "=== DETECTOR TRAIN DONE $(Get-Date -Format HH:mm:ss) ==="
