$env:PYTHONUNBUFFERED = "1"
$env:TRANSFORMERS_VERBOSITY = "warning"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$PY = ".\.venv\Scripts\python.exe"
Write-Output "=== TRAIN START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.train.train --single
if ($LASTEXITCODE -ne 0) { Write-Output "TRAIN FAILED (exit $LASTEXITCODE)"; exit 1 }
Write-Output "=== TRAIN DONE $(Get-Date -Format HH:mm:ss) ==="
