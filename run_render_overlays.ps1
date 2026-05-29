$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$PY = ".\.venv\Scripts\python.exe"
Write-Output "=== OVERLAY RENDER START $(Get-Date -Format HH:mm:ss) ==="
& $PY src\build_dataset.py `
    --n 50000 `
    --start-index 300000 `
    --overlay-prob 1.0 `
    --merge-manifest `
    --workers 22 `
    --image-format jpg `
    --jpeg-quality 85 `
    --dsf 1 `
    --min-free-gb 15 `
    --seed 42
if ($LASTEXITCODE -ne 0) { Write-Output "RENDER FAILED (exit $LASTEXITCODE)"; exit 1 }
Write-Output "=== OVERLAY RENDER DONE $(Get-Date -Format HH:mm:ss) ==="
