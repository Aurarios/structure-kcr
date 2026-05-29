$env:PYTHONUNBUFFERED = "1"
$PY = ".\.venv\Scripts\python.exe"
Write-Output "=== RENDER START $(Get-Date -Format HH:mm:ss) ==="
& $PY src\build_dataset.py `
    --n 300000 `
    --workers 22 `
    --image-format jpg `
    --jpeg-quality 85 `
    --dsf 1 `
    --min-free-gb 15
if ($LASTEXITCODE -ne 0) { Write-Output "RENDER FAILED (exit $LASTEXITCODE)"; exit 1 }
Write-Output "=== RENDER DONE $(Get-Date -Format HH:mm:ss) ==="
