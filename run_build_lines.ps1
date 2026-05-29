$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$PY = ".\.venv\Scripts\python.exe"
Write-Output "=== BUILD LINE CROPS START $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.recognize.data.build_line_crops `
    --pages 100000 `
    --workers 10 `
    --start-seed 1000 `
    --jpeg-quality 88 `
    --val-frac 0.02 `
    --out-dir "E:\kcr-recognize\lines"
if ($LASTEXITCODE -ne 0) { Write-Output "BUILD LINES FAILED (exit $LASTEXITCODE)"; exit 1 }
Write-Output "=== BUILD LINE CROPS DONE $(Get-Date -Format HH:mm:ss) ==="
