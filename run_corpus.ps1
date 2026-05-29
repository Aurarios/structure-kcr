$env:PYTHONUNBUFFERED = "1"
$PY = ".\.venv\Scripts\python.exe"
Write-Output "=== FETCH $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.corpus.registry --fetch --limit 100000
if ($LASTEXITCODE -ne 0) { Write-Output "FETCH FAILED"; exit 1 }
Write-Output "=== CLEAN $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.corpus.clean_normalize
if ($LASTEXITCODE -ne 0) { Write-Output "CLEAN FAILED"; exit 1 }
Write-Output "=== DEDUP $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.corpus.dedup
if ($LASTEXITCODE -ne 0) { Write-Output "DEDUP FAILED"; exit 1 }
Write-Output "=== QUALITY $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.corpus.quality_filter
if ($LASTEXITCODE -ne 0) { Write-Output "QUALITY FAILED"; exit 1 }
Write-Output "=== PACKAGE $(Get-Date -Format HH:mm:ss) ==="
& $PY -m src.corpus.package_lm
if ($LASTEXITCODE -ne 0) { Write-Output "PACKAGE FAILED"; exit 1 }
Write-Output "=== CORPUS PIPELINE DONE $(Get-Date -Format HH:mm:ss) ==="
