# Launch the KCR OCR Studio web UI (Windows / PowerShell).
# Uses the bundled V1 detector + recognizer. CPU by default. Then open http://localhost:8000
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$PY = if (Test-Path ".\.venv\Scripts\python.exe") { ".\.venv\Scripts\python.exe" } else { "python" }
& $PY -m kcr_ocr.app --device cpu --port 8000
