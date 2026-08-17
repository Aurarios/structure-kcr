#!/usr/bin/env bash
# Launch the KCR OCR Studio web UI (macOS / Linux).
# Uses the bundled V1 detector + recognizer. CPU by default. Then open http://localhost:8000
set -e
export PYTHONIOENCODING=utf-8
PY="python3"
[ -x ".venv/bin/python" ] && PY=".venv/bin/python"
exec "$PY" -m kcr_ocr.app --device cpu --port 8000
