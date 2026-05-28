#!/bin/bash
export PYTHONUNBUFFERED=1
PY=./.venv/bin/python
echo "=== FETCH $(date +%T) ==="
$PY -m src.corpus.registry --fetch --limit 40000
echo "=== CLEAN $(date +%T) ==="
$PY -m src.corpus.clean_normalize
echo "=== DEDUP $(date +%T) ==="
$PY -m src.corpus.dedup
echo "=== QUALITY $(date +%T) ==="
$PY -m src.corpus.quality_filter
echo "=== PACKAGE $(date +%T) ==="
$PY -m src.corpus.package_lm
echo "=== CORPUS PIPELINE DONE $(date +%T) ==="
