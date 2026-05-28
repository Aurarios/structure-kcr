#!/bin/bash
export PYTHONUNBUFFERED=1
./.venv/bin/python src/build_dataset.py --n 15000 --workers 6 --image-format jpg \
  --jpeg-quality 85 --dsf 1 --min-free-gb 3
echo "=== RENDER DONE $(date +%T) ==="
