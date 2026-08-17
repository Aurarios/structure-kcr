#!/usr/bin/env bash
# V4 best-accuracy pipeline on the 3x A5000 box: render 350K -> train --parallel detector ->
# build line crops -> train --parallel recognizer (V3, fixes the real-scan hallucination).
# Big data on /mnt/DATA_3 (root / is 97% full). See LINUX_A5000_RUNBOOK.md.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/mnt/DATA_1/pheng/kcr
WORKERS=${WORKERS:-16}            # render/line-crop CPU workers; override: WORKERS=12 ./run_v4_a5000.sh
NPROC=${NPROC:-3}                 # GPUs for DDP
TS() { date +%H:%M:%S; }

echo "=== [1/5] RENDER 350K  $(TS) ==="
python src/build_dataset.py --n 350000 --workers "$WORKERS" --image-format jpg --dsf 1 \
    --out-dir "$ROOT/v3" --manifest-dir manifests_v3 \
    --assets-dir "$ROOT/assets" --overlay-prob 0.12 --min-free-gb 20

echo "=== [2/5] QA GATES  $(TS) ==="
python -m src.validate.roundtrip_check
python -m src.validate.box_sanity

echo "=== [3/5] DETECTOR --parallel (3x A5000)  $(TS) ==="
accelerate launch --multi_gpu --num_processes "$NPROC" --mixed_precision bf16 \
    -m src.detect.train_obj_detector --parallel --config obj_parallel_v4

echo "=== [4/5] LINE CROPS 200K  $(TS) ==="
python -m src.recognize.data.build_line_crops --pages 200000 --workers "$WORKERS" \
    --out-dir "$ROOT/recognize-v3/lines"

echo "=== [5/5] RECOGNIZER --parallel on V3  $(TS) ==="
accelerate launch --multi_gpu --num_processes "$NPROC" --mixed_precision bf16 \
    -m src.recognize.train_recognizer --parallel --config parallel_v3

echo "=== V4 A5000 PIPELINE COMPLETE  $(TS) ==="
echo "detector  -> $ROOT/checkpoints/detector_obj_v4_parallel/<latest>/model.pt"
echo "recognizer-> $ROOT/checkpoints/recognizer_parallel_v3/<latest>/model.pt"
