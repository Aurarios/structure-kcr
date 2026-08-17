#!/usr/bin/env bash
# V5 generalized-data pipeline on the 3x A5000 box: render 350K (style diversity + compositional
# engine + class rebalance) -> QA gates (incl. the NEW class-balance gate) -> train --parallel
# detector. Optional recognizer refresh (RECOGNIZER=1) retrains on crops from the V5 pages so the
# recognizer sees colored text / ruled paper too. Big data on /mnt/DATA_1 (root / is 97% full).
# See LINUX_A5000_RUNBOOK.md. Check nvidia-smi for other users' jobs BEFORE launching.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/mnt/DATA_1/pheng/kcr
WORKERS=${WORKERS:-16}            # render CPU workers; override: WORKERS=12 ./run_v5_a5000.sh
NPROC=${NPROC:-3}                 # GPUs for DDP
N_PAGES=${N_PAGES:-350000}
RECOGNIZER=${RECOGNIZER:-0}       # 1 = also refresh the recognizer on V5 crops (LR 5e-4!)
TS() { date +%H:%M:%S; }

echo "=== [1/4] RENDER ${N_PAGES}  $(TS) ==="
python src/build_dataset.py --n "$N_PAGES" --workers "$WORKERS" --image-format jpg --dsf 1 \
    --out-dir "$ROOT/v5" --manifest-dir manifests_v5 \
    --assets-dir "$ROOT/assets" --overlay-prob 0.12 --min-free-gb 20

echo "=== [2/4] QA GATES (roundtrip + boxes + class balance)  $(TS) ==="
python -m src.validate.roundtrip_check --root "$ROOT/v5"
python -m src.validate.box_sanity --root "$ROOT/v5"
python -m src.dataset_tools.stats --root "$ROOT/v5" --per-layout 400 --gate

echo "=== [3/4] DETECTOR --parallel (3x A5000)  $(TS) ==="
accelerate launch --multi_gpu --num_processes "$NPROC" --mixed_precision bf16 \
    -m src.detect.train_obj_detector --parallel --config obj_parallel_v5

echo "=== [4/4] STANDALONE EVAL (F1@.5/.75 + IoU + per-class threshold sweep)  $(TS) ==="
LATEST=$(ls -d "$ROOT"/checkpoints/detector_obj_v5_parallel/step_* | sort | tail -1)
python -m src.eval.det_eval --ckpt "$LATEST" --manifest data/manifests_v5/val.jsonl \
    --profile parallel --size 1280 --sweep --out "$ROOT/checkpoints/det_eval_v5_parallel.json"

if [ "$RECOGNIZER" = "1" ]; then
  echo "=== [opt] LINE CROPS 200K from V5 + RECOGNIZER refresh  $(TS) ==="
  python -m src.recognize.data.build_line_crops --pages 200000 --workers "$WORKERS" \
      --out-dir "$ROOT/recognize-v5/lines"
  # NOTE: set lr 5e-4 in the recognizer config first — 8e-4 caused the step-5000 CER spike on V4.
  accelerate launch --multi_gpu --num_processes "$NPROC" --mixed_precision bf16 \
      -m src.recognize.train_recognizer --parallel --config parallel_v3
fi

echo "=== V5 A5000 PIPELINE COMPLETE  $(TS) ==="
echo "detector -> $ROOT/checkpoints/detector_obj_v5_parallel/<latest>/model.pt"
echo "eval     -> $ROOT/checkpoints/det_eval_v5_parallel.json"
