#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false

PY="/remote-home/zty/conda/LM_zty/repetition/bin/python"
WHITEBOX_MODEL="/remote-home/model/llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct"
BASE_MODEL="/remote-home/model/llama-3.1-8B/Llama-3.1-8B"
OUT="outputs/pseudo_safety_direction"
LOG_DIR="logs"
mkdir -p "$OUT" "$LOG_DIR"

METRIC="${METRIC:-alr}"
PREFIX="mixed_${METRIC}_last_layer"
DATASETS="jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack"
MAX_ITEMS_ARG=()

if [[ "${SMOKE:-0}" == "1" ]]; then
  PREFIX="smoke_${METRIC}_last_layer"
  MAX_ITEMS_ARG=(--max-items-per-question 8 --train-qids 0,2,3 --test-qids 9,30)
  EPOCHS="${EPOCHS:-1}"
  BATCH_SIZE="${BATCH_SIZE:-2}"
  GRAD_ACC="${GRAD_ACC:-1}"
else
  EPOCHS="${EPOCHS:-2}"
  BATCH_SIZE="${BATCH_SIZE:-8}"
  GRAD_ACC="${GRAD_ACC:-2}"
fi

"$PY" scripts/build_pseudo_safety_direction.py \
  --model "$WHITEBOX_MODEL" \
  --dataset-keys "$DATASETS" \
  --metric "$METRIC" \
  --batch-size 2 \
  --max-length 512 \
  --output-dir "$OUT" \
  --prefix "$PREFIX" \
  "${MAX_ITEMS_ARG[@]}"

for DATASET in jbb-llama-ofa jbb-llama-pair jbb-llama-drattack; do
  "$PY" scripts/build_pseudo_safety_pairs.py \
    --scores "$OUT/${PREFIX}_scores.jsonl" \
    --dataset-key "$DATASET" \
    --min-delta 0.15 \
    --max-train-pairs-per-question 2000 \
    --output-dir "$OUT/pairs/${PREFIX}"

  "$PY" scripts/train_pseudo_direction_compsp.py \
    --base-model "$BASE_MODEL" \
    --train-file "$OUT/pairs/${PREFIX}/${DATASET}/train_pairs.json" \
    --test-file "$OUT/pairs/${PREFIX}/${DATASET}/test_pairs.json" \
    --output-dir "$OUT/models/${PREFIX}/${DATASET}" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRAD_ACC" \
    --learning-rate 2e-4

done
