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

if [[ "${SKIP_EXISTING:-0}" == "1" && -s "$OUT/${PREFIX}_scores.jsonl" && -s "$OUT/${PREFIX}_direction_report.json" ]]; then
  echo "跳过已有方向和投影：$PREFIX"
else
  "$PY" scripts/build_pseudo_safety_direction.py \
    --model "$WHITEBOX_MODEL" \
    --dataset-keys "$DATASETS" \
    --metric "$METRIC" \
    --batch-size 2 \
    --max-length 512 \
    --output-dir "$OUT" \
    --prefix "$PREFIX" \
    "${MAX_ITEMS_ARG[@]}"
fi

for DATASET in jbb-llama-ofa jbb-llama-pair jbb-llama-drattack; do
  PAIR_DIR="$OUT/pairs/${PREFIX}/${DATASET}"
  MODEL_DIR="$OUT/models/${PREFIX}/${DATASET}"
  if [[ "${SKIP_EXISTING:-0}" == "1" && -s "$PAIR_DIR/train_pairs.json" && -s "$PAIR_DIR/test_pairs.json" ]]; then
    echo "跳过已有 pair 数据：$DATASET"
  else
    "$PY" scripts/build_pseudo_safety_pairs.py \
      --scores "$OUT/${PREFIX}_scores.jsonl" \
      --dataset-key "$DATASET" \
      --min-delta 0.15 \
      --max-train-pairs-per-question 2000 \
      --output-dir "$OUT/pairs/${PREFIX}"
  fi

  if [[ "${SKIP_EXISTING:-0}" == "1" && -s "$MODEL_DIR/train_summary.json" ]]; then
    echo "跳过已有完整模型：$DATASET"
  else
    "$PY" scripts/train_pseudo_direction_compsp.py \
      --base-model "$BASE_MODEL" \
      --train-file "$PAIR_DIR/train_pairs.json" \
      --test-file "$PAIR_DIR/test_pairs.json" \
      --output-dir "$MODEL_DIR" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --gradient-accumulation-steps "$GRAD_ACC" \
      --learning-rate 2e-4
  fi

done
