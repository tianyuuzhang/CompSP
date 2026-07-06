#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TOKENIZERS_PARALLELISM=false

PY="/remote-home/zty/conda/LM_zty/repetition/bin/python"
MODEL="/remote-home/model/llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct"
QIDS="9,30,73,10,37,15,42,1,50,41,45,65,70,72,21,55,56,78,48,76"
OUT="outputs/whitebox"

mkdir -p "$OUT" logs

"$PY" scripts/build_refusal_direction.py \
  --model "$MODEL" \
  --n 80 \
  --batch-size 2 \
  --max-length 512 \
  --output "$OUT/llama31_instruct_refusal_direction_n80.pt"

for DATASET in jbb-llama-pair jbb-llama-drattack; do
  "$PY" scripts/score_whitebox_q1.py \
    --model "$MODEL" \
    --direction "$OUT/llama31_instruct_refusal_direction_n80.pt" \
    --dataset-key "$DATASET" \
    --question-ids "$QIDS" \
    --batch-size 2 \
    --max-length 512 \
    --output "$OUT/${DATASET}_whitebox_scores.jsonl"

  "$PY" scripts/analyze_whitebox_compsp_alignment.py \
    --scores "$OUT/${DATASET}_whitebox_scores.jsonl" \
    --dataset-key "$DATASET" \
    --metric alr \
    --output "$OUT/${DATASET}_alignment_alr.json"

  "$PY" scripts/analyze_whitebox_compsp_alignment.py \
    --scores "$OUT/${DATASET}_whitebox_scores.jsonl" \
    --dataset-key "$DATASET" \
    --metric asr \
    --output "$OUT/${DATASET}_alignment_asr.json"
done
