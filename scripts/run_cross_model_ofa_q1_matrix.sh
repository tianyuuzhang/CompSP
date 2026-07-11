#!/usr/bin/env bash
# JBB-OFA 四目标模型的 q1-only TF-IDF 跨模型基线矩阵。
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
export PYTHONPATH="$PWD/src"
PY="/remote-home/zty/conda/LM_zty/repetition/bin/python"
SCORES="outputs/alternative_asr/all_14_datasets/alternative_asr_scores.jsonl"
OUT="outputs/response_safety_structure/cross_model_ofa_q1_tfidf_n1"

MODELS=(llama qwen 4omini deepseek)
declare -A DATASET=(
  [llama]="jbb-llama-ofa"
  [qwen]="jbb-qwen-ofa"
  [4omini]="jbb-4omini-ofa"
  [deepseek]="jbb-deepseek-ofa"
)

mkdir -p "$OUT"
for SOURCE in "${MODELS[@]}"; do
  for TARGET in "${MODELS[@]}"; do
    CELL="$OUT/${SOURCE}_to_${TARGET}"
    if [[ -s "$CELL/response_safety_structure_report.json" ]]; then
      echo "跳过已有矩阵单元：${SOURCE}->${TARGET}"
      continue
    fi
    echo "开始 q1 基线单元：${SOURCE}->${TARGET}"
    "$PY" scripts/analyze_response_safety_structure.py \
      --scores "$SCORES" \
      --sample-sizes 1 \
      --targets asr,hazard_weighted_asr,alr \
      --methods tfidf \
      --text-views q1 \
      --train-datasets "${DATASET[$SOURCE]}" \
      --test-datasets "${DATASET[$TARGET]}" \
      --max-features 30000 \
      --output-dir "$CELL"
  done
done

echo "JBB-OFA 四模型 q1 基线矩阵完成。"

