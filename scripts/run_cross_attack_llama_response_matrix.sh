#!/usr/bin/env bash
# Llama 上 OFA/PAIR/DrAttack/混合数据的单回答 TF-IDF 迁移矩阵。
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
export PYTHONPATH="$PWD/src"
PY="/remote-home/zty/conda/LM_zty/repetition/bin/python"
SCORES="outputs/alternative_asr/all_14_datasets/alternative_asr_scores.jsonl"
OUT="outputs/response_safety_structure/cross_attack_llama_tfidf_n1"

DOMAINS=(ofa pair drattack mixed)
declare -A DATASETS=(
  [ofa]="jbb-llama-ofa"
  [pair]="jbb-llama-pair"
  [drattack]="jbb-llama-drattack"
  [mixed]="jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack"
)

mkdir -p "$OUT"
for SOURCE in "${DOMAINS[@]}"; do
  for TARGET in "${DOMAINS[@]}"; do
    CELL="$OUT/${SOURCE}_to_${TARGET}"
    if [[ -s "$CELL/response_safety_structure_report.json" ]]; then
      echo "跳过已有矩阵单元：${SOURCE}->${TARGET}"
      continue
    fi
    echo "开始矩阵单元：${SOURCE}->${TARGET}"
    "$PY" scripts/analyze_response_safety_structure.py \
      --scores "$SCORES" \
      --sample-sizes 1 \
      --targets asr,hazard_weighted_asr,alr \
      --methods tfidf \
      --text-views response \
      --train-datasets "${DATASETS[$SOURCE]}" \
      --test-datasets "${DATASETS[$TARGET]}" \
      --max-features 30000 \
      --output-dir "$CELL"
  done
done

echo "Llama 跨攻击/混合迁移矩阵完成。"

