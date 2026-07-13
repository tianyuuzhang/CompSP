#!/usr/bin/env bash
set -euo pipefail

# 运行 prefix-zero 隐蔽高风险模型的回答归属打乱负对照。
# 该对照在同一数据集、同一问题、同一 split 内打乱回答文本，保留 q1 和标签。
# 若打乱后 response 仍很强，说明模型可能主要依赖题目或攻击模板边际分布；
# 若性能明显下降，则支持“具体回答内容与后续风险存在对应关系”。

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/remote-home/zty/conda/LM_zty/repetition/bin/python}"
DATASET_KEYS="${DATASET_KEYS:-jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack}"
SAMPLE_SIZES="${SAMPLE_SIZES:-1,2,4}"
SEEDS="${SEEDS:-5}"
MAX_FEATURES="${MAX_FEATURES:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/response_safety_structure/sequential_asr_sampling}"
PREFIX="${PREFIX:-llama_three_attack_prefix_zero_hidden_risk_response_shuffle}"
SUFFIX="${SUFFIX:-seed5}"

mkdir -p "$OUTPUT_DIR" logs

START_TS="$(date +%Y%m%d_%H%M%S)"
echo "开始 prefix-zero 回答归属打乱负对照: ${DATASET_KEYS}"
PYTHONPATH="$PWD/src:$PWD/scripts" "$PYTHON_BIN" \
  scripts/analyze_prefix_zero_hidden_risk.py \
  --dataset-keys "$DATASET_KEYS" \
  --sample-sizes "$SAMPLE_SIZES" \
  --seeds "$SEEDS" \
  --high-threshold 0.5 \
  --top-fraction 0.2 \
  --max-features "$MAX_FEATURES" \
  --text-view response \
  --response-shuffle within_question_attack \
  --output "$OUTPUT_DIR/${PREFIX}_${SUFFIX}.json" \
  2>&1 | tee "logs/prefix_zero_response_shuffle_${START_TS}.log"

echo "prefix-zero 回答归属打乱负对照完成。"
