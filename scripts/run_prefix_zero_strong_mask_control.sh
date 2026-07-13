#!/usr/bin/env bash
set -euo pipefail

# 运行 prefix-zero 隐蔽高风险模型的强掩码负对照。
# 在拒绝词/危险词掩码基础上，进一步掩掉常见模板词、格式词和
# 高权重审计中反复出现的占位/诱饵词，用来检查 response 信号
# 是否主要来自明显词面捷径。

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/remote-home/zty/conda/LM_zty/repetition/bin/python}"
DATASET_KEYS="${DATASET_KEYS:-jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack}"
SAMPLE_SIZES="${SAMPLE_SIZES:-1,2,4}"
SEEDS="${SEEDS:-5}"
MAX_FEATURES="${MAX_FEATURES:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/response_safety_structure/sequential_asr_sampling}"
PREFIX="${PREFIX:-llama_three_attack_prefix_zero_hidden_risk_response_strong_mask}"
SUFFIX="${SUFFIX:-seed5}"

mkdir -p "$OUTPUT_DIR" logs

START_TS="$(date +%Y%m%d_%H%M%S)"
echo "开始 prefix-zero 强掩码负对照: ${DATASET_KEYS}"
PYTHONPATH="$PWD/src:$PWD/scripts" "$PYTHON_BIN" \
  scripts/analyze_prefix_zero_hidden_risk.py \
  --dataset-keys "$DATASET_KEYS" \
  --sample-sizes "$SAMPLE_SIZES" \
  --seeds "$SEEDS" \
  --high-threshold 0.5 \
  --top-fraction 0.2 \
  --max-features "$MAX_FEATURES" \
  --text-view response \
  --text-cleaning mask_strong_artifacts \
  --output "$OUTPUT_DIR/${PREFIX}_${SUFFIX}.json" \
  2>&1 | tee "logs/prefix_zero_strong_mask_${START_TS}.log"

echo "prefix-zero 强掩码负对照完成。"
