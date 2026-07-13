#!/usr/bin/env bash
set -euo pipefail

# 运行 prefix-zero 强掩码 response 的低维 SVD 语义探针。
# 这是 frozen embedding 前的轻量实验：用 TF-IDF SVD 近似语义/姿态方向，
# 检查少数低维分量是否足以预测后缀高风险。

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/remote-home/zty/conda/LM_zty/repetition/bin/python}"
DATASET_KEYS="${DATASET_KEYS:-jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack}"
SAMPLE_SIZES="${SAMPLE_SIZES:-1,2,4}"
SEEDS="${SEEDS:-5}"
COMPONENTS="${COMPONENTS:-32}"
MAX_FEATURES="${MAX_FEATURES:-12000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/response_safety_structure/sequential_asr_sampling}"
OUTPUT="${OUTPUT:-$OUTPUT_DIR/llama_three_attack_prefix_zero_response_strong_mask_svd_seed5.json}"

mkdir -p "$OUTPUT_DIR" logs
START_TS="$(date +%Y%m%d_%H%M%S)"

PYTHONPATH="$PWD/src:$PWD/scripts" "$PYTHON_BIN" \
  scripts/analyze_prefix_zero_semantic_svd.py \
  --dataset-keys "$DATASET_KEYS" \
  --sample-sizes "$SAMPLE_SIZES" \
  --seeds "$SEEDS" \
  --components "$COMPONENTS" \
  --max-features "$MAX_FEATURES" \
  --output "$OUTPUT" \
  2>&1 | tee "logs/prefix_zero_semantic_svd_${START_TS}.log"
