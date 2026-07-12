#!/usr/bin/env bash
set -euo pipefail

# 运行 prefix-zero 隐蔽高风险模型的三视图对照。
# 目的：区分“回答文本本身”相对“原始攻击指令 q1”的增量信号。
# 输出：
# - response：只使用前缀回答文本；
# - q1：只使用原始攻击指令；
# - joint：拼接 q1 与前缀回答。

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/remote-home/zty/conda/LM_zty/repetition/bin/python}"
DATASET_KEYS="${DATASET_KEYS:-jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack}"
SAMPLE_SIZES="${SAMPLE_SIZES:-1,2,4}"
SEEDS="${SEEDS:-1}"
MAX_FEATURES="${MAX_FEATURES:-8000}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/response_safety_structure/sequential_asr_sampling}"

mkdir -p "$OUTPUT_DIR" logs

for TEXT_VIEW in response q1 joint; do
  echo "开始 prefix-zero 三视图对照: text_view=${TEXT_VIEW}"
  START_TS="$(date +%Y%m%d_%H%M%S)"
  PYTHONPATH="$PWD/src:$PWD/scripts" "$PYTHON_BIN" \
    scripts/analyze_prefix_zero_hidden_risk.py \
    --dataset-keys "$DATASET_KEYS" \
    --sample-sizes "$SAMPLE_SIZES" \
    --seeds "$SEEDS" \
    --high-threshold 0.5 \
    --top-fraction 0.2 \
    --max-features "$MAX_FEATURES" \
    --text-view "$TEXT_VIEW" \
    --output "$OUTPUT_DIR/llama_three_attack_prefix_zero_hidden_risk_${TEXT_VIEW}_fast.json" \
    2>&1 | tee "logs/prefix_zero_text_view_${TEXT_VIEW}_${START_TS}.log"
done

PYTHONPATH="$PWD/src:$PWD/scripts" "$PYTHON_BIN" \
  scripts/summarize_prefix_zero_text_view_ablation.py \
  --input-dir "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/llama_three_attack_prefix_zero_hidden_risk_text_view_summary.json"

echo "prefix-zero 三视图对照完成。"
