#!/usr/bin/env bash
set -euo pipefail

# 成分受控信息增益实验。
# 目的：固定前缀 unsafe/safe 成分后，检验回答文本是否仍能预测未观察后缀 ASR。
# 预计耗时：三种成分、5 seed、6000 TF-IDF 特征约 35-65 分钟；实际以日志进度为准。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/remote-home/zty/conda/LM_zty/repetition/bin/python}"
LOG_DIR="$ROOT_DIR/logs"
OUT_DIR="$ROOT_DIR/outputs/response_safety_structure/sequential_asr_sampling"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$OUT_DIR"

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/scripts:${PYTHONPATH:-}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/analyze_composition_controlled_response_gain.py" \
  --dataset-keys jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack \
  --compositions unsafe_only,safe_long_only,safe3_unsafe1 \
  --targets future_asr \
  --seeds 5 \
  --max-features 6000 \
  --text-cleaning mask_strong_artifacts \
  --output "$OUT_DIR/llama_three_attack_composition_controlled_gain_seed5.json" \
  2>&1 | tee "$LOG_DIR/composition_controlled_response_gain_${STAMP}.log"
