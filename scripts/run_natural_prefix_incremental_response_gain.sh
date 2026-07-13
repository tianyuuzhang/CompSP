#!/usr/bin/env bash
set -euo pipefail

# 自然前缀回答文本条件增量实验。
# 目的：不事后挑 safe/unsafe 组成，只取自然前 k 条回答；
# 比较 count+q1 与 count+q1+response 对未观察后缀 ASR 的条件增量。
# 预计耗时：k=1,2,4；5 seed；12000 特征；200 bootstrap 约 60-120 分钟。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/remote-home/zty/conda/LM_zty/repetition/bin/python}"
LOG_DIR="$ROOT_DIR/logs"
OUT_DIR="$ROOT_DIR/outputs/response_safety_structure/sequential_asr_sampling"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$OUT_DIR"

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/scripts:${PYTHONPATH:-}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/analyze_natural_prefix_incremental_response_gain.py" \
  --dataset-keys jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack \
  --sample-sizes 1,2,4 \
  --targets future_asr \
  --seeds 5 \
  --bootstrap 200 \
  --max-features 12000 \
  --text-cleaning mask_strong_artifacts \
  --output "$OUT_DIR/llama_three_attack_natural_prefix_incremental_gain_seed5.json" \
  2>&1 | tee "$LOG_DIR/natural_prefix_incremental_response_gain_${STAMP}.log"
