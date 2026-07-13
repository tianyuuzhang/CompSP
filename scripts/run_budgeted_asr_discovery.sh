#!/usr/bin/env bash
set -euo pipefail

# 固定预算追加采样模拟：每条先看 k 次回答，再把额外访问预算给 top 候选。

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-/remote-home/zty/conda/LM_zty/repetition/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/response_safety_structure/sequential_asr_sampling}"
OUTPUT="${OUTPUT:-$OUTPUT_DIR/llama_three_attack_budgeted_asr_discovery_seed5.json}"

mkdir -p "$OUTPUT_DIR" logs
START_TS="$(date +%Y%m%d_%H%M%S)"

PYTHONPATH="$PWD/src:$PWD/scripts" "$PYTHON_BIN" \
  scripts/simulate_budgeted_asr_discovery.py \
  --output "$OUTPUT" \
  2>&1 | tee "logs/budgeted_asr_discovery_${START_TS}.log"
