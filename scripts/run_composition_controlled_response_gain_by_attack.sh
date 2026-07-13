#!/usr/bin/env bash
set -euo pipefail

# 成分受控信息增益的目标与攻击方案分层实验。
# 目的：在固定前缀 unsafe/safe 成分后，同时预测未观察后缀 ASR 与 ALR；
# 并分别在 OFA、PAIR、DrAttack 三种攻击方案内复验 response 增量。
# 预计耗时：混合双目标约 45-75 分钟；三攻击分层合计约 40-80 分钟。
# 实际进度以日志中 “完成 composition=..., seed=...” 为准。

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/remote-home/zty/conda/LM_zty/repetition/bin/python}"
LOG_DIR="$ROOT_DIR/logs"
OUT_DIR="$ROOT_DIR/outputs/response_safety_structure/sequential_asr_sampling"
STAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR" "$OUT_DIR"

export PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR/scripts:${PYTHONPATH:-}"

run_one() {
  local label="$1"
  local dataset_keys="$2"
  local output="$OUT_DIR/llama_${label}_composition_controlled_gain_asr_alr_seed5.json"
  local log="$LOG_DIR/composition_controlled_${label}_asr_alr_${STAMP}.log"
  echo "开始 ${label}: ${dataset_keys}"
  "$PYTHON_BIN" "$ROOT_DIR/scripts/analyze_composition_controlled_response_gain.py" \
    --dataset-keys "$dataset_keys" \
    --compositions unsafe_only,safe_long_only,safe3_unsafe1 \
    --targets future_asr,future_alr \
    --seeds 5 \
    --max-features 6000 \
    --text-cleaning mask_strong_artifacts \
    --output "$output" \
    2>&1 | tee "$log"
}

run_one "three_attack" "jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack"
run_one "ofa" "jbb-llama-ofa"
run_one "pair" "jbb-llama-pair"
run_one "drattack" "jbb-llama-drattack"
