#!/usr/bin/env bash
set -euo pipefail

# 固定追加访问预算的序贯 ASR 发现曲线。
# 运行方式：
#   bash scripts/run_fixed_extra_budget_asr_discovery.sh
# 日志写入 logs，结果写入 outputs/response_safety_structure/sequential_asr_sampling。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/remote-home/zty/conda/LM_zty/repetition/bin/python"
OUTPUT_DIR="$ROOT/outputs/response_safety_structure/sequential_asr_sampling"
LOG_DIR="$ROOT/logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

export PYTHONPATH="$ROOT/src:$ROOT/scripts"

nohup "$PYTHON" "$ROOT/scripts/simulate_fixed_extra_budget_asr_discovery.py" \
  --dataset-keys jbb-llama-ofa,jbb-llama-pair,jbb-llama-drattack \
  --sample-sizes 1,2,4 \
  --seeds 5 \
  --extra-budget-per-row 1,2,3,4 \
  --hidden-quotas 0.25,0.5 \
  --high-threshold 0.5 \
  --text-cleaning mask_strong_artifacts \
  --max-features 12000 \
  --output "$OUTPUT_DIR/llama_three_attack_fixed_extra_budget_curve_seed5.json" \
  > "$LOG_DIR/fixed_extra_budget_asr_discovery_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

echo "已启动固定 extra budget 曲线实验，PID=$!"
echo "预计耗时：35-75 分钟。可用 tail -f logs/fixed_extra_budget_asr_discovery_*.log 查看进度。"
