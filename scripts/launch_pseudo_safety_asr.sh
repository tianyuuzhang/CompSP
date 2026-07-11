#!/usr/bin/env bash
# 后台启动 ASR 伪安全结构实验。重复执行时会拒绝覆盖已有完整结果。
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
mkdir -p logs outputs/pseudo_safety_direction

PREFIX="mixed_asr_last_layer"
if [[ -f "outputs/pseudo_safety_direction/models/${PREFIX}/jbb-llama-drattack/train_summary.json" ]]; then
  echo "ASR 三组结果已经完整存在，未重复启动。"
  exit 0
fi

if pgrep -af "run_pseudo_safety_learnability.sh|train_pseudo_direction_compsp.py" | grep -q "asr\|${PREFIX}"; then
  echo "检测到 ASR 实验已经运行，未重复启动。"
  pgrep -af "run_pseudo_safety_learnability.sh|train_pseudo_direction_compsp.py"
  exit 0
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pseudo_safety_asr_${TS}.log"
METRIC=asr SKIP_EXISTING=1 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
  nohup scripts/run_pseudo_safety_learnability.sh >"${LOG}" 2>&1 &
PID=$!
echo "ASR 实验已启动：PID=${PID}"
echo "日志：${LOG}"
echo "按 ALR 三组实测速度预计总时长 45~55 小时；产生 tqdm 后以日志 ETA 为准。"
