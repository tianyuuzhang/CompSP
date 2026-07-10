#!/usr/bin/env bash
# 在远端后台启动 ALR 非边界子集评估，并打印 PID、日志和预计耗时。
set -euo pipefail
cd /remote-home/zty/tidy_CompSP
mkdir -p logs
TS=$(date +%Y%m%d_%H%M%S)
LOG="logs/pseudo_subset_eval_${TS}.log"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" nohup scripts/run_pseudo_subset_eval.sh > "${LOG}" 2>&1 &
PID=$!
echo "PID=${PID}"
echo "LOG=${LOG}"
echo "预计耗时：OFA+PAIR 两组约 40-60 分钟；实际以 tqdm ETA 为准。"