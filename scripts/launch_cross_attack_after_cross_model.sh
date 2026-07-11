#!/usr/bin/env bash
# 等待跨模型 OFA 矩阵完成后接力运行 Llama 跨攻击/混合矩阵。
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
while pgrep -f '^bash scripts/run_cross_model_ofa_response_matrix.sh$' >/dev/null; do
  sleep 30
done

exec scripts/run_cross_attack_llama_response_matrix.sh

