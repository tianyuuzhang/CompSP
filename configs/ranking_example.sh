#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
export PYTHONPATH="$PWD/src"

python scripts/run_pairwise_matrix.py \
  --input data/work/Q_9_0_75_txfilled.json \
  --output data/work/Q_9_0_24pair.json \
  --base-model /remote-home/model/llama-3.1-8B/Llama-3.1-8B \
  --lora /remote-home/model/llama-3.1-8B/llama-3.1-8B-ZTY-classification_qwen_80Q_num75_train60_test20_500_tx_ratio_ge_0_2026-04-04-15/lora \
  --start 0 \
  --end 24

python scripts/run_bt_rank.py \
  --matrix data/work/Q_9_0_24pair.json \
  --q-id 9 \
  --start-id 0 \
  --output data/work/Q_9_0_24_ranking_scores.json
