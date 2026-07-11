#!/usr/bin/env bash
# 等待 DrAttack ALR 子集评估释放 GPU 0，再接力启动回答 hidden embedding 实验。
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
PY="/remote-home/zty/conda/LM_zty/repetition/bin/python"
SUBSET_OUTPUT="outputs/pseudo_safety_direction/subset_eval/mixed_alr_last_layer/jbb-llama-drattack_alr_middle_eval.json"
OUTPUT_DIR="outputs/response_safety_structure/hidden_embedding_alr_direction_n1"

while pgrep -f "evaluate_pseudo_direction_compsp_subsets.py.*jbb-llama-drattack" >/dev/null; do
  sleep 30
done

if [[ ! -s "$SUBSET_OUTPUT" ]]; then
  echo "DrAttack 子集评估进程已退出，但结果文件不存在，拒绝接力启动。"
  exit 1
fi
if [[ -s "$OUTPUT_DIR/response_hidden_embedding_report.json" ]]; then
  echo "回答 hidden embedding 结果已经存在，未重复运行。"
  exit 0
fi

echo "GPU 0 已释放，开始回答 hidden embedding 实验；预计 1.5~2.5 小时，以 tqdm ETA 为准。"
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0
exec "$PY" scripts/analyze_response_hidden_embeddings.py \
  --scores outputs/pseudo_safety_direction/mixed_alr_last_layer_scores.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 4 \
  --max-length 512

