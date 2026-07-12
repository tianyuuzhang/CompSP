#!/usr/bin/env bash
# JBB-OFA 四目标模型的回答残差信号跨模型迁移矩阵。
# 该实验先用长度、拒绝和格式特征拟合目标，再用遮蔽后的回答 TF-IDF 拟合残差，
# 用于检验回答文本是否在朴素长度/模板之外仍有跨模型可迁移信息。
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
export PYTHONPATH="$PWD/src"
PY="/remote-home/zty/conda/LM_zty/repetition/bin/python"
SCORES="outputs/alternative_asr/all_14_datasets/alternative_asr_scores.jsonl"
OUT="outputs/response_safety_structure/cross_model_ofa_response_residual_masked_n1"
LOG_DIR="logs"
LOG_FILE="$LOG_DIR/cross_model_ofa_response_residual_$(date +%Y%m%d_%H%M%S).log"

MODELS=(llama qwen 4omini deepseek)
declare -A DATASET=(
  [llama]="jbb-llama-ofa"
  [qwen]="jbb-qwen-ofa"
  [4omini]="jbb-4omini-ofa"
  [deepseek]="jbb-deepseek-ofa"
)

mkdir -p "$OUT" "$LOG_DIR"
echo "日志：$LOG_FILE" | tee "$LOG_FILE"
echo "预计耗时：20~60 分钟；每个单元完成后会打印输出路径。" | tee -a "$LOG_FILE"

for SOURCE in "${MODELS[@]}"; do
  for TARGET in "${MODELS[@]}"; do
    CELL="$OUT/${SOURCE}_to_${TARGET}"
    if [[ -s "$CELL/response_residual_signal_report.json" ]]; then
      echo "跳过已有残差矩阵单元：${SOURCE}->${TARGET}" | tee -a "$LOG_FILE"
      continue
    fi
    echo "开始残差矩阵单元：${SOURCE}->${TARGET} $(date '+%F %T')" | tee -a "$LOG_FILE"
    "$PY" scripts/analyze_response_residual_signal.py \
      --scores "$SCORES" \
      --sample-size 1 \
      --targets asr,hazard_weighted_asr,alr \
      --text-cleaning mask_refusal_hazard_terms \
      --train-datasets "${DATASET[$SOURCE]}" \
      --test-datasets "${DATASET[$TARGET]}" \
      --max-features 30000 \
      --output-dir "$CELL" 2>&1 | tee -a "$LOG_FILE"
  done
done

"$PY" scripts/summarize_response_residual_transfer_matrix.py \
  --root "$OUT" \
  --output-dir "$OUT" 2>&1 | tee -a "$LOG_FILE"

echo "JBB-OFA 四模型遮蔽残差迁移矩阵完成：$OUT" | tee -a "$LOG_FILE"
