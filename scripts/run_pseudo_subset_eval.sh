#!/usr/bin/env bash
# 评估已完成的伪安全方向 CompSP 模型在原始 ALR 非边界子集上的表现。
# 默认只跑已经完成训练的 OFA 和 PAIR；DrAttack 训练完成后可通过 DATASETS 覆盖补跑。
set -euo pipefail
BASE_MODEL="${BASE_MODEL:-/remote-home/model/llama-3.1-8B/Llama-3.1-8B}"
PYTHON_BIN="${PYTHON_BIN:-/remote-home/zty/conda/LM_zty/repetition/bin/python}"
MODEL_ROOT="${MODEL_ROOT:-outputs/pseudo_safety_direction/models/mixed_alr_last_layer}"
PAIR_ROOT="${PAIR_ROOT:-outputs/pseudo_safety_direction/pairs/mixed_alr_last_layer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pseudo_safety_direction/subset_eval/mixed_alr_last_layer}"
DATASETS="${DATASETS:-jbb-llama-ofa jbb-llama-pair}"
BATCH_SIZE="${BATCH_SIZE:-8}"
mkdir -p "${OUTPUT_ROOT}"
for dataset_key in ${DATASETS}; do
  echo "==== ${dataset_key} $(date)"
  "${PYTHON_BIN}" scripts/evaluate_pseudo_direction_compsp_subsets.py \
    --base-model "${BASE_MODEL}" \
    --model-dir "${MODEL_ROOT}/${dataset_key}" \
    --test-file "${PAIR_ROOT}/${dataset_key}/test_pairs.json" \
    --output-file "${OUTPUT_ROOT}/${dataset_key}_alr_middle_eval.json" \
    --batch-size "${BATCH_SIZE}"
done