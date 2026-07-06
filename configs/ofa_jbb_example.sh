#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
export PYTHONPATH="$PWD/src"

QUESTION_ID="${1:-9}"

python scripts/run_ofa.py \
  --stage init \
  --questions /remote-home/zty/data/jailbreak_questions.json \
  --question-id "$QUESTION_ID" \
  --start 0 \
  --end 75 \
  --tot-qa 20 \
  --output "data/work/Q_${QUESTION_ID}_0_75_init.json"

python scripts/run_ofa.py \
  --stage q1 \
  --input "data/work/Q_${QUESTION_ID}_0_75_init.json" \
  --output "data/work/Q_${QUESTION_ID}_0_75_q1.json" \
  --outline-model gpt-3.5-turbo-0125
