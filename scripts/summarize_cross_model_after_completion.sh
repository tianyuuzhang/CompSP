#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
while pgrep -f '^bash scripts/run_cross_model_ofa_response_matrix.sh$' >/dev/null; do
  sleep 30
done

export PYTHONPATH="$PWD/src"
exec /remote-home/zty/conda/LM_zty/repetition/bin/python scripts/summarize_response_transfer_matrix.py \
  --root outputs/response_safety_structure/cross_model_ofa_tfidf_n1 \
  --output-dir outputs/response_safety_structure/cross_model_ofa_tfidf_n1

