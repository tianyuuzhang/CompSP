#!/usr/bin/env bash
set -euo pipefail

cd /remote-home/zty/tidy_CompSP
while pgrep -f '^bash scripts/run_cross_model_ofa_response_matrix.sh$' >/dev/null; do
  sleep 30
done

exec scripts/run_cross_model_ofa_q1_matrix.sh

