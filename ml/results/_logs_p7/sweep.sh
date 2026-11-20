#!/usr/bin/env bash
# Phase 7 sweep (docs/field2.md §4). Run from ml/. Existing runs are skipped.
set -euo pipefail
PY=.venv/bin/python
$PY -m leafdx.field.phase7 reference
for model in p3_mobilenet_v1 p3_efficientnet_b0; do
  for method in P PF; do
    for seed in 0 1 2; do
      $PY -m leafdx.field.phase7 run --model $model --method $method --seed $seed
    done
  done
done
