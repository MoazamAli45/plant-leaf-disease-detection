#!/usr/bin/env bash
# Phase 6 main sweep (docs/field.md §4 and §7). Run from ml/. Skips runs that already exist.
set -u
R=".venv/bin/python -m leafdx.field.run finetune"
for k in 5 10 20; do for s in 0 1 2; do $R --model p3_mobilenet_v1 --k $k --seed $s; done; done
for s in 0 1 2; do $R --model p3_mobilenet_v1 --seed $s --replay 50 --save-model; done
for s in 1 2; do $R --model p3_mobilenet_v1 --seed $s --save-model; done
for k in 5 10 20; do for s in 0 1 2; do $R --model p3_efficientnet_b0 --k $k --seed $s; done; done
for s in 0 1 2; do $R --model p3_efficientnet_b0 --seed $s; done
