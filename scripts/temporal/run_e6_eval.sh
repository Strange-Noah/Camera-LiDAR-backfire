#!/bin/bash
# Paired-subset eval for E6 joint grid (9 cells) + T1 camera-only points (k1..10)
# + E2 lidar-only points (k1..9) + baselines, ALL on one common subset so every
# number is directly comparable. Run AFTER cycle_e6.py prints ALL_DONE.
set -eo pipefail
PY=/root/miniconda3/envs/bevfusion/bin/python
P=/root/autodl-tmp/preds_temporal
FILES="/root/autodl-tmp/preds/fusion_cumulative.json /root/autodl-tmp/preds/lidar_cumulative.json"
for kc in 2 5 8; do
  for kl in 1 3 6; do
    FILES="$FILES $P/e6_c${kc}_l${kl}_cumulative.json"
  done
done
cd /root/mmdetection3d
$PY /root/subset_eval.py $FILES 2>/root/autodl-tmp/e6_subset_eval.err | tee /root/autodl-tmp/e6_subset_eval.log
