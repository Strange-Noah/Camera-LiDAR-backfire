#!/bin/bash
# Paired-subset eval for E5 vs E2 (all 21 files: clean fusion, LiDAR-only,
# e2_k1..9, e5_k1..9). Common subset = intersection over all -> identical
# sample set for every number. Run AFTER cycle_e5.py prints ALL_DONE.
# 2026-11-03: subset_eval.py reads e2 .gz files transparently; no gunzip needed.
set -o pipefail
PY=/root/miniconda3/envs/bevfusion/bin/python
P=/root/autodl-tmp/preds_temporal
FILES="/root/autodl-tmp/preds/fusion_cumulative.json /root/autodl-tmp/preds/lidar_cumulative.json"
for k in 1 2 3 4 5 6 7 8 9; do
  FILES="$FILES $P/e2_k${k}_cumulative.json $P/e5_k${k}_cumulative.json"
done
cd /root/mmdetection3d
$PY /root/subset_eval.py $FILES 2>/root/autodl-tmp/e5_subset_eval.err | tee /root/autodl-tmp/e5_subset_eval.log
