#!/bin/bash
# 自动链:等阶段A完成 → pilot(blob01,含clean恒等) → pilot_check → 全量10 blob
echo "CHAIN_START $(date +%H:%M:%S)"
while ! grep -q HIST_EXTRACT_ALL_DONE /root/autodl-tmp/hist_extract.log; do
  sleep 60
done
echo "CHAIN_HIST_READY $(date +%H:%M:%S)"
LOG_NAME=pilot_pipeline ONLY_BLOBS=01 RUN_CLEAN=1 bash /root/temporal_pipeline.sh
echo "CHAIN_PILOT_INF_DONE $(date +%H:%M:%S)"
mkdir -p /root/autodl-tmp/pilot_pkls
cp /root/autodl-tmp/chunk/nuscenes_infos_val.pkl /root/autodl-tmp/pilot_pkls/ 2>/dev/null
cp /root/autodl-tmp/chunk/nuscenes_infos_val_k1.pkl /root/autodl-tmp/pilot_pkls/ 2>/dev/null
cp /root/autodl-tmp/chunk/nuscenes_infos_val_k2.pkl /root/autodl-tmp/pilot_pkls/ 2>/dev/null
/root/miniconda3/envs/bevfusion/bin/python /root/pilot_check.py
echo "CHAIN_PILOT_CHECK_DONE $(date +%H:%M:%S)"
LOG_NAME=temporal_pipeline bash /root/temporal_pipeline.sh
echo "CHAIN_ALL_DONE $(date +%H:%M:%S)"
