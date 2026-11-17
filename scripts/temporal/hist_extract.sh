#!/bin/bash
# 阶段A:从 10 个 blob 抽取全部历史相机图到 camhist(常驻)
set -x
SRC=/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
HIST=/root/autodl-tmp/camhist
mkdir -p $HIST
for i in 01 02 03 04 05 06 07 08 09 10; do
  echo "=== hist blob $i start $(date +%H:%M:%S) ==="
  tar -xzf $SRC/v1.0-trainval${i}_blobs.tgz -C $HIST -T /root/autodl-tmp/hist_patterns_full.txt
  echo "=== hist blob $i done $(date +%H:%M:%S) ==="
done
echo "HIST_EXTRACT_ALL_DONE $(date +%H:%M:%S)"
find $HIST -type f | wc -l
