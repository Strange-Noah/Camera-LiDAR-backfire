#!/bin/bash
# 从 10 个 trainval blob 中选择性解压 val 场景(18 个日志前缀)的样本与 LiDAR 帧
exec > /root/autodl-tmp/extract_val.log 2>&1
set -x

DST=/root/autodl-tmp/nuscenes_val
SRC=/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval

mkdir -p $DST/v1.0-trainval
cp -r /root/autodl-tmp/nuscenes_meta/maps $DST/
cp /root/autodl-tmp/nuscenes_meta/v1.0-trainval/*.json $DST/v1.0-trainval/

# 生成 tar 通配符清单
P=/root/autodl-tmp/tar_patterns.txt
: > $P
while read pre; do
  for d in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT LIDAR_TOP; do
    echo "samples/$d/${pre}*" >> $P
  done
  echo "sweeps/LIDAR_TOP/${pre}*" >> $P
done < /root/autodl-tmp/val_log_prefixes.txt
wc -l $P

for i in 01 02 03 04 05 06 07 08 09 10; do
  echo "=== blob $i start $(date) ==="
  df -h /root/autodl-tmp | tail -1
  tar -xzf $SRC/v1.0-trainval${i}_blobs.tgz -C $DST --wildcards -T $P
  echo "=== blob $i done $(date) ==="
done
echo "EXTRACT_ALL_DONE $(date)"
du -sh $DST
df -h /root/autodl-tmp | tail -1
