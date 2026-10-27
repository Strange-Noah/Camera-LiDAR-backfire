#!/bin/bash
# pass3 推理重跑:chunk 已修复(0 missing)
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bevfusion
cd /root/mmdetection3d
CHUNK=/root/autodl-tmp/chunk
PRED=/root/autodl-tmp/preds
# 确保 sweep 软链仍在
rm -rf data/nuscenes_trainval/sweeps
ln -s $CHUNK/sweeps data/nuscenes_trainval/sweeps
for model in lidar fusion; do
  if [ "$model" = "lidar" ]; then
    CFG=projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
    CKPT=checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
  else
    CFG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
    CKPT=checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth
  fi
  echo "=== P3R_${model^^}_START $(date +%H:%M:%S) ==="
  python tools/test.py $CFG $CKPT \
    --cfg-options test_dataloader.dataset.data_root=$CHUNK/ \
    test_evaluator.ann_file=$CHUNK/nuscenes_infos_val.pkl \
    test_evaluator.format_only=True test_evaluator.jsonfile_prefix=$PRED/$model/pass3 \
    && echo "P3R_${model^^}_OK" || echo "P3R_${model^^}_FAIL"
done
echo "P3R_ALL_DONE $(date +%H:%M:%S)"
