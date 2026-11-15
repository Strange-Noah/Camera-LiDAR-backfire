#!/bin/bash
# 阶段B:temporal T1 全量推理 —— 逐 blob 组 chunk → k1/k2 错位 pkl → 仅 fusion 推理 → 累计
# 环境变量 ONLY_BLOBS 可限制只跑指定 blob(如 "01" 做 pilot)
exec > /root/autodl-tmp/${LOG_NAME:-temporal_pipeline}.log 2>&1
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bevfusion
cd /root/mmdetection3d

STG=/root/autodl-tmp/staging
CHUNK=/root/autodl-tmp/chunk
P=/root/autodl-tmp/tar_patterns.txt
PRED=/root/autodl-tmp/preds_temporal
HIST=/root/autodl-tmp/camhist
SRC=/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
shopt -s nullglob

mkdir -p $CHUNK/samples $CHUNK/sweeps $PRED/k1 $PRED/k2
ln -sfn /root/autodl-tmp/nuscenes_meta/maps $CHUNK/maps
ln -sfn /root/autodl-tmp/nuscenes_meta/v1.0-trainval $CHUNK/v1.0-trainval

FUSION_CFG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
FUSION_CKPT=checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth

# base chunk pkl:跳过已完成的 token(_done.json 为权威清单)
cat > /root/filter_chunk_t.py <<'PYEOF'
import pickle, os, sys, json
STG, CHUNK, PRED = sys.argv[1], sys.argv[2], sys.argv[3]
info = pickle.load(open('data/nuscenes_trainval/nuscenes_infos_val.pkl', 'rb'))
donep = f'{PRED}/_done.json'
done = set(json.load(open(donep))) if os.path.exists(donep) else set()
def have(sub, name):
    return os.path.exists(os.path.join(STG, sub, name))
kept = []
for d in info['data_list']:
    if d['token'] in done:
        continue
    if not have('samples/LIDAR_TOP', d['lidar_points']['lidar_path']):
        continue
    ok = True
    for cam, im in d['images'].items():
        if not have(f'samples/{cam}', im['img_path']):
            ok = False; break
    if not ok:
        continue
    for sw in d.get('lidar_sweeps', []):
        if not have('sweeps/LIDAR_TOP', os.path.basename(sw['lidar_points']['lidar_path'])):
            ok = False; break
    if ok:
        kept.append(d)
for i, d in enumerate(kept):
    d['sample_idx'] = i
out = dict(metainfo=info['metainfo'], data_list=kept)
pickle.dump(out, open(f'{CHUNK}/nuscenes_infos_val.pkl', 'wb'))
print('CHUNK_SAMPLES', len(kept))
PYEOF

cat > /root/accum_t.py <<'PYEOF'
import json, os, sys
PRED, pre, K = sys.argv[1], sys.argv[2], sys.argv[3]
src = f'{PRED}/k{K}/{pre}/pred_instances_3d/results_nusc.json'
cum_path = f'{PRED}/k{K}_cumulative.json'
done_path = f'{PRED}/_done.json'
new = json.load(open(src))
cum = json.load(open(cum_path)) if os.path.exists(cum_path) else {'meta': new.get('meta', {}), 'results': {}}
done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
before = len(cum['results'])
cum['results'].update(new['results'])
done.update(new['results'].keys())
json.dump(cum, open(cum_path, 'w'))
json.dump(sorted(done), open(done_path, 'w'))
print('ACCUM_T', pre, 'K', K, 'NEW', len(new['results']), 'TOTAL', len(cum['results']), 'DUP', len(new['results']) - (len(cum['results']) - before))
PYEOF

for i in ${ONLY_BLOBS:-01 02 03 04 05 06 07 08 09 10}; do
  rm -rf $STG; mkdir -p $STG
  echo "=== T blob $i extract start $(date +%H:%M:%S) ==="
  tar -xzf $SRC/v1.0-trainval${i}_blobs.tgz -C $STG --wildcards -T $P
  echo "=== T blob $i extract done $(date +%H:%M:%S) ==="
  python /root/filter_chunk_t.py $STG $CHUNK $PRED
  N=$(python -c "import pickle; print(len(pickle.load(open('$CHUNK/nuscenes_infos_val.pkl','rb'))['data_list']))")
  echo "T_BLOB $i USABLE_SAMPLES $N"
  if [ "$N" -gt 0 ]; then
    rm -rf $CHUNK/sweeps
    mv $STG/samples/* $CHUNK/samples/ 2>/dev/null
    mv $STG/sweeps $CHUNK/sweeps
    rm -rf data/nuscenes_trainval/sweeps
    ln -s $CHUNK/sweeps data/nuscenes_trainval/sweeps
    for K in 1 2; do
      python /root/make_kpkl.py $CHUNK $HIST /root/autodl-tmp/sub_table.json $K
      NK=$(python -c "import pickle; print(len(pickle.load(open('$CHUNK/nuscenes_infos_val_k$K.pkl','rb'))['data_list']))")
      echo "T_BLOB $i K$K USABLE $NK"
      if [ "$NK" -gt 0 ]; then
        python tools/test.py $FUSION_CFG $FUSION_CKPT \
          --cfg-options test_dataloader.dataset.data_root=$CHUNK/ \
          test_dataloader.dataset.ann_file=$CHUNK/nuscenes_infos_val_k$K.pkl \
          test_evaluator.ann_file=$CHUNK/nuscenes_infos_val_k$K.pkl \
          test_evaluator.format_only=True test_evaluator.jsonfile_prefix=$PRED/k$K/blob$i \
          && python /root/accum_t.py $PRED blob$i $K \
          || echo "T_FUSION_FAIL blob$i k$K"
      fi
    done
    # pilot 恒等性验证:clean 重跑本块(默认关闭)
    if [ "$RUN_CLEAN" = "1" ]; then
      python tools/test.py $FUSION_CFG $FUSION_CKPT \
        --cfg-options test_dataloader.dataset.data_root=$CHUNK/ \
        test_evaluator.ann_file=$CHUNK/nuscenes_infos_val.pkl \
        test_evaluator.format_only=True test_evaluator.jsonfile_prefix=$PRED/clean/blob$i \
        || echo "T_CLEAN_FAIL blob$i"
    fi
    # 清理本 blob 数据:删除链数为 1 的文件(camhist 硬链保留)
    find $CHUNK/samples -type f -links 1 -delete
    rm -rf $CHUNK/sweeps; mkdir -p $CHUNK/sweeps
  fi
  rm -rf $STG
  df -h /root/autodl-tmp | tail -1
done
echo "TEMPORAL_PIPELINE_DONE $(date +%H:%M:%S)"
