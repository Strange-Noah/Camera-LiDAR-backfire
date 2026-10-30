#!/bin/bash
# 分块全量 val 评测 v2:逐 blob 扫包 → 按"文件实际存在"筛选 pkl 子集 → 两模型推理 → 累计预测 → 删数据
# 与日志如何切分到 blob 无关,每个样本恰好被预测一次,最后整体评测
exec > /root/autodl-tmp/chunk_pipeline.log 2>&1
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bevfusion
cd /root/mmdetection3d

STG=/root/autodl-tmp/staging
CHUNK=/root/autodl-tmp/chunk
P=/root/autodl-tmp/tar_patterns.txt
PRED=/root/autodl-tmp/preds
SRC=/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
shopt -s nullglob

mkdir -p $CHUNK/samples $CHUNK/sweeps $PRED/lidar $PRED/fusion
ln -sfn /root/autodl-tmp/nuscenes_meta/maps $CHUNK/maps
ln -sfn /root/autodl-tmp/nuscenes_meta/v1.0-trainval $CHUNK/v1.0-trainval

# tar 通配符清单(每个 val 日志前缀 × 7 个样本目录 + LiDAR sweeps)
: > $P
while read pre; do
  for d in CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT LIDAR_TOP; do
    echo "samples/$d/${pre}__*" >> $P
  done
  echo "sweeps/LIDAR_TOP/${pre}__*" >> $P
done < /root/autodl-tmp/val_log_prefixes.txt
wc -l $P

LIDAR_CFG=projects/BEVFusion/configs/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
LIDAR_CKPT=checkpoints/bevfusion_lidar_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-2628f933.pth
FUSION_CFG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
FUSION_CKPT=checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth

cat > /root/filter_chunk.py <<'PYEOF'
import pickle, os, sys, json
STG, CHUNK, PRED = sys.argv[1], sys.argv[2], sys.argv[3]
info = pickle.load(open('data/nuscenes_trainval/nuscenes_infos_val.pkl', 'rb'))
done = set(json.load(open(f'{PRED}/lidar/_done.json'))) if os.path.exists(f'{PRED}/lidar/_done.json') else set()
def avail(name):
    return os.path.exists(os.path.join(STG, 'samples', name.split('__')[-2] if '__' in name else 'LIDAR_TOP', name)) if False else None
def have(sub, name):
    return os.path.exists(os.path.join(STG, sub, name))
kept = []
for d in info['data_list']:
    if d['token'] in done:
        continue
    lp = d['lidar_points']['lidar_path']
    if not have('samples/LIDAR_TOP', lp):
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
# 重排 sample_idx(format 阶段用它索引本 chunk 的 data_infos)
for i, d in enumerate(kept):
    d['sample_idx'] = i
out = dict(metainfo=info['metainfo'], data_list=kept)
pickle.dump(out, open(f'{CHUNK}/nuscenes_infos_val.pkl', 'wb'))
print('CHUNK_SAMPLES', len(kept))
PYEOF

cat > /root/accumulate.py <<'PYEOF'
import json, os, sys
PRED, model, pre = sys.argv[1], sys.argv[2], sys.argv[3]
src = f'{PRED}/{model}/{pre}/pred_instances_3d'
cum_path = f'{PRED}/{model}_cumulative.json'
done_path = f'{PRED}/{model}/_done.json'
new = json.load(open(src))
cum = json.load(open(cum_path)) if os.path.exists(cum_path) else {'meta': new.get('meta', {}), 'results': {}}
done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
before = len(cum['results'])
cum['results'].update(new['results'])
done.update(new['results'].keys())
json.dump(cum, open(cum_path, 'w'))
json.dump(sorted(done), open(done_path, 'w'))
print('ACCUM', model, pre, 'NEW', len(new['results']), 'TOTAL', len(cum['results']), 'DUP', len(new['results']) - (len(cum['results']) - before))
PYEOF

for i in 01 02 03 04 05 06 07 08 09 10; do
  rm -rf $STG; mkdir -p $STG
  echo "=== blob $i extract start $(date +%H:%M:%S) ==="
  tar -xzf $SRC/v1.0-trainval${i}_blobs.tgz -C $STG --wildcards -T $P
  echo "=== blob $i extract done $(date +%H:%M:%S) ==="
  python /root/filter_chunk.py $STG $CHUNK $PRED
  N=$(python -c "import pickle; print(len(pickle.load(open('$CHUNK/nuscenes_infos_val.pkl','rb'))['data_list']))")
  echo "BLOB $i USABLE_SAMPLES $N"
  if [ "$N" -gt 0 ]; then
    rm -rf $CHUNK/samples $CHUNK/sweeps
    mv $STG/samples $CHUNK/samples
    mv $STG/sweeps $CHUNK/sweeps
    # pkl 中 sweep 路径为 './data/nuscenes_trainval/sweeps/...'(相对进程 CWD),软链到 chunk
    rm -rf data/nuscenes_trainval/sweeps
    ln -s $CHUNK/sweeps data/nuscenes_trainval/sweeps
    for model in lidar fusion; do
      if [ "$model" = "lidar" ]; then CFG=$LIDAR_CFG; CKPT=$LIDAR_CKPT; else CFG=$FUSION_CFG; CKPT=$FUSION_CKPT; fi
      pre="blob${i}"
      python tools/test.py $CFG $CKPT \
        --cfg-options test_dataloader.dataset.data_root=$CHUNK/ \
        test_evaluator.ann_file=$CHUNK/nuscenes_infos_val.pkl \
        test_evaluator.format_only=True test_evaluator.jsonfile_prefix=$PRED/$model/$pre \
        && python /root/accumulate.py $PRED $model $pre \
        || echo "${model^^}_FAIL blob$i"
    done
    rm -rf $CHUNK/samples $CHUNK/sweeps
    mkdir -p $CHUNK/samples $CHUNK/sweeps
  fi
  rm -rf $STG
  df -h /root/autodl-tmp | tail -1
done
echo "CHUNK_PIPELINE_DONE $(date +%H:%M:%S)"
