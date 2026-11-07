#!/bin/bash
# 补漏 pass2:对未完成样本,按"精确文件名"跨 blob 收集到常驻 gather 目录,最后统一推理
exec > /root/autodl-tmp/pass2.log 2>&1
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bevfusion
cd /root/mmdetection3d

GATHER=/root/autodl-tmp/gather
CHUNK=/root/autodl-tmp/chunk
P2=/root/autodl-tmp/pass2_patterns.txt
PRED=/root/autodl-tmp/preds
SRC=/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
shopt -s nullglob

mkdir -p $GATHER/samples $GATHER/sweeps $CHUNK

# 1) 从全量 pkl 生成 leftover 样本的全部所需文件清单(精确路径)
source /root/miniconda3/etc/profile.d/conda.sh
python - <<'PYEOF'
import pickle, json, os
PRED = '/root/autodl-tmp/preds'
info = pickle.load(open('data/nuscenes_trainval/nuscenes_infos_val.pkl', 'rb'))
done = set(json.load(open(f'{PRED}/lidar/_done.json'))) if os.path.exists(f'{PRED}/lidar/_done.json') else set()
paths = []
n_left = 0
for d in info['data_list']:
    if d['token'] in done:
        continue
    n_left += 1
    paths.append('samples/LIDAR_TOP/' + d['lidar_points']['lidar_path'])
    for cam, im in d['images'].items():
        paths.append(f'samples/{cam}/' + im['img_path'])
    for sw in d.get('lidar_sweeps', []):
        paths.append('sweeps/LIDAR_TOP/' + os.path.basename(sw['lidar_points']['lidar_path']))
open('/root/autodl-tmp/pass2_patterns.txt', 'w').write('\n'.join(paths))
print('LEFTOVER_SAMPLES', n_left, 'FILE_PATTERNS', len(paths))
PYEOF
wc -l $P2

# 2) 扫全部 blob,收集 leftover 文件(gather 常驻,不删)
for i in 01 02 03 04 05 06 07 08 09 10; do
  echo "=== pass2 blob $i start $(date +%H:%M:%S) ==="
  tar -xzf $SRC/v1.0-trainval${i}_blobs.tgz -C $GATHER -T $P2
  echo "=== pass2 blob $i done $(date +%H:%M:%S) ==="
  df -h /root/autodl-tmp | tail -1
done

# 3) 组装 chunk:只保留文件齐全的 leftover 样本
python - <<'PYEOF'
import pickle, json, os
GATHER = '/root/autodl-tmp/gather'
CHUNK = '/root/autodl-tmp/chunk'
PRED = '/root/autodl-tmp/preds'
info = pickle.load(open('data/nuscenes_trainval/nuscenes_infos_val.pkl', 'rb'))
done = set(json.load(open(f'{PRED}/lidar/_done.json')))
def have(sub, name):
    return os.path.exists(os.path.join(GATHER, sub, name))
kept = []
for d in info['data_list']:
    if d['token'] in done:
        continue
    lp = d['lidar_points']['lidar_path']
    if not have('samples/LIDAR_TOP', lp):
        continue
    ok = all(have(f'samples/{cam}', im['img_path']) for cam, im in d['images'].items())
    if not ok:
        continue
    ok = all(have('sweeps/LIDAR_TOP', os.path.basename(sw['lidar_points']['lidar_path'])) for sw in d.get('lidar_sweeps', []))
    if ok:
        kept.append(d)
for i, d in enumerate(kept):
    d['sample_idx'] = i
out = dict(metainfo=info['metainfo'], data_list=kept)
pickle.dump(out, open(f'{CHUNK}/nuscenes_infos_val.pkl', 'wb'))
print('PASS2_SAMPLES', len(kept))
PYEOF

# 4) 推理并累计
N=$(python -c "import pickle; print(len(pickle.load(open('$CHUNK/nuscenes_infos_val.pkl','rb'))['data_list']))")
if [ "$N" -gt 0 ]; then
  rm -rf $CHUNK/samples $CHUNK/sweeps
  cp -r $GATHER/samples $CHUNK/samples
  cp -r $GATHER/sweeps $CHUNK/sweeps
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
    python tools/test.py $CFG $CKPT \
      --cfg-options test_dataloader.dataset.data_root=$CHUNK/ \
      test_evaluator.ann_file=$CHUNK/nuscenes_infos_val.pkl \
      test_evaluator.format_only=True test_evaluator.jsonfile_prefix=$PRED/$model/pass2 \
      && python /root/accumulate2.py $PRED $model \
      || echo "PASS2_${model^^}_FAIL"
  done
fi
echo "PASS2_DONE $(date +%H:%M:%S)"
