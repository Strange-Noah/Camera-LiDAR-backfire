#!/bin/bash
# 补漏 pass3:585 个样本的 sweep 文件在 tar 里被放到 samples/LIDAR_TOP/ 下,双位置匹配后归位再推理
exec > /root/autodl-tmp/pass3.log 2>&1
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bevfusion
cd /root/mmdetection3d

GATHER=/root/autodl-tmp/gather
G2=/root/autodl-tmp/gather2
CHUNK=/root/autodl-tmp/chunk
P3=/root/autodl-tmp/pass3_patterns.txt
PRED=/root/autodl-tmp/preds
SRC=/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
shopt -s nullglob
mkdir -p $G2/samples $G2/sweeps/LIDAR_TOP $CHUNK

# 1) 找出仍缺 sweep 的样本,生成双位置模式清单
python - <<'PYEOF'
import pickle, json, os
G='/root/autodl-tmp/gather'
PRED='/root/autodl-tmp/preds'
info = pickle.load(open('data/nuscenes_trainval/nuscenes_infos_val.pkl','rb'))
done = set(json.load(open(f'{PRED}/lidar/_done.json')))
lines = []
n = 0
for d in info['data_list']:
    if d['token'] in done: continue
    miss = [os.path.basename(sw['lidar_points']['lidar_path'])
            for sw in d.get('lidar_sweeps', [])
            if not os.path.exists(f"{G}/sweeps/LIDAR_TOP/{os.path.basename(sw['lidar_points']['lidar_path'])}")]
    if not miss: continue
    n += 1
    for m in miss:
        lines.append('samples/LIDAR_TOP/' + m)
        lines.append('sweeps/LIDAR_TOP/' + m)
open('/root/autodl-tmp/pass3_patterns.txt','w').write('\n'.join(lines))
print('PASS3_SAMPLES', n, 'PATTERNS', len(lines))
PYEOF

# 2) 扫全部 blob 收集(G2 常驻)
for i in 01 02 03 04 05 06 07 08 09 10; do
  echo "=== pass3 blob $i start $(date +%H:%M:%S) ==="
  tar -xzf $SRC/v1.0-trainval${i}_blobs.tgz -C $G2 -T $P3
  echo "=== pass3 blob $i done $(date +%H:%M:%S) ==="
done
# 归位:samples/LIDAR_TOP 下的 sweep 文件挪到 sweeps/LIDAR_TOP
mv $G2/samples/LIDAR_TOP/* $G2/sweeps/LIDAR_TOP/ 2>/dev/null
ls $G2/sweeps/LIDAR_TOP | wc -l

# 3) 组装 chunk = gather + gather2(带齐全性校验)
python - <<'PYEOF'
import pickle, json, os
G='/root/autodl-tmp/gather'; G2='/root/autodl-tmp/gather2'; CHUNK='/root/autodl-tmp/chunk'
PRED='/root/autodl-tmp/preds'
info = pickle.load(open('data/nuscenes_trainval/nuscenes_infos_val.pkl','rb'))
done = set(json.load(open(f'{PRED}/lidar/_done.json')))
def have(sub, name):
    n = os.path.basename(name)
    return os.path.exists(os.path.join(G, sub, n)) or os.path.exists(os.path.join(G2, sub, n))
kept = []
for d in info['data_list']:
    if d['token'] in done: continue
    lp = d['lidar_points']['lidar_path']
    if not have('samples/LIDAR_TOP', lp): continue
    if not all(have(f'samples/{cam}', im['img_path']) for cam, im in d['images'].items()): continue
    if not all(have('sweeps/LIDAR_TOP', sw['lidar_points']['lidar_path']) for sw in d.get('lidar_sweeps', [])): continue
    kept.append(d)
for i, d in enumerate(kept):
    d['sample_idx'] = i
out = dict(metainfo=info['metainfo'], data_list=kept)
pickle.dump(out, open(f'{CHUNK}/nuscenes_infos_val.pkl', 'wb'))
print('PASS3_USABLE', len(kept))
PYEOF

# 4) 复制数据并推理
N=$(/root/miniconda3/envs/bevfusion/bin/python -c "import pickle; print(len(pickle.load(open('$CHUNK/nuscenes_infos_val.pkl','rb'))['data_list']))")
echo "PASS3_N $N"
if [ "$N" -gt 0 ]; then
  rm -rf $CHUNK/samples $CHUNK/sweeps
  mkdir -p $CHUNK/samples $CHUNK/sweeps
  cp -r $GATHER/samples/* $CHUNK/samples/
  cp -r $GATHER/sweeps/* $CHUNK/sweeps/ 2>/dev/null
  mkdir -p $CHUNK/sweeps/LIDAR_TOP
  cp -n $G2/sweeps/LIDAR_TOP/* $CHUNK/sweeps/LIDAR_TOP/
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
      test_evaluator.format_only=True test_evaluator.jsonfile_prefix=$PRED/$model/pass3 \
      || echo "PASS3_${model^^}_FAIL"
  done
fi
echo "PASS3_DONE $(date +%H:%M:%S)"
