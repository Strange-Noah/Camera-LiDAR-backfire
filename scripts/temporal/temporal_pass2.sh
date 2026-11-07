#!/bin/bash
# temporal 补漏:主循环后,为"文件跨 blob/错位"而未完成的合格样本按精确文件名收集 → 推理
exec > /root/autodl-tmp/temporal_pass2.log 2>&1
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate bevfusion
cd /root/mmdetection3d

GT=/root/autodl-tmp/gather_t
CHUNK=/root/autodl-tmp/chunk
HIST=/root/autodl-tmp/camhist
PRED=/root/autodl-tmp/preds_temporal
SRC=/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval
shopt -s nullglob
mkdir -p $GT/samples $GT/sweeps/LIDAR_TOP $CHUNK

FUSION_CFG=projects/BEVFusion/configs/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d.py
FUSION_CKPT=checkpoints/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.pth

# 1) 生成补漏样本清单 + 精确文件名 patterns(双位置: sweeps/ 与 samples/ 都试)
python - <<'PYEOF'
import pickle, json, os
PRED='/root/autodl-tmp/preds_temporal'
info = pickle.load(open('data/nuscenes_trainval/nuscenes_infos_val.pkl','rb'))
table = json.load(open('/root/autodl-tmp/sub_table.json'))
done = set(json.load(open(f'{PRED}/_done.json')))
lines, keep = [], []
for d in info['data_list']:
    tok = d['token']
    if tok in done: continue
    e = table.get(tok)
    if e is None: continue
    if any(e[c].get('k1') is None or e[c].get('k2') is None for c in e): continue
    keep.append(tok)
    lines.append('samples/LIDAR_TOP/' + d['lidar_points']['lidar_path'])
    for cam, im in d['images'].items():
        lines.append(f'samples/{cam}/' + im['img_path'])
    for sw in d.get('lidar_sweeps', []):
        n = os.path.basename(sw['lidar_points']['lidar_path'])
        lines.append('sweeps/LIDAR_TOP/' + n)
        lines.append('samples/LIDAR_TOP/' + n)   # 585 错位双位置
open('/root/autodl-tmp/tp2_tokens.json','w').write(json.dumps(keep))
open('/root/autodl-tmp/tp2_patterns.txt','w').write('\n'.join(lines))
print('TP2_TOKENS', len(keep), 'PATTERNS', len(lines))
PYEOF

# 2) 扫全部 blob 收集
for i in 01 02 03 04 05 06 07 08 09 10; do
  echo "=== tp2 blob $i start $(date +%H:%M:%S) ==="
  tar -xzf $SRC/v1.0-trainval${i}_blobs.tgz -C $GT -T /root/autodl-tmp/tp2_patterns.txt
  echo "=== tp2 blob $i done $(date +%H:%M:%S) ==="
done
find $GT -type f | wc -l

# 3) 组 chunk + kpkl + 推理
cat > /root/tp2_build.py <<'PYEOF'
import pickle, json, os, sys
GT='/root/autodl-tmp/gather_t'; CHUNK='/root/autodl-tmp/chunk'; HIST='/root/autodl-tmp/camhist'
tokens = set(json.load(open('/root/autodl-tmp/tp2_tokens.json')))
info = pickle.load(open('data/nuscenes_trainval/nuscenes_infos_val.pkl','rb'))
def have(sub, name):
    return os.path.exists(os.path.join(GT, sub, name))
kept = []
for d in info['data_list']:
    if d['token'] not in tokens: continue
    if not have('samples/LIDAR_TOP', d['lidar_points']['lidar_path']): continue
    ok = True
    for cam, im in d['images'].items():
        if not have(f'samples/{cam}', im['img_path']): ok=False; break
    if not ok: continue
    for sw in d.get('lidar_sweeps', []):
        n = os.path.basename(sw['lidar_points']['lidar_path'])
        if not (have('sweeps/LIDAR_TOP', n) or have('samples/LIDAR_TOP', n)): ok=False; break
    if ok: kept.append(d)
for i, d in enumerate(kept):
    d['sample_idx'] = i
pickle.dump(dict(metainfo=info['metainfo'], data_list=kept), open(f'{CHUNK}/nuscenes_infos_val.pkl','wb'))
print('TP2_USABLE', len(kept))
PYEOF
python /root/tp2_build.py
N=$(python -c "import pickle; print(len(pickle.load(open('$CHUNK/nuscenes_infos_val.pkl','rb'))['data_list']))")
if [ "$N" -gt 0 ]; then
  python /root/tp2_move.py
  rm -rf data/nuscenes_trainval/sweeps
  ln -s $CHUNK/sweeps data/nuscenes_trainval/sweeps
  for K in 1 2; do
    python /root/make_kpkl.py $CHUNK $HIST /root/autodl-tmp/sub_table.json $K
    NK=$(python -c "import pickle; print(len(pickle.load(open('$CHUNK/nuscenes_infos_val_k$K.pkl','rb'))['data_list']))")
    echo "TP2 K$K USABLE $NK"
    if [ "$NK" -gt 0 ]; then
      python tools/test.py $FUSION_CFG $FUSION_CKPT \
        --cfg-options test_dataloader.dataset.data_root=$CHUNK/ \
        test_dataloader.dataset.ann_file=$CHUNK/nuscenes_infos_val_k$K.pkl \
        test_evaluator.ann_file=$CHUNK/nuscenes_infos_val_k$K.pkl \
        test_evaluator.format_only=True test_evaluator.jsonfile_prefix=$PRED/k$K/tp2 \
        && python /root/accum_t.py $PRED tp2 $K \
        || echo "TP2_FUSION_FAIL k$K"
    fi
  done
fi
echo "TP2_ALL_DONE $(date +%H:%M:%S)"
/root/miniconda3/envs/bevfusion/bin/python -c "import json; d=json.load(open('$PRED/_done.json')); print('DONE_TOTAL', len(d))"
