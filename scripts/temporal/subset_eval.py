# 同资格子集评测:clean fusion vs k1 vs k2 在完全相同的样本集合上算官方指标
# 用法: subset_eval.py <pred.json> [<pred.json> ...]   (每个 json 出一套子集指标)
# 共同子集 = 所有传入 json 的 results 键集合的交集(且属于 val 6019)
import json, os, sys
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

EVALROOT = '/root/autodl-tmp/evalroot'
CLEAN_FULL = '/root/autodl-tmp/preds/fusion_cumulative.json'  # 仅用于取 val 全量 token 做 padding
OUTROOT = '/root/autodl-tmp/subeval'

paths = sys.argv[1:]
sets = []
for p in paths:
    r = json.load(open(p))
    sets.append(set(r['results'].keys()))
common = set.intersection(*sets)
print('COMMON_SUBSET', len(common))

# val 全量 token(用 clean full 的键,应与 GT 完全一致)
full_tokens = set(json.load(open(CLEAN_FULL))['results'].keys())
cfg = config_factory('detection_cvpr_2019')
nusc = NuScenes(version='v1.0-trainval', dataroot=EVALROOT, verbose=False)

for p in paths:
    name = os.path.basename(p).replace('.json', '')
    res = json.load(open(p))
    # padding:补齐缺失 token 为空结果,满足 devkit 的样本集合断言
    for t in full_tokens:
        res['results'].setdefault(t, [])
    padded = f'{OUTROOT}/{name}_padded.json'
    os.makedirs(OUTROOT, exist_ok=True)
    json.dump(res, open(padded, 'w'))
    ev = DetectionEval(nusc, cfg, padded, 'val', output_dir=f'{OUTROOT}/{name}', verbose=False)
    # 子集过滤:GT 与预测都限制在共同子集
    for attr in ('gt_boxes', 'pred_boxes'):
        b = getattr(ev, attr)
        b.boxes = {t: v for t, v in b.boxes.items() if t in common}
    ev.sample_tokens = [t for t in ev.sample_tokens if t in common]
    metrics, _ = ev.evaluate()
    ms = metrics.serialize()
    tp = ms['tp_errors']
    print(f'[{name}] SUBSET_N={len(common)} mAP={ms["mean_ap"]:.4f} NDS={ms["nd_score"]:.4f} '
          f'mATE={tp["trans_err"]:.4f} mASE={tp["scale_err"]:.4f} mAOE={tp["orient_err"]:.4f} '
          f'mAVE={tp["vel_err"]:.4f} mAAE={tp["attr_err"]:.4f}')
