# -*- coding: utf-8 -*-
"""E3: per-class AP for k1..k5 on the common subset (fills the onset curve).
Also dumps GT boxes for the common subset (for local E4 FP/FN analysis).
Output: /root/autodl-tmp/breakdown_k15.json + /root/autodl-tmp/gt_common.json"""
import json, os
import numpy as np
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

EVALROOT = '/root/autodl-tmp/evalroot'
CLEAN = '/root/autodl-tmp/preds/fusion_cumulative.json'
LIDAR = '/root/autodl-tmp/preds/lidar_cumulative.json'
KROOT = '/root/autodl-tmp/preds_temporal/k%s_cumulative.json'
ALL_SETTINGS = [('fusion_clean', CLEAN)] + [('k%d' % k, KROOT % k) for k in range(1, 11)] + [('lidar_only', LIDAR)]
TARGET = ['k%d' % k for k in (1, 2, 3, 4, 5)]

cfg = config_factory('detection_cvpr_2019')
nusc = NuScenes(version='v1.0-trainval', dataroot=EVALROOT, verbose=False)

sets = [set(json.load(open(p))['results'].keys()) for _, p in ALL_SETTINGS]
common = set.intersection(*sets)
print('COMMON_SUBSET', len(common), flush=True)

# ---- dump GT boxes once (from a clean-run eval object) for local E4 ----
res = json.load(open(CLEAN))
for t in set(json.load(open(CLEAN))['results'].keys()):
    res['results'].setdefault(t, [])
padded = '/root/autodl-tmp/subeval/_tmp_padded.json'
os.makedirs('/root/autodl-tmp/subeval', exist_ok=True)
json.dump(res, open(padded, 'w'))
ev = DetectionEval(nusc, cfg, padded, 'val', output_dir='/root/autodl-tmp/subeval/_tmp', verbose=False)
for attr in ('gt_boxes',):
    b = getattr(ev, attr)
    b.boxes = {t: v for t, v in b.boxes.items() if t in common}
ev.sample_tokens = [t for t in ev.sample_tokens if t in common]

gt_dump = {}
for t in common:
    gt_dump[t] = [{'translation': list(g.translation), 'name': g.detection_name,
                   'size': list(g.size), 'rotation': list(g.rotation)} for g in ev.gt_boxes.boxes.get(t, [])]
json.dump(gt_dump, open('/root/autodl-tmp/gt_common.json', 'w'))
print('GT_DUMP_DONE', len(gt_dump), flush=True)

OUT = {'per_class': {}}

def run_eval(pred_path, tokens):
    res = json.load(open(pred_path))
    full_tokens = set(json.load(open(CLEAN))['results'].keys())
    for t in full_tokens:
        res['results'].setdefault(t, [])
    json.dump(res, open(padded, 'w'))
    e2 = DetectionEval(nusc, cfg, padded, 'val', output_dir='/root/autodl-tmp/subeval/_tmp', verbose=False)
    for attr in ('gt_boxes', 'pred_boxes'):
        b = getattr(e2, attr)
        b.boxes = {t: v for t, v in b.boxes.items() if t in tokens}
    e2.sample_tokens = [t for t in e2.sample_tokens if t in tokens]
    ms, _ = e2.evaluate()
    return ms.serialize()

paths = dict(ALL_SETTINGS)
for name in TARGET:
    ms = run_eval(paths[name], common)
    OUT['per_class'][name] = {'mean_ap': ms['mean_ap'], 'nd_score': ms['nd_score'], 'label_aps': ms['label_aps']}
    print('PER_CLASS', name, ms['mean_ap'], flush=True)

json.dump(OUT, open('/root/autodl-tmp/breakdown_k15.json', 'w'), indent=1)
print('ALL_DONE /root/autodl-tmp/breakdown_k15.json')
