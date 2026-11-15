# -*- coding: utf-8 -*-
"""E4 FN-definition A/B check (verification of @heyz's catch, 2026-11-03).

The original e4_fp_fn_local.py only visited classes with >=1 prediction
>= SCORE_MIN in a sample, silently skipping GT boxes in samples where the
model predicted nothing of that class. That skip rate grows as the model
degrades, biasing the FN growth clean->k10 downward (+38.0% reported vs
+40.4% corrected). This script reproduces both definitions side by side.

Verified: fusion_clean FN 12,607 -> 14,892; k10 FN 17,400 -> 20,903;
lidar_only FN 15,380 -> 18,015; FP identical under both definitions.
Output: results/e4_fp_fn_fixed.json (corrected definition).
"""
import json, gzip, os
import numpy as np

SCORE_MIN = 0.1
ROOT = 'results/remote_pull'
GT = json.load(open(os.path.join(ROOT, 'gt_common.json')))
TOKENS = sorted(GT.keys())
print('GT tokens:', len(TOKENS), flush=True)

SETTINGS = [('fusion_clean', 'fusion_cumulative.json')] + \
           [('k%d' % k, 'k%d_cumulative.json.gz' % k) for k in range(1, 11)] + \
           [('lidar_only', 'lidar_cumulative.json')]

CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

GT_XY = {}
GT_N = {c: 0 for c in CLASSES}
TOK_GT_CLASSES = {}
for t in TOKENS:
    cs = set()
    for g in GT[t]:
        c = g['name']
        if c in CLASSES:
            GT_XY.setdefault((t, c), []).append((g['translation'][0], g['translation'][1]))
            cs.add(c)
    TOK_GT_CLASSES[t] = cs
for k, v in GT_XY.items():
    GT_XY[k] = np.array(v, dtype=np.float32)
    GT_N[k[1]] += len(v)

def match_setting(preds, count_all_gt):
    out = {c: {'gt': GT_N[c], 'fn': 0, 'fp': 0} for c in CLASSES}
    for t in TOKENS:
        by_c = {}
        for p in preds.get(t, []):
            c = p['detection_name']
            if c in CLASSES:
                s = p['detection_score']
                if s >= SCORE_MIN:
                    by_c.setdefault(c, []).append(p)
        if count_all_gt:
            classes_here = set(by_c) | TOK_GT_CLASSES[t]
        else:
            classes_here = set(by_c)  # original (buggy) behavior
        for c in classes_here:
            gxy = GT_XY.get((t, c))
            pc = by_c.get(c, [])
            if gxy is None:
                out[c]['fp'] += len(pc)
                continue
            if not pc:
                out[c]['fn'] += len(gxy)
                continue
            pc.sort(key=lambda p: -p['detection_score'])
            matched = np.zeros(len(gxy), dtype=bool)
            px = np.array([p['translation'][0] for p in pc], dtype=np.float32)
            py = np.array([p['translation'][1] for p in pc], dtype=np.float32)
            fp = 0
            for i in range(len(pc)):
                d = np.hypot(px[i] - gxy[:, 0], py[i] - gxy[:, 1])
                dm = np.where(matched, np.inf, d)
                j = int(np.argmin(dm))
                if dm[j] < 2.0:
                    matched[j] = True
                else:
                    fp += 1
            out[c]['fn'] += int((~matched).sum())
            out[c]['fp'] += fp
    return out

summary = {}
for name, fname in SETTINGS:
    path = os.path.join(ROOT, fname)
    if fname.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            d = json.load(f)
    else:
        d = json.load(open(path))
    old = match_setting(d['results'], count_all_gt=False)
    new = match_setting(d['results'], count_all_gt=True)
    fn_o = sum(old[c]['fn'] for c in CLASSES)
    fp_o = sum(old[c]['fp'] for c in CLASSES)
    fn_n = sum(new[c]['fn'] for c in CLASSES)
    fp_n = sum(new[c]['fp'] for c in CLASSES)
    summary[name] = {'fn_old': fn_o, 'fp_old': fp_o, 'fn_new': fn_n, 'fp_new': fp_n,
                      'new': new}
    print('%-12s FN old=%5d new=%5d | FP old=%5d new=%5d' % (name, fn_o, fn_n, fp_o, fp_n), flush=True)

json.dump({k: v['new'] for k, v in summary.items()},
          open('results/e4_fp_fn_fixed.json', 'w'), indent=1)
clean, k10 = summary['fusion_clean'], summary['k10']
g_old = 100.0 * (k10['fn_old'] - clean['fn_old']) / clean['fn_old']
g_new = 100.0 * (k10['fn_new'] - clean['fn_new']) / clean['fn_new']
print('FN growth clean->k10: old +%.1f%%  new +%.1f%%' % (g_old, g_new))
print('DONE results/e4_fp_fn_fixed.json')
