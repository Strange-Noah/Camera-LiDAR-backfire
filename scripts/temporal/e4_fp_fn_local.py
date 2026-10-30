# -*- coding: utf-8 -*-
"""E4 (local, zero GPU): FP/FN error-type decomposition for temporal misalignment.

Question: is the backfire driven by more false positives (phantom/misaligned
detections) or more false negatives (missed objects)?

Method: greedy score-ordered matching per (sample_token, class) with a 2 m
ground-plane center distance threshold on the common 5712-sample subset.
- Predictions with score < SCORE_MIN are not matched; they are counted as
  fp_below (low-score noise, not actionable).
- FN = GT boxes with no matched prediction (score >= SCORE_MIN)
- FP = unmatched predictions with score >= SCORE_MIN, binned by score.

Settings: fusion_clean, k1..k10, lidar_only. Output: results/e4_fp_fn.json
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

# ---- cache GT xy arrays per (token, class) once ----
GT_XY = {}   # (token, class) -> (n,2) array
GT_N = {c: 0 for c in CLASSES}
for t in TOKENS:
    for g in GT[t]:
        c = g['name']
        if c in CLASSES:
            GT_XY.setdefault((t, c), []).append((g['translation'][0], g['translation'][1]))
for k, v in GT_XY.items():
    GT_XY[k] = np.array(v, dtype=np.float32)
    GT_N[k[1]] += len(v)

def match_setting(preds):
    out = {c: {'gt': GT_N[c], 'fn': 0, 'fp': 0, 'fp_low': 0, 'fp_mid': 0,
               'fp_high': 0, 'fp_below': 0} for c in CLASSES}
    for t in TOKENS:
        # group predictions by class once
        by_c = {}
        for p in preds.get(t, []):
            c = p['detection_name']
            if c in CLASSES:
                s = p['detection_score']
                if s < SCORE_MIN:
                    out[c]['fp_below'] += 1
                else:
                    by_c.setdefault(c, []).append(p)
        for c, pc in by_c.items():
            gxy = GT_XY.get((t, c))
            if gxy is None:
                out[c]['fp'] += len(pc)
                for p in pc:
                    s = p['detection_score']
                    if s < 0.2: out[c]['fp_low'] += 1
                    elif s < 0.5: out[c]['fp_mid'] += 1
                    else: out[c]['fp_high'] += 1
                continue
            pc.sort(key=lambda p: -p['detection_score'])
            matched = np.zeros(len(gxy), dtype=bool)
            fp_here = []
            px = np.array([p['translation'][0] for p in pc], dtype=np.float32)
            py = np.array([p['translation'][1] for p in pc], dtype=np.float32)
            ps = [p['detection_score'] for p in pc]
            for i in range(len(pc)):
                d = np.hypot(px[i] - gxy[:, 0], py[i] - gxy[:, 1])
                dm = np.where(matched, np.inf, d)
                j = int(np.argmin(dm))
                if dm[j] < 2.0:
                    matched[j] = True
                else:
                    fp_here.append(ps[i])
            out[c]['fn'] += int((~matched).sum())
            out[c]['fp'] += len(fp_here)
            for s in fp_here:
                if s < 0.2: out[c]['fp_low'] += 1
                elif s < 0.5: out[c]['fp_mid'] += 1
                else: out[c]['fp_high'] += 1
    return out

results = {}
for name, fname in SETTINGS:
    path = os.path.join(ROOT, fname)
    if fname.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            d = json.load(f)
    else:
        d = json.load(open(path))
    results[name] = match_setting(d['results'])
    fn_total = sum(results[name][c]['fn'] for c in CLASSES)
    fp_total = sum(results[name][c]['fp'] for c in CLASSES)
    print('%-12s FN=%5d FP=%5d' % (name, fn_total, fp_total), flush=True)

json.dump(results, open('results/e4_fp_fn.json', 'w'), indent=1)
print('DONE results/e4_fp_fn.json', flush=True)
