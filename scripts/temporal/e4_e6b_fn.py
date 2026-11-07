# -*- coding: utf-8 -*-
"""E4-for-E6B: FN/FP decomposition of the E6B joint-compensated 3x3 grid.
Same matching protocol as e4_t1b_fn.py (greedy score-ordered, center distance
< 2 m, score >= 0.1, unconditional FN = every unmatched GT box).

GT basis: intersection of gt_common.json tokens with tokens present in ALL 12
prediction files (fusion clean, LiDAR-only, 9 E6B cells) so every setting is
compared on identical samples. Output: results/e4_e6b_fn_summary.json.
"""
import json, gzip, os
import numpy as np

SCORE_MIN = 0.1
ROOT = 'results/remote_pull'
PRED_E6B = os.path.join(ROOT, 'preds_e6b')
CACHE = 'results/e4_e6b_cache'
os.makedirs(CACHE, exist_ok=True)

GT_ALL = json.load(open(os.path.join(ROOT, 'gt_common.json')))

CELLS = [(c, l) for c in (2, 5, 8) for l in (1, 3, 6)]
SETTINGS = [('fusion_clean', os.path.join(ROOT, 'fusion_cumulative.json'), False)] + \
           [('lidar_only', os.path.join(ROOT, 'lidar_cumulative.json'), False)] + \
           [('e6b_c%d_l%d' % cell, os.path.join(PRED_E6B, 'e6b_c%d_l%d_cumulative.json.gz' % cell), True)
            for cell in CELLS]

CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']


def load_preds(path, gz):
    if gz:
        with gzip.open(path, 'rt') as f:
            return json.load(f)['results']
    return json.load(open(path))['results']


# token coverage per setting
cov = {}
for name, path, gz in SETTINGS:
    d = load_preds(path, gz)
    cov[name] = set(d.keys())
    print('%-16s tokens=%d' % (name, len(d)), flush=True)

common = set(GT_ALL.keys())
for name, _, _ in SETTINGS:
    common &= cov[name]
TOKENS = sorted(common)
print('common tokens (GT & all 12 settings):', len(TOKENS), flush=True)

GT_XY = {}
GT_N = {c: 0 for c in CLASSES}
TOK_GT_CLASSES = {}
for t in TOKENS:
    cs = set()
    for g in GT_ALL[t]:
        c = g['name']
        if c in CLASSES:
            GT_XY.setdefault((t, c), []).append((g['translation'][0], g['translation'][1]))
            cs.add(c)
    TOK_GT_CLASSES[t] = cs
for k, v in GT_XY.items():
    GT_XY[k] = np.array(v, dtype=np.float32)
    GT_N[k[1]] += len(v)


def match_setting(preds):
    out = {c: {'gt': GT_N[c], 'fn': 0, 'fp': 0} for c in CLASSES}
    for t in TOKENS:
        by_c = {}
        for p in preds.get(t, []):
            c = p['detection_name']
            if c in CLASSES:
                s = p['detection_score']
                if s >= SCORE_MIN:
                    by_c.setdefault(c, []).append(p)
        classes_here = set(by_c) | TOK_GT_CLASSES[t]
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
for name, path, gz in SETTINGS:
    cache_p = os.path.join(CACHE, name + '.json')
    if os.path.exists(cache_p):
        summary[name] = json.load(open(cache_p))
        print('%-16s CACHED FN=%5d FP=%5d' % (name, sum(summary[name][c]['fn'] for c in CLASSES),
                                              sum(summary[name][c]['fp'] for c in CLASSES)), flush=True)
        continue
    preds = load_preds(path, gz)
    new = match_setting(preds)
    del preds
    json.dump(new, open(cache_p, 'w'))
    summary[name] = new
    print('%-16s FN=%5d FP=%5d' % (name, sum(new[c]['fn'] for c in CLASSES),
                                    sum(new[c]['fp'] for c in CLASSES)), flush=True)

json.dump(summary, open('results/e4_e6b_fn.json', 'w'), indent=1)

tot = {}
for name, _, _ in SETTINGS:
    s = summary[name]
    fn = sum(s[c]['fn'] for c in CLASSES)
    fp = sum(s[c]['fp'] for c in CLASSES)
    gt = sum(s[c]['gt'] for c in CLASSES)
    tot[name] = {'fn': fn, 'fp': fp, 'gt': gt, 'M_pct': round(100.0 * fn / gt, 2)}

print('\n=== overall miss rate / FP ===')
for name, _, _ in SETTINGS:
    print('%-16s miss%%=%5.2f  FN=%6d  FP=%6d' % (name, tot[name]['M_pct'], tot[name]['fn'], tot[name]['fp']))

json.dump({'n_tokens': len(TOKENS), 'overall': tot},
          open('results/e4_e6b_fn_summary.json', 'w'), indent=1)
print('\nDONE results/e4_e6b_fn_summary.json')
