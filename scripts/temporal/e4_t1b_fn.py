# -*- coding: utf-8 -*-
"""E4-for-T1B: FN/FP decomposition of T1B (ego-consistent delayed camera) on the
E4 common subset (n=5712, gt_common.json). Identical matching protocol as
e4_fp_fn_fixed.py (greedy score-ordered, center distance < 2 m, score >= 0.1,
unconditional FN = every unmatched GT box). Incremental: per-setting results
are cached under results/e4_t1b_cache/ so re-runs skip completed settings.
"""
import json, gzip, os
import numpy as np

SCORE_MIN = 0.1
ROOT = 'results/remote_pull'
CACHE = 'results/e4_t1b_cache'
os.makedirs(CACHE, exist_ok=True)

GT = json.load(open(os.path.join(ROOT, 'gt_common.json')))
TOKENS = sorted(GT.keys())
print('GT tokens:', len(TOKENS), flush=True)

SETTINGS = [('fusion_clean', 'fusion_cumulative.json')] + \
           [('t1b_k%d' % k, 'preds_t1b/t1b_k%d_cumulative.json.gz' % k) for k in (2, 5, 8, 10, 12)] + \
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
        classes_here = set(by_c) | TOK_GT_CLASSES[t]  # corrected: unconditional FN
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
    cache_p = os.path.join(CACHE, name + '.json')
    if os.path.exists(cache_p):
        summary[name] = json.load(open(cache_p))
        print('%-12s CACHED FN=%5d FP=%5d' % (name, sum(summary[name][c]['fn'] for c in CLASSES),
                                              sum(summary[name][c]['fp'] for c in CLASSES)), flush=True)
        continue
    path = fname if os.path.exists(fname) else os.path.join(ROOT, fname)
    if fname.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            d = json.load(f)
    else:
        d = json.load(open(path))
    new = match_setting(d['results'])
    del d
    fn_n = sum(new[c]['fn'] for c in CLASSES)
    fp_n = sum(new[c]['fp'] for c in CLASSES)
    summary[name] = new
    json.dump(new, open(cache_p, 'w'))
    print('%-12s FN=%5d FP=%5d' % (name, fn_n, fp_n), flush=True)

json.dump(summary, open('results/e4_t1b_fn.json', 'w'), indent=1)

# ---- G_c table vs LiDAR-only (miss-rate percentage points) ----
lid = summary['lidar_only']
clean = summary['fusion_clean']
rows = []
for c in CLASSES:
    m_l = 100.0 * lid[c]['fn'] / lid[c]['gt']
    m_c = 100.0 * clean[c]['fn'] / clean[c]['gt']
    row = {'class': c, 'M_lidar': round(m_l, 1), 'M_clean': round(m_c, 1)}
    for name in ['t1b_k2', 't1b_k5', 't1b_k8', 't1b_k10', 't1b_k12']:
        s = summary[name]
        m_n = 100.0 * s[c]['fn'] / s[c]['gt']
        row['M_' + name] = round(m_n, 1)
        row['G_' + name] = round(m_n - m_l, 1)
        row['B_%s' % name] = round(m_l - m_c, 1)
    rows.append(row)
print('\n%-22s %6s %6s |' % ('class', 'Mclean', 'Mlidar') + ''.join(' %6s' % ('G_' + n) for n in ['t1b_k2', 't1b_k5', 't1b_k8', 't1b_k10', 't1b_k12']))
for r in rows:
    print('%-22s %6.1f %6.1f |' % (r['class'], r['M_clean'], r['M_lidar']) +
          ''.join(' %6.1f' % r['G_' + n] for n in ['t1b_k2', 't1b_k5', 't1b_k8', 't1b_k10', 't1b_k12']))
tot = {}
for name in ['fusion_clean'] + ['t1b_k%d' % k for k in (2, 5, 8, 10, 12)] + ['lidar_only']:
    s = summary[name]
    fn = sum(s[c]['fn'] for c in CLASSES)
    gt = sum(s[c]['gt'] for c in CLASSES)
    tot[name] = {'fn': fn, 'gt': gt, 'M_pct': round(100.0 * fn / gt, 2)}
print('\nOVERALL:', json.dumps(tot))
json.dump({'per_class': rows, 'overall': tot}, open('results/e4_t1b_fn_summary.json', 'w'), indent=1)
print('DONE results/e4_t1b_fn_summary.json')
