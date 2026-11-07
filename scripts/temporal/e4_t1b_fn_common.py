# -*- coding: utf-8 -*-
"""Re-run of T1B FN decomposition restricted to the common population
(tokens present in gt_common AND in ALL of t1b k2/k5/k8/k10/k12 predictions).
Response to wangYilia's audit: k12 predictions miss 89 gt_common tokens
(1,795 GT boxes); counting them as FN inflated the k12 miss rate.
Same matching protocol (greedy, 2 m, score >= 0.1, unconditional FN).
"""
import json, gzip, os
import numpy as np

SCORE_MIN = 0.1
GT = json.load(open('results/remote_pull/gt_common.json'))
gt_tokens = set(GT.keys())

cov = {}
for k in (2, 5, 8, 10, 12):
    with gzip.open('preds_t1b/t1b_k%d_cumulative.json.gz' % k, 'rt') as f:
        cov[k] = set(json.load(f)['results'].keys())

TOKENS = sorted(gt_tokens & cov[2] & cov[5] & cov[8] & cov[10] & cov[12])
print('common tokens:', len(TOKENS))
GT_N_TOK = {t: GT[t] for t in TOKENS}

SETTINGS = [('fusion_clean', 'results/remote_pull/fusion_cumulative.json', False),
            ('lidar_only', 'results/remote_pull/lidar_cumulative.json', False)] + \
           [('t1b_k%d' % k, 'preds_t1b/t1b_k%d_cumulative.json.gz' % k, True) for k in (2, 5, 8, 10, 12)]

CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']


GT_XY = {}
GT_N = {c: 0 for c in CLASSES}
TOK_GT_CLASSES = {}
for t in TOKENS:
    cs = set()
    for g in GT_N_TOK[t]:
        c = g['name']
        if c in CLASSES:
            GT_XY.setdefault((t, c), []).append((g['translation'][0], g['translation'][1]))
            cs.add(c)
    TOK_GT_CLASSES[t] = cs
for k, v in GT_XY.items():
    GT_XY[k] = np.array(v, dtype=np.float32)
    GT_N[k[1]] += len(v)
GT_TOT = sum(GT_N.values())


def match_setting(preds):
    out = {c: {'gt': GT_N[c], 'fn': 0, 'fp': 0} for c in CLASSES}
    for t in TOKENS:
        by_c = {}
        for p in preds.get(t, []):
            c = p['detection_name']
            if c in CLASSES and p['detection_score'] >= SCORE_MIN:
                by_c.setdefault(c, []).append(p)
        for c in set(by_c) | TOK_GT_CLASSES[t]:
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
    if gz:
        with gzip.open(path, 'rt') as f:
            preds = json.load(f)['results']
    else:
        preds = json.load(open(path))['results']
    summary[name] = match_setting(preds)
    print('%-12s FN=%5d FP=%5d miss=%.2f%%' % (name,
          sum(summary[name][c]['fn'] for c in CLASSES),
          sum(summary[name][c]['fp'] for c in CLASSES),
          100.0 * sum(summary[name][c]['fn'] for c in CLASSES) / GT_TOT), flush=True)

lid = summary['lidar_only']
print('\nper-class miss%% and G_c vs lidar_only at k12:')
print('%-22s %6s %6s %8s' % ('class', 'M_lid', 'M_k12', 'G_k12'))
for c in CLASSES:
    m_l = 100.0 * lid[c]['fn'] / lid[c]['gt']
    m_k = 100.0 * summary['t1b_k12'][c]['fn'] / summary['t1b_k12'][c]['gt']
    print('%-22s %6.2f %6.2f %8.2f' % (c, m_l, m_k, m_k - m_l))

print('\noverall G_c per setting (miss%% - lidar_only):')
m_l = 100.0 * sum(lid[c]['fn'] for c in CLASSES) / GT_TOT
for name, _, _ in SETTINGS:
    m = 100.0 * sum(summary[name][c]['fn'] for c in CLASSES) / GT_TOT
    print('%-12s miss=%.2f%%  G=%+.2f pp' % (name, m, m - m_l))

json.dump({'n_tokens': len(TOKENS),
           'overall': {n: {'fn': sum(summary[n][c]['fn'] for c in CLASSES), 'gt': GT_TOT} for n, _, _ in SETTINGS}},
          open('results/e4_t1b_fn_common_summary.json', 'w'), indent=1)
print('\nDONE')
