"""Step 2: per-GT-box hit labels and false-positive counts for all 12 settings.

Matching rule (same as E4): predictions with score >= threshold, greedy in descending score order
within each (sample, class), matched to the nearest unmatched GT box if the ground-plane center
distance is < 2 m. Computed for score thresholds 0.1 (main), 0.3 and 0.5 in a single pass per file.

Output per setting: <out>/labels/<setting>.npz with
    hit_<thr>     : GT box detected
    present_<thr> : the sample has >= 1 same-class prediction (reproduces the original E4 FN definition)
and <out>/labels/<setting>_fp.csv: predictions / false positives by predicted class x predicted distance.
Re-running skips settings that are already done.
"""
import gc
import os
import time

import numpy as np
import pandas as pd

from common import (CONDITIONS, EXPECTED, MATCH_DIST_M, PRED_FILES, SCORE_THRESHOLDS, SETTINGS, find_file, get_args,
                    load_json, log, save_csv)

DIST_EDGES = CONDITIONS['distance_xy'][0]
DIST_LABELS = CONDITIONS['distance_xy'][1]


def match_setting(results, gt_idx, tokens, ego_xy, n_boxes):
    hit = {t: np.zeros(n_boxes, bool) for t in SCORE_THRESHOLDS}
    present = {t: np.zeros(n_boxes, bool) for t in SCORE_THRESHOLDS}
    counts = {}                                           # (thr, cls, dist_label) -> [n_pred, n_fp]
    empty = (np.array([], dtype=int), np.zeros((0, 2)))
    for tok in tokens:
        ex, ey = ego_xy.get(tok, (np.nan, np.nan))
        by_cls = {}
        for p in results.get(tok, []):
            if p['detection_score'] >= SCORE_THRESHOLDS[0]:
                by_cls.setdefault(p['detection_name'], []).append(p)
        for cls, preds in by_cls.items():
            preds.sort(key=lambda p: -p['detection_score'])
            score = np.array([p['detection_score'] for p in preds])
            pxy = np.array([p['translation'][:2] for p in preds], dtype=float)
            dist_bin = np.digitize(np.hypot(pxy[:, 0] - ex, pxy[:, 1] - ey), DIST_EDGES[1:-1])
            ids, gxy = gt_idx.get((tok, cls), empty)
            for thr in SCORE_THRESHOLDS:
                sel = np.flatnonzero(score >= thr)
                if len(sel) and len(ids):
                    present[thr][ids] = True
                matched = np.zeros(len(ids), bool)
                is_fp = np.ones(len(sel), bool)
                if len(ids):
                    for n, i in enumerate(sel):
                        d = np.hypot(pxy[i, 0] - gxy[:, 0], pxy[i, 1] - gxy[:, 1])
                        d[matched] = np.inf
                        j = int(np.argmin(d))
                        if d[j] < MATCH_DIST_M:
                            matched[j] = True
                            is_fp[n] = False
                    hit[thr][ids[matched]] = True
                for b, lab in enumerate(DIST_LABELS):
                    m = dist_bin[sel] == b
                    c = counts.setdefault((thr, cls, lab), [0, 0])
                    c[0] += int(m.sum())
                    c[1] += int((is_fp & m).sum())
    fp = pd.DataFrame([dict(threshold=k[0], cls=k[1], pred_distance=k[2], n_pred=v[0], n_fp=v[1]) for k, v in counts.items()])
    return hit, present, fp


def main(a):
    gtf = pd.read_csv(os.path.join(a.out, 'gt_features.csv.gz'), usecols=['box_id', 'sample_token', 'cls', 'x', 'y'])
    gtf = gtf.sort_values('box_id').reset_index(drop=True)
    tokens = sorted(gtf.sample_token.unique())
    gt_idx = {k: (g.box_id.to_numpy(), g[['x', 'y']].to_numpy()) for k, g in gtf.groupby(['sample_token', 'cls'])}
    ego = pd.read_csv(os.path.join(a.out, 'ego_samples.csv'))
    ego_xy = {r.sample_token: (r.ego_x, r.ego_y) for r in ego.itertuples()}

    for s in SETTINGS:
        npz, fpcsv = os.path.join(a.out, 'labels', f'{s}.npz'), os.path.join(a.out, 'labels', f'{s}_fp.csv')
        if os.path.exists(npz) and os.path.exists(fpcsv):
            log(f'  {s:13s} already done, skipped')
            continue
        path = find_file(a.data, PRED_FILES[s])
        if not path:
            log(f'  {s:13s} MISSING prediction file {PRED_FILES[s]}(.gz)')
            continue
        t0 = time.time()
        results = load_json(path)['results']
        hit, present, fp = match_setting(results, gt_idx, tokens, ego_xy, len(gtf))
        del results
        gc.collect()
        np.savez_compressed(npz, **{f'hit_{t}': hit[t] for t in SCORE_THRESHOLDS}, **{f'present_{t}': present[t] for t in SCORE_THRESHOLDS})
        fp['setting'] = s
        save_csv(fp, fpcsv, index=False)
        log(f'  {s:13s} {time.time() - t0:.0f}s  ' + '  '.join(f'FN@{t}={int((~hit[t]).sum())}' for t in SCORE_THRESHOLDS))

    log('\nReconciliation at score >= 0.1')
    for s in SETTINGS:
        p = os.path.join(a.out, 'labels', f'{s}.npz')
        if not os.path.exists(p):
            continue
        z = np.load(p)
        fn_all, fn_old = int((~z['hit_0.1']).sum()), int((~z['hit_0.1'] & z['present_0.1']).sum())
        ref_all, ref_old = EXPECTED['fn_all_unmatched'].get(s), EXPECTED['fn_old_e4'].get(s)
        msg = f'  {s:13s} FN (all unmatched) {fn_all:6d}'
        if ref_all is not None:
            msg += f' [corrected E4 {ref_all}: {"OK" if fn_all == ref_all else "MISMATCH"}]'
        msg += f' | FN (old E4 definition) {fn_old:6d}'
        if ref_old is not None:
            msg += f' [original E4 {ref_old}: {"OK" if fn_old == ref_old else "MISMATCH"}]'
        fpf = os.path.join(a.out, 'labels', f'{s}_fp.csv')
        if os.path.exists(fpf):
            f = pd.read_csv(fpf)
            n_fp = int(f[f.threshold == 0.1].n_fp.sum())
            ref_fp = EXPECTED['fp'].get(s)
            msg += f' | FP {n_fp:6d}' + (f' [E4 {ref_fp}: {"OK" if n_fp == ref_fp else "MISMATCH"}]' if ref_fp is not None else '')
        log(msg)


if __name__ == '__main__':
    main(get_args('Step 2: per-box hit labels and FP counts'))
