"""Step 4: dose-response in r, control analyses and robustness checks.

  4a  r500 bins per group (car, pedestrian, traffic_cone, ALL): crossover and benefit lost with CIs
  4b  distance x r500 (car)                    -> is the r effect explained by distance?
  4c  r500 x ego speed (car)                   -> does ego speed matter once r is fixed?
  4d  r500 x object speed (car, pedestrian)    -> static vs moving objects at the same r
  4e  r trend test: share of scene-bootstrap replicates in which the r ordering holds
  4f  score-threshold sensitivity (0.1 / 0.3 / 0.5)
  4g  false positives by predicted class x predicted distance across delays
Main miss definition (every unmatched GT box) throughout. Outputs: <out>/tables/*.csv, <out>/key_numbers.json
"""
import json
import os
import time

import numpy as np
import pandas as pd

from common import (CLASSES, CONDITIONS, DELAY_MS, R_BINS, SCORE_THRESHOLDS, SETTINGS, SceneBootstrap, crossover,
                    empty_result, fmt_cross, get_args, group_mask, load_features, load_miss, log, save_csv,
                    support_status)

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
GROUPS = ['car', 'pedestrian', 'traffic_cone', 'ALL']


def cross_table(gtf, boot, MISS, base, row_col, col_col, require='SUPPORTED'):
    rows = []
    for rb in CONDITIONS[row_col][1] if row_col in CONDITIONS else gtf[row_col + '_bin'].cat.categories:
        for cb in CONDITIONS[col_col][1]:
            m = base & gtf[row_col + '_bin'].eq(rb).to_numpy() & gtf[col_col + '_bin'].eq(cb).to_numpy()
            nt, ns, top1, st = support_status(gtf[m])
            r = boot.analyze(m, MISS) if st == require else empty_result(m.sum(), 'skip')
            rows.append({row_col: rb, col_col: cb, 'N_scene': ns, 'status': st, **r})
    return pd.DataFrame(rows)


def show(df, cols):
    log(df[cols].round(1).to_string(index=False))


def main(a):
    t0 = time.time()
    gtf = load_features(a.out)
    boot = SceneBootstrap(gtf.scene_token)
    MISS = load_miss(a.out, 0.1, 'all')
    key = {}
    cols = ['N_boxes', 'status', 'crossover_ms', 'ci_lo', 'ci_hi', 'gap_clean', 'gap_k10', 'benefit_lost', 'bl_lo', 'bl_hi']

    # ---- 4a ----
    rows = []
    for g in GROUPS:
        for rb in R_BINS:
            m = group_mask(gtf, g) & gtf.r500_bin.eq(rb).to_numpy()
            nt, ns, top1, st = support_status(gtf[m])
            rows.append(dict(group=g, r_bin=rb, status=st, **boot.analyze(m, MISS)))
    rbin = pd.DataFrame(rows)
    save_csv(rbin, os.path.join(a.out, 'tables', 'r500_bins.csv'), index=False)
    log('4a. r500 bins: crossover (ms) and benefit lost (pct points), 95% CIs')
    show(rbin, ['group', 'r_bin'] + cols)
    key['r500_bins'] = {f'{r.group}|{r.r_bin}': dict(crossover_ms=r.crossover_ms, ci=[r.ci_lo, r.ci_hi],
                                                       benefit_lost=r.benefit_lost) for r in rbin.itertuples()}

    # ---- 4b / 4c / 4d ----
    car = group_mask(gtf, 'car')
    t_b = cross_table(gtf, boot, MISS, car, 'distance_xy', 'r500')
    save_csv(t_b, os.path.join(a.out, 'tables', 'car_distance_x_r500.csv'), index=False)
    log('\n4b. car: distance x r500 (SUPPORTED cells only)')
    show(t_b, ['distance_xy', 'r500', 'N_boxes', 'N_scene', 'status', 'crossover_ms', 'ci_lo', 'ci_hi', 'benefit_lost'])

    t_c = cross_table(gtf, boot, MISS, car, 'r500', 'ego_speed')
    save_csv(t_c, os.path.join(a.out, 'tables', 'car_r500_x_ego_speed.csv'), index=False)
    log('\n4c. car: r500 x ego speed')
    show(t_c, ['r500', 'ego_speed', 'N_boxes', 'N_scene', 'status', 'crossover_ms', 'ci_lo', 'ci_hi', 'benefit_lost'])

    parts = []
    for g in ['car', 'pedestrian']:
        t = cross_table(gtf, boot, MISS, group_mask(gtf, g), 'r500', 'object_speed')
        t.insert(0, 'group', g)
        parts.append(t)
    t_d = pd.concat(parts, ignore_index=True)
    save_csv(t_d, os.path.join(a.out, 'tables', 'r500_x_object_speed.csv'), index=False)
    log('\n4d. r500 x object speed (static vs moving at the same r)')
    show(t_d, ['group', 'r500', 'object_speed', 'N_boxes', 'status', 'crossover_ms', 'ci_lo', 'ci_hi', 'benefit_lost', 'bl_lo', 'bl_hi'])

    # ---- 4e ----
    rows = []
    for g in GROUPS:
        BL, CX = [], []
        for rb in R_BINS:
            n_s, m_s = boot.sums(group_mask(gtf, g) & gtf.r500_bin.eq(rb).to_numpy(), MISS)
            bgap = boot.boot_gaps(n_s, m_s)
            BL.append((bgap[:, -1] - bgap[:, 0]) * 100)
            CX.append([crossover(x)[0] if np.all(np.isfinite(x)) else np.nan for x in bgap])
        BL = np.array(BL)
        CX = np.where(np.isfinite(np.array(CX)), np.array(CX), np.inf)     # never crossing = infinitely late
        rows.append(dict(group=g,
                         loss_increases_across_all_4_bins=float(np.mean(np.all(np.diff(BL, axis=0) > 0, axis=0))),
                         loss_r_ge1_greater_than_r_lt01=float(np.mean(BL[3] > BL[0])),
                         crossover_earlier_across_all_4_bins=float(np.mean((CX[3] < CX[2]) & (CX[2] < CX[1]) & (CX[1] <= CX[0]))),
                         crossover_r_ge1_earlier_than_r_lt01=float(np.mean(CX[3] < CX[0]))))
    trend = pd.DataFrame(rows)
    save_csv(trend, os.path.join(a.out, 'tables', 'r_trend_bootstrap.csv'), index=False)
    log(f'\n4e. r trend test: share of {boot.W.shape[0]} scene-bootstrap replicates in which the ordering holds (>= 0.95 = robust)')
    log(trend.round(3).to_string(index=False))
    key['r_trend'] = trend.set_index('group').to_dict(orient='index')

    # ---- 4f ----
    rows = []
    for thr in SCORE_THRESHOLDS:
        M = load_miss(a.out, thr, 'all')
        tot = M.sum(0)
        log(f'\n4f. score >= {thr}: FN clean {int(tot[0])}, k10 {int(tot[-2])}, LiDAR-only {int(tot[-1])} '
            f'(clean -> k10 {100 * (tot[-2] / tot[0] - 1):+.1f}%)')
        for g in CLASSES + ['ALL']:
            rows.append(dict(threshold=thr, group=g, r_bin='(all)', **boot.analyze(group_mask(gtf, g), M)))
        for g in GROUPS:
            for rb in R_BINS:
                rows.append(dict(threshold=thr, group=g, r_bin=rb, **boot.analyze(group_mask(gtf, g) & gtf.r500_bin.eq(rb).to_numpy(), M)))
    sens = pd.DataFrame(rows)
    save_csv(sens, os.path.join(a.out, 'tables', 'threshold_sensitivity.csv'), index=False)
    sens['cell'] = sens.apply(fmt_cross, axis=1)
    log('\nClass-level crossover (ms) [95% CI] by score threshold:')
    log(sens[sens.r_bin == '(all)'].pivot(index='group', columns='threshold', values='cell').reindex(CLASSES + ['ALL']).to_string())
    sens['cell2'] = sens.apply(lambda r: f"{fmt_cross(r)} | loss {r['benefit_lost']:.1f}", axis=1)
    log('\nr500 bins by score threshold: crossover (ms) | benefit lost (pts):')
    log(sens[sens.r_bin != '(all)'].pivot_table(index=['group', 'r_bin'], columns='threshold', values='cell2',
                                                 aggfunc='first', sort=False).to_string())

    # ---- 4g ----
    fp = pd.concat([pd.read_csv(os.path.join(a.out, 'labels', f'{s}_fp.csv')) for s in SETTINGS
                    if os.path.exists(os.path.join(a.out, 'labels', f'{s}_fp.csv'))], ignore_index=True)
    save_csv(fp, os.path.join(a.out, 'tables', 'fp_counts_all_settings.csv'), index=False)
    log('\n4g. False positives')
    for thr in SCORE_THRESHOLDS:
        t = fp[fp.threshold == thr].groupby('setting').n_fp.sum()
        log(f'  score >= {thr}: FP clean {int(t.get("fusion_clean", 0))} -> k10 {int(t.get("k10", 0))} '
            f'({100 * (t.get("k10", 0) / max(t.get("fusion_clean", 1), 1) - 1):+.1f}%), LiDAR-only {int(t.get("lidar_only", 0))}')
    f = fp[fp.threshold == 0.1].pivot_table(index=['cls', 'pred_distance'], columns='setting', values='n_fp', aggfunc='sum')
    t = pd.DataFrame({'FP_clean': f.get('fusion_clean'), 'FP_k5': f.get('k5'), 'FP_k10': f.get('k10'), 'FP_lidar_only': f.get('lidar_only')})
    t['change_clean_to_k10_pct'] = 100 * (t.FP_k10 / t.FP_clean - 1)
    t['k10_minus_lidar_only'] = t.FP_k10 - t.FP_lidar_only
    t = t.reindex(pd.MultiIndex.from_product([CLASSES, CONDITIONS['distance_xy'][1]], names=['cls', 'pred_distance'])).dropna(how='all')
    save_csv(t, os.path.join(a.out, 'tables', 'fp_by_class_distance.csv'))
    log('\n  score >= 0.1, FP by predicted class x predicted distance:')
    log(t.round(1).to_string())
    up = t[(t.FP_k10 > t.FP_clean) & (t.FP_k10 >= 50)]
    log(f'\n  cells where FP increases from clean to k10 (and k10 >= 50): {len(up)}')
    if len(up):
        log(up.round(1).to_string())
    tot = fp[fp.threshold == 0.1].groupby('setting').n_fp.sum()
    key['fp_total_score_0.1'] = {s: int(tot.get(s, 0)) for s in SETTINGS}

    with open(os.path.join(a.out, 'key_numbers.json'), 'w') as fjson:
        json.dump(key, fjson, indent=1, default=float)
    log(f'\nstep 4 done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(get_args('Step 4: dose-response, controls and robustness checks'))
