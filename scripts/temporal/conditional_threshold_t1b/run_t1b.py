"""T1B conditional analysis (continues the T1 conditional-threshold pipeline in ../conditional_threshold)

Usage (Colab, after the T1 pipeline has produced gt_features.csv.gz and labels/ in --out):
    !python run_t1b.py --root /content/drive/MyDrive/nuscenes
It reuses common.py and step2_match.py from ../conditional_threshold (keep both folders side by side).

Conditional analysis of T1B (ego-consistent delayed camera) and comparison with T1.

T1B delivers each historical camera frame with its own capture-time calibration (LiDAR current). It uses the same
kind of historical images as T1; T1 and T1B are compared at the same frame index k only where the mean delays are close (k <= 8).
Aggregate T1B stays above LiDAR-only, so a crossover usually does not exist; this step asks what T1B still fails at.

Inputs : <data>/t1b_k{2,5,8,10,12}_cumulative.json(.gz)   (files that are missing are skipped)
         <out>/gt_features.csv.gz, <out>/labels/*.npz   (steps 1 and 2)
Population: GT boxes of samples that appear in EVERY available T1B file (a sample missing from a file would
            otherwise count all of its boxes as misses).
Main metric: GT miss rate (every unmatched box), score >= 0.1.

  7a  class level                       B, D(T1B, kmax), gap(T1B, kmax), D(T1, kcmp), D(T1B, kcmp), recovered share
  7b  1-D screening (car, pedestrian, traffic_cone, ALL) over all conditions, incl. object displacement
  7c  2-D controls                      object speed x distance, object displacement x distance, ...
  7d  figures

Definitions (pct points):  B = miss(LiDAR-only) - miss(clean);  D_T1B(k) = miss(T1B, k) - miss(clean);
  gap(k) = D(k) - B;  recovered share at kcmp = 1 - D_T1B(kcmp) / D_T1(kcmp)   (share of the T1 decay removed by
  ego-consistent calibration; reported only when D_T1(kcmp) > 1 pp).  kmax = largest available T1B k,
  kcmp = largest k with similar delay in T1 and T1B (k8; k10 differs by about 120 ms, see KCMP_MAX).
Outputs: <out>/tables/t1b_*.csv, <out>/figures/t1b_*.png, <out>/t1b_key_numbers.json
"""
import gc
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'conditional_threshold'))
from common import (CLASSES, CONDITIONS, DELAY_MS, SETTINGS, SceneBootstrap, find_file, get_args, load_features,
                    load_json, load_miss, log, save_csv, support_status)
from step2_match import match_setting
from t1b_config import T1B_FILES, T1B_K, T1B_SETTINGS

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)
THR = 0.1
# T1 and T1B share the frame index k but not the delay: mean delay at k8 is 677 ms (T1) vs about 700 ms (T1B), at k10 it is
# about 958 ms (T1, reconstructed) vs 834 ms (T1B, measured), because the two substitution tables are built differently.
# The T1 vs T1B comparison (recovered share) therefore uses k8, the largest k with similar delay in both.
KCMP_MAX = 8
GROUPS = ['ALL', 'car', 'pedestrian', 'traffic_cone']
# 1-D screening conditions: (column prefix, display name)
SCREEN_COLS = ['distance_xy', 'relative_lateral_velocity', 'object_speed', 'ego_speed', 'num_lidar_pts', 'abs_ego_yaw_rate',
               'relative_radial_velocity', 'r500', 'object_disp_1s', 'visibility', 'night', 'intersection']
# object-only displacement over 1 s in object widths (|object velocity| * 1 s / width): the mismatch that remains
# once the ego motion is handled by the capture-time calibration
OBJ_DISP_EDGES = [-np.inf, 0.1, 1.0, 3.0, np.inf]
OBJ_DISP_LABELS = ['od<0.1', '0.1<=od<1', '1<=od<3', 'od>=3']
PAIRS = [('object_speed', 'distance_xy'), ('object_disp_1s', 'distance_xy'), ('visibility', 'distance_xy'),
         ('num_lidar_pts', 'object_speed'), ('object_disp_1s', 'r500')]


# --------------------------------------------------------------------------------------
# 7-0  hit labels for the T1B files
# --------------------------------------------------------------------------------------
def make_labels(a):
    gtf = pd.read_csv(os.path.join(a.out, 'gt_features.csv.gz'), usecols=['box_id', 'sample_token', 'cls', 'x', 'y'])
    gtf = gtf.sort_values('box_id').reset_index(drop=True)
    tokens = sorted(gtf.sample_token.unique())
    gt_idx = {k: (g.box_id.to_numpy(), g[['x', 'y']].to_numpy()) for k, g in gtf.groupby(['sample_token', 'cls'])}
    ego = pd.read_csv(os.path.join(a.out, 'ego_samples.csv'))
    ego_xy = {r.sample_token: (r.ego_x, r.ego_y) for r in ego.itertuples()}
    have = []
    for s in T1B_SETTINGS:
        npz = os.path.join(a.out, 'labels', f'{s}.npz')
        if os.path.exists(npz):
            have.append(s)
            continue
        path = find_file(a.data, T1B_FILES[s])
        if not path:
            log(f'  {s:9s} MISSING prediction file {T1B_FILES[s]}(.gz) in {a.data}, skipped')
            continue
        t0 = time.time()
        results = load_json(path)['results']
        keys = set(results)
        in_file = gtf.sample_token.isin(keys).to_numpy()
        hit, present, fp = match_setting(results, gt_idx, tokens, ego_xy, len(gtf))
        del results
        gc.collect()
        np.savez_compressed(npz, in_file=in_file, **{f'hit_{t}': hit[t] for t in hit}, **{f'present_{t}': present[t] for t in present})
        have.append(s)
        log(f'  {s:9s} {time.time() - t0:.0f}s  samples in file: {len(keys & set(tokens))}/{len(tokens)}  FN@0.1 (all boxes)={int((~hit[THR]).sum())}')
    return have


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------
def pct_ci(x):
    x = x[np.isfinite(x)]
    return tuple(np.percentile(x, [2.5, 97.5])) if len(x) >= 10 else (np.nan, np.nan)


def metrics(r, idx, kmax, kcmp):
    """r: miss rates (..., n_cols). Returns dict of arrays/scalars in pct points."""
    clean, lid = r[..., idx['clean']], r[..., idx['lidar']]
    B = (lid - clean) * 100
    D1B = (r[..., idx[f't1b_k{kmax}']] - clean) * 100
    out = dict(B=B, D_T1B_kmax=D1B, gap_T1B_kmax=D1B - B)
    if kcmp is not None:
        d1 = (r[..., idx[f't1_k{kcmp}']] - clean) * 100
        d1b = (r[..., idx[f't1b_k{kcmp}']] - clean) * 100
        out.update(D_T1_kcmp=d1, D_T1B_kcmp=d1b, gap_T1_kcmp=d1 - B, gap_T1B_kcmp=d1b - B,
                   recovered=np.where(d1 > 1.0, 1 - d1b / np.where(d1 > 1.0, d1, 1), np.nan))
    return out


def cell_stats(boot, mask, MM, idx, kmax, kcmp, ci=True):
    n = int(mask.sum())
    if n == 0:
        return dict(N_boxes=0)
    n_s, m_s = boot.sums(mask, MM)
    rate = m_s.sum(0) / n_s.sum()
    pt = metrics(rate, idx, kmax, kcmp)
    row = dict(N_boxes=n, miss_clean=float(rate[idx['clean']]), miss_lidar=float(rate[idx['lidar']]))
    row.update({k: float(v) for k, v in pt.items()})
    for k in T1B_K_AVAIL:
        row[f'D_T1B_k{k}'] = float((rate[idx[f't1b_k{k}']] - rate[idx['clean']]) * 100)
    if ci:
        bn, bm = boot.W @ n_s, boot.W @ m_s
        with np.errstate(invalid='ignore', divide='ignore'):
            br = bm / bn[:, None]
        bt = metrics(br, idx, kmax, kcmp)
        for k in ('B', 'D_T1B_kmax', 'gap_T1B_kmax', 'recovered'):
            if k in bt:
                row[k + '_lo'], row[k + '_hi'] = pct_ci(bt[k])
        row['frac_gap_pos'] = float(np.mean(bt['gap_T1B_kmax'][np.isfinite(bt['gap_T1B_kmax'])] > 0))
    return row


def flag(row):
    lo = row.get('gap_T1B_kmax_lo', np.nan)
    dlo = row.get('D_T1B_kmax_lo', np.nan)
    if np.isfinite(lo) and lo > 0:
        return 'RESIDUAL_BACKFIRE'
    if np.isfinite(dlo) and dlo > 0:
        return 'residual_degradation'
    return ''



# --------------------------------------------------------------------------------------
# 7e  which conditions still drive the residual, after controlling for the others
# --------------------------------------------------------------------------------------
ADJ_FACTORS = [('object motion', 'object_disp_1s'), ('object motion', 'object_speed'), ('object motion', 'relative_radial_velocity'),
               ('relative displacement', 'r500'), ('relative displacement', 'distance_xy'),
               ('visibility / evidence', 'visibility'), ('visibility / evidence', 'num_lidar_pts')]


def cluster_ols(X, y, clusters):
    """OLS with scene-clustered (CR1) standard errors. Returns beta, se."""
    n, p = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    e = y - X @ beta
    codes = pd.factorize(clusters)[0]
    G = codes.max() + 1
    U = np.zeros((G, p))
    np.add.at(U, codes, X * e[:, None])
    V = XtX_inv @ (U.T @ U) @ XtX_inv * (G / (G - 1)) * ((n - 1) / max(n - p, 1))
    return beta, np.sqrt(np.clip(np.diag(V), 0, None))


def adjusted_model(g, MM, idx, kmax, tdir, fdir):
    """Per-box outcome y = miss(T1B, kmax) - miss(LiDAR-only)  (its mean over a group is the gap; > 0 = fusion worse than
    LiDAR-only) regressed on all condition bins at once (linear probability model, reference = first bin of each condition,
    scene-clustered SE).  A coefficient is the change in the gap (pp) when a box is in that bin instead of the reference
    bin, with the other conditions held fixed.  The second outcome, yD = miss(T1B) - miss(clean), is the residual degradation."""
    out = []
    y_gap = MM[:, idx[f't1b_k{kmax}']] - MM[:, idx['lidar']]
    y_deg = MM[:, idx[f't1b_k{kmax}']] - MM[:, idx['clean']]
    for grp in ['ALL', 'car']:
        m = np.ones(len(g), bool) if grp == 'ALL' else g.cls.eq('car').to_numpy()
        gg = g[m]
        cols, meta = [np.ones(m.sum())], [('', 'intercept', '')]
        for fam, cond in ADJ_FACTORS:
            lv = levels(gg, cond)
            b = gg[cond + '_bin'].astype(object).where(gg[cond + '_bin'].notna(), 'missing')
            ref = lv[0]
            for l in list(lv[1:]) + (['missing'] if (b == 'missing').any() else []):
                col = (b == l).to_numpy().astype(float)
                if col.sum() >= 50:
                    cols.append(col)
                    meta.append((fam, cond, l))
        X = np.column_stack(cols)
        for outcome, y in (('gap_vs_lidar', y_gap), ('degradation_vs_clean', y_deg)):
            beta, se = cluster_ols(X, y[m] * 100, gg.scene_token.to_numpy())
            for (fam, cond, l), bb, ss, c in zip(meta, beta, se, X.T):
                out.append(dict(group=grp, outcome=outcome, family=fam, condition=cond, bin=l, N_boxes=int(c.sum()),
                                coef_pp=bb, se=ss, lo=bb - 1.96 * ss, hi=bb + 1.96 * ss))
    adj = pd.DataFrame(out)
    save_csv(adj, os.path.join(tdir, 't1b_adjusted_effects.csv'), index=False)
    # forest plot of the adjusted effect on the gap, with the unadjusted per-bin gap for comparison
    for grp in ['ALL', 'car']:
        a_ = adj[(adj.group == grp) & (adj.outcome == 'gap_vs_lidar') & (adj.condition != 'intercept')].reset_index(drop=True)
        if a_.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(a_))))
        yy = np.arange(len(a_))[::-1]
        sig = (a_.lo > 0) | (a_.hi < 0)
        ax.errorbar(a_.coef_pp, yy, xerr=[a_.coef_pp - a_.lo, a_.hi - a_.coef_pp], fmt='none', ecolor='0.6', lw=1)
        ax.scatter(a_.coef_pp[sig], yy[sig], color='C3', zorder=3, label='95% CI excludes 0')
        ax.scatter(a_.coef_pp[~sig], yy[~sig], color='0.4', zorder=3, label='CI includes 0')
        ax.axvline(0, color='k', lw=0.8)
        ax.set_yticks(yy, [f'{r.family} | {r.condition}: {r.bin}' for r in a_.itertuples()], fontsize=7)
        ax.set_xlabel(f'change in gap to LiDAR-only at k{kmax} (pp), other conditions held fixed')
        ax.set_title(f'{grp}: what still pushes fusion below LiDAR-only after ego compensation (vs first bin of each condition)')
        ax.legend(fontsize=8, loc='lower right')
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, f't1b_residual_factors_{grp}.png'), dpi=130)
        plt.close(fig)
    return adj

# --------------------------------------------------------------------------------------
def add_bins(g):
    g['object_disp_1s'] = g.object_speed * 1.0 / g.width
    g['object_disp_1s_bin'] = pd.cut(g.object_disp_1s, bins=OBJ_DISP_EDGES, labels=OBJ_DISP_LABELS, right=True)
    return g


def levels(g, col):
    c = g[col + '_bin']
    return list(c.cat.categories) if hasattr(c, 'cat') else [x for x in sorted(c.dropna().unique())]


def main(a):
    global T1B_K_AVAIL
    t0 = time.time()
    log('7-0. hit labels for the T1B files')
    have = make_labels(a)
    if not have:
        raise SystemExit(f'No T1B prediction file found in {a.data}. Put t1b_k*_cumulative.json.gz there.')
    T1B_K_AVAIL = [int(s.split('_k')[1]) for s in have]
    kmax = max(T1B_K_AVAIL)
    common_k = [k for k in T1B_K_AVAIL if k <= KCMP_MAX]   # T1 and T1B have similar delay only up to k8, see KCMP_MAX
    kcmp = max(common_k) if common_k else None
    log(f'T1B delays available (frame index k): {T1B_K_AVAIL}; kmax = {kmax}; k used for the T1 comparison = {kcmp}')

    gtf = add_bins(load_features(a.out))
    MISS_T1 = load_miss(a.out, THR, 'all')                       # columns: SETTINGS order
    zs = {s: np.load(os.path.join(a.out, 'labels', f'{s}.npz')) for s in have}
    pop = np.logical_and.reduce([zs[s]['in_file'] for s in have])
    n_pop_samples = gtf.loc[pop, 'sample_token'].nunique()
    log(f'\nPopulation = samples present in every T1B file: {n_pop_samples} of {gtf.sample_token.nunique()} samples, '
        f'{int(pop.sum()):,} GT boxes (the team reports about 5,623 for its paired subset)')

    names = ['clean', 'lidar'] + [f't1_k{k}' for k in range(1, 11)] + [f't1b_k{k}' for k in T1B_K_AVAIL]
    cols = [MISS_T1[:, SETTINGS.index('fusion_clean')], MISS_T1[:, SETTINGS.index('lidar_only')]]
    cols += [MISS_T1[:, SETTINGS.index(f'k{k}')] for k in range(1, 11)]
    cols += [(~zs[f't1b_k{k}']['hit_0.1']).astype(float) for k in T1B_K_AVAIL]
    MM_all = np.column_stack(cols)
    idx = {n: i for i, n in enumerate(names)}

    g = gtf[pop].reset_index(drop=True)
    MM = MM_all[pop]
    boot = SceneBootstrap(g.scene_token)
    tdir, fdir = os.path.join(a.out, 'tables'), os.path.join(a.out, 'figures')

    # ---- aggregate table on the population ----
    rate = MM.mean(0)
    agg = pd.DataFrame({'setting': names, 'miss_rate': rate, 'FN': MM.sum(0).astype(int)})
    agg['D_vs_clean_pp'] = (agg.miss_rate - rate[idx['clean']]) * 100
    save_csv(agg, os.path.join(tdir, 't1b_aggregate_miss.csv'), index=False)
    log('\nAggregate GT miss rate on the population (score >= 0.1):')
    log(agg[agg.setting.isin(['clean', 'lidar'] + [f't1_k{k}' for k in common_k] + [f't1b_k{k}' for k in T1B_K_AVAIL])].round(4).to_string(index=False))

    # ---- 7a class level ----
    rows = []
    for grp in ['ALL'] + CLASSES:
        m = np.ones(len(g), bool) if grp == 'ALL' else g.cls.eq(grp).to_numpy()
        nt, ns, top1, st = support_status(g[m])
        r = cell_stats(boot, m, MM, idx, kmax, kcmp, ci=(st != 'SPARSE'))
        rows.append(dict(group=grp, N_scene=ns, status=st, **r, flag=flag(r)))
    cls = pd.DataFrame(rows)
    save_csv(cls, os.path.join(tdir, 't1b_class_level.csv'), index=False)
    show = ['group', 'N_boxes', 'B', f'D_T1_kcmp', 'D_T1B_kcmp', 'recovered', 'D_T1B_kmax', 'gap_T1B_kmax', 'gap_T1B_kmax_lo', 'flag']
    log(f'\n7a. Class level (pp). kcmp={kcmp} (T1 vs T1B), kmax={kmax}:')
    log(cls[[c for c in show if c in cls.columns]].round(2).to_string(index=False))

    # ---- 7b 1-D screening ----
    rows = []
    for grp in GROUPS:
        base = np.ones(len(g), bool) if grp == 'ALL' else g.cls.eq(grp).to_numpy()
        for cond in SCREEN_COLS:
            for lev in levels(g, cond):
                m = base & g[cond + '_bin'].eq(lev).to_numpy()
                nt, ns, top1, st = support_status(g[m])
                r = cell_stats(boot, m, MM, idx, kmax, kcmp, ci=(st == 'SUPPORTED'))
                rows.append(dict(group=grp, condition=cond, bin=lev, N_scene=ns, status=st, **r, flag=flag(r)))
    scr = pd.DataFrame(rows)
    save_csv(scr, os.path.join(tdir, 't1b_screening_1d.csv'), index=False)
    sup = scr[scr.status == 'SUPPORTED'].copy()
    log(f'\n7b. 1-D screening: {len(scr)} cells, {len(sup)} SUPPORTED. Largest residual degradation D(T1B, k{kmax}) (SUPPORTED, N>=300):')
    top = sup[sup.N_boxes >= 300].sort_values('D_T1B_kmax', ascending=False).head(15)
    log(top[['group', 'condition', 'bin', 'N_boxes', 'B', 'D_T1B_kmax', 'D_T1B_kmax_lo', 'D_T1B_kmax_hi', 'gap_T1B_kmax', 'recovered', 'flag']].round(2).to_string(index=False))
    rb = sup[sup.flag == 'RESIDUAL_BACKFIRE']
    log(f'\nCells where fusion is still worse than LiDAR-only at k{kmax} (95% CI of gap above 0): {len(rb)}')
    if len(rb):
        log(rb[['group', 'condition', 'bin', 'N_boxes', 'B', 'D_T1B_kmax', 'gap_T1B_kmax', 'gap_T1B_kmax_lo', 'gap_T1B_kmax_hi']].round(2).to_string(index=False))
    # which conditions explain most of the residual? spread of D across bins of each condition (ALL group)
    spread = []
    for cond in SCREEN_COLS:
        s = sup[(sup.group == 'ALL') & (sup.condition == cond)]
        if len(s) >= 2:
            spread.append(dict(condition=cond, n_bins=len(s), D_min=s.D_T1B_kmax.min(), D_max=s.D_T1B_kmax.max(),
                               spread=s.D_T1B_kmax.max() - s.D_T1B_kmax.min(),
                               worst_bin=s.loc[s.D_T1B_kmax.idxmax(), 'bin']))
    spread = pd.DataFrame(spread).sort_values('spread', ascending=False)
    save_csv(spread, os.path.join(tdir, 't1b_condition_spread_all.csv'), index=False)
    log(f'\nSpread of residual D(T1B, k{kmax}) across the bins of each condition (ALL classes, SUPPORTED bins): a larger spread means the condition separates the residual more')
    log(spread.round(2).to_string(index=False))

    # ---- 7c 2-D controls ----
    rows = []
    for grp in ['ALL', 'car']:
        base = np.ones(len(g), bool) if grp == 'ALL' else g.cls.eq(grp).to_numpy()
        for c1, c2 in PAIRS:
            for l1 in levels(g, c1):
                for l2 in levels(g, c2):
                    m = base & g[c1 + '_bin'].eq(l1).to_numpy() & g[c2 + '_bin'].eq(l2).to_numpy()
                    nt, ns, top1, st = support_status(g[m])
                    r = cell_stats(boot, m, MM, idx, kmax, kcmp, ci=(st == 'SUPPORTED'))
                    rows.append(dict(group=grp, var1=c1, bin1=l1, var2=c2, bin2=l2, N_scene=ns, status=st, **r, flag=flag(r)))
    two = pd.DataFrame(rows)
    save_csv(two, os.path.join(tdir, 't1b_controls_2d.csv'), index=False)
    log('\n7c. 2-D controls saved to tables/t1b_controls_2d.csv. Car, object speed x distance, D(T1B, kmax) / recovered share:')
    s = two[(two.group == 'car') & (two.var1 == 'object_speed') & (two.var2 == 'distance_xy')]
    log(s.pivot(index='bin1', columns='bin2', values='D_T1B_kmax').round(2).to_string())
    log(s.pivot(index='bin1', columns='bin2', values='recovered').round(2).to_string())


    # ---- 7e adjusted model: which conditions still drive the residual, other conditions held fixed ----
    adj = adjusted_model(g, MM, idx, kmax, tdir, fdir)
    for grp in ['ALL', 'car']:
        a_ = adj[(adj.group == grp) & (adj.outcome == 'gap_vs_lidar') & (adj.condition != 'intercept') & ((adj.lo > 0) | (adj.hi < 0))]
        log(f'\n7e. {grp}: bins whose adjusted effect on the gap at k{kmax} excludes 0 (pp, vs first bin of each condition):')
        log(a_.sort_values('coef_pp', ascending=False)[['family', 'condition', 'bin', 'N_boxes', 'coef_pp', 'lo', 'hi']].round(2).to_string(index=False)
            if len(a_) else '  none')

    log(f'\n7c. Does r500 matter once object displacement is fixed?  Car, gap to LiDAR-only at k{kmax} (pp), rows = object_disp_1s, columns = r500:')
    s2 = two[(two.group == 'car') & (two.var1 == 'object_disp_1s') & (two.var2 == 'r500')]
    log(s2.pivot(index='bin1', columns='bin2', values='gap_T1B_kmax').round(2).to_string())
    log('   N boxes:')
    log(s2.pivot(index='bin1', columns='bin2', values='N_boxes').to_string())

    # ---- 7d figures ----
    ks = list(range(1, 11))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ks, [rate[idx[f't1_k{k}']] for k in ks], 'o-', label='T1 (current-frame calibration)')
    ax.plot(T1B_K_AVAIL, [rate[idx[f't1b_k{k}']] for k in T1B_K_AVAIL], 's--', label='T1B (capture-time calibration)')
    ax.axhline(rate[idx['clean']], color='g', ls=':', label='fusion clean')
    ax.axhline(rate[idx['lidar']], color='r', ls='-.', label='LiDAR-only')
    ax.set_xlabel('frames back k (T1 and T1B delays are similar up to k8 only)')
    ax.set_ylabel('GT miss rate (score >= 0.1)')
    ax.set_title('All classes, common population')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, 't1b_aggregate.png'), dpi=130)
    plt.close(fig)

    car = g.cls.eq('car').to_numpy()
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.5), sharey=True)
    for ax, cond in zip(axs, ['object_speed', 'distance_xy', 'object_disp_1s']):
        for lev in levels(g, cond):
            m = car & g[cond + '_bin'].eq(lev).to_numpy()
            if m.sum() < 200:
                continue
            r = MM[m].mean(0)
            d1 = [(r[idx[f't1_k{k}']] - r[idx['clean']]) * 100 for k in ks]
            d1b = [(r[idx[f't1b_k{k}']] - r[idx['clean']]) * 100 for k in T1B_K_AVAIL]
            line, = ax.plot(ks, d1, '-', lw=1, label=f'{lev} T1 (n={int(m.sum())})')
            ax.plot(T1B_K_AVAIL, d1b, 's--', color=line.get_color(), ms=4)
            ax.axhline((r[idx['lidar']] - r[idx['clean']]) * 100, color=line.get_color(), ls=':', lw=0.8)
        ax.set_title(f'car by {cond}')
        ax.set_xlabel('frames back k')
        ax.legend(fontsize=7)
    axs[0].set_ylabel('degradation D (pp): T1 solid, T1B dashed; dotted = benefit B')
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, 't1b_car_degradation.png'), dpi=130)
    plt.close(fig)

    s = sup[(sup.group == 'ALL') & (sup.N_boxes >= 300) & sup.D_T1_kcmp.notna()].copy() if kcmp else pd.DataFrame()
    if len(s):
        s['label'] = s.condition + ': ' + s.bin.astype(str)
        s = s.sort_values('D_T1_kcmp')
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(s))))
        y = np.arange(len(s))
        ax.barh(y + 0.2, s.D_T1_kcmp, 0.4, label=f'T1, k{kcmp}')
        ax.barh(y - 0.2, s.D_T1B_kcmp, 0.4, label=f'T1B, k{kcmp}')
        ax.set_yticks(y, s.label, fontsize=7)
        ax.set_xlabel('degradation D (pp)')
        ax.set_title('All classes: degradation before and after ego-consistent calibration')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, 't1b_recovery_by_condition.png'), dpi=130)
        plt.close(fig)

    key = dict(n_samples=int(n_pop_samples), n_boxes=int(pop.sum()), t1b_k=T1B_K_AVAIL, kmax=kmax, kcmp=kcmp,
               miss_clean=float(rate[idx['clean']]), miss_lidar=float(rate[idx['lidar']]),
               **{f'miss_t1b_k{k}': float(rate[idx[f't1b_k{k}']]) for k in T1B_K_AVAIL},
               **{f'miss_t1_k{k}': float(rate[idx[f't1_k{k}']]) for k in common_k})
    with open(os.path.join(a.out, 't1b_key_numbers.json'), 'w') as f:
        json.dump(key, f, indent=2)
    log(f'\nT1B analysis done in {time.time() - t0:.0f}s')


T1B_K_AVAIL = []

if __name__ == '__main__':
    main(get_args('T1B conditional analysis and comparison with T1'))
