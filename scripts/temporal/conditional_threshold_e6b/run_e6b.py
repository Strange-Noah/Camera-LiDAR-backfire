"""E6B conditional analysis (continues the T1 / T1B conditional-threshold pipeline in ../conditional_threshold)

Usage (Colab, after the T1 pipeline has produced gt_features.csv.gz and labels/ in --out):
    !python run_e6b.py --root /content/drive/MyDrive/nuscenes
It reuses common.py and step2_match.py from ../conditional_threshold (keep both folders side by side).

E6B = joint misalignment with BOTH branches ego-consistent: the camera frame is the k_c-th past frame delivered with its
own capture-time calibration (as T1B), and the LiDAR is the k_l-th past sweep with ego-motion compensation (as E5).
Grid: camera k_c in {2, 5, 8} (about 167 / 420 / 672 ms) x LiDAR k_l in {1, 3, 6} (about 50 / 151 / 301 ms).
Cell names: e6b_c{kc}_l{kl}.  Same GT, same matching and the same miss-rate metric as the T1 and T1B analyses.

Questions
  1. Which GT boxes still lose fusion value after both branches are compensated?
  2. Is the loss driven by the LiDAR delay or by the camera delay?  (increments below)
  3. Is it concentrated in moving objects, as T1B suggested for the camera?

Inputs : <data>/e6b_c{2,5,8}_l{1,3,6}_cumulative.json(.gz)   (missing cells are skipped)
         <out>/gt_features.csv.gz, <out>/labels/*.npz   (steps 1 and 2 of the T1 pipeline: fusion clean and LiDAR-only)
Population: GT boxes of samples that appear in EVERY available E6B file (a sample missing from a file would
            otherwise count all of its boxes as misses).
Metric  : GT miss rate (every unmatched box), score >= 0.1.  False positives are NOT analysed here: the team's E6B-FN
          note shows the l1 and l3 rows are almost pure FN damage, while the l6 row adds a large FP increase.

Definitions (pct points): B = miss(LiDAR-only) - miss(clean);  D(cell) = miss(cell) - miss(clean);
  gap(cell) = D(cell) - B  (gap > 0: fusion is worse than the LiDAR-only fallback).
  Probe cells:   LIDAR_PROBE = e6b_c2_l3  (LiDAR 151 ms, camera nearly fresh)    -> LiDAR-side effect
                 CAM_PROBE   = e6b_c8_l1  (camera 672 ms, LiDAR nearly fresh)    -> camera-side effect
  LiDAR increment = D(c2_l3) - D(c2_l1);   camera increment = D(c8_l1) - D(c2_l1).
  BACKFIRE flag: 95% scene-bootstrap lower bound of the gap above 0 (per probe cell).
Outputs: <out>/tables/e6b_*.csv, <out>/figures/e6b_*.png, <out>/e6b_key_numbers.json
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
from common import (CLASSES, SETTINGS, SceneBootstrap, find_file, get_args, load_features, load_json, load_miss, log,
                    save_csv, support_status)
from step2_match import match_setting

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 50)
THR = 0.1
KC = [2, 5, 8]
KL = [1, 3, 6]
CELLS = [f'e6b_c{c}_l{l}' for l in KL for c in KC]
FILES = {s: f'{s}_cumulative.json' for s in CELLS}
CAM_MS = {2: 167, 5: 420, 8: 672}       # measured mean CAM_FRONT delay (heyz T1B FN v2)
LID_MS = {1: 50, 3: 151, 6: 301}
LIDAR_PROBE, CAM_PROBE, BASE = 'e6b_c2_l3', 'e6b_c8_l1', 'e6b_c2_l1'
RELEASE_URL = 'https://github.com/Strange-Noah/Camera-LiDAR-backfire/releases/download/predictions-e6b'

GROUPS = ['ALL', 'car', 'pedestrian']
SCREEN_COLS = ['distance_xy', 'relative_lateral_velocity', 'object_speed', 'ego_speed', 'num_lidar_pts', 'abs_ego_yaw_rate',
               'relative_radial_velocity', 'r500', 'object_disp_1s', 'object_disp_lidar', 'visibility', 'night', 'intersection']
# object-only displacement in object widths = |object velocity| * horizon / width
#   object_disp_1s    : horizon 1 s     (comparable with the T1B analysis; camera side)
#   object_disp_lidar : horizon 0.151 s (the LiDAR delay of the l3 row)
OD1_EDGES, OD1_LABELS = [-np.inf, 0.1, 1.0, 3.0, np.inf], ['od<0.1', '0.1<=od<1', '1<=od<3', 'od>=3']
ODL_EDGES, ODL_LABELS = [-np.inf, 0.05, 0.25, 0.75, np.inf], ['odL<0.05', '0.05<=odL<0.25', '0.25<=odL<0.75', 'odL>=0.75']
PAIRS = [('object_speed', 'distance_xy'), ('object_disp_lidar', 'distance_xy'), ('visibility', 'distance_xy'),
         ('num_lidar_pts', 'object_speed'), ('object_disp_lidar', 'r500'), ('object_disp_lidar', 'visibility')]
CELL_AVAIL = []


# --------------------------------------------------------------------------------------
# hit labels for the E6B files
# --------------------------------------------------------------------------------------
def make_labels(a):
    gtf = pd.read_csv(os.path.join(a.out, 'gt_features.csv.gz'), usecols=['box_id', 'sample_token', 'cls', 'x', 'y'])
    gtf = gtf.sort_values('box_id').reset_index(drop=True)
    tokens = sorted(gtf.sample_token.unique())
    gt_idx = {k: (g.box_id.to_numpy(), g[['x', 'y']].to_numpy()) for k, g in gtf.groupby(['sample_token', 'cls'])}
    ego = pd.read_csv(os.path.join(a.out, 'ego_samples.csv'))
    ego_xy = {r.sample_token: (r.ego_x, r.ego_y) for r in ego.itertuples()}
    have = []
    for s in CELLS:
        npz = os.path.join(a.out, 'labels', f'{s}.npz')
        if os.path.exists(npz):
            have.append(s)
            continue
        path = find_file(a.data, FILES[s])
        if not path:
            log(f'  {s:10s} MISSING prediction file {FILES[s]}(.gz) in {a.data}, skipped')
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
        log(f'  {s:10s} {time.time() - t0:.0f}s  samples in file: {len(keys & set(tokens))}/{len(tokens)}  FN@0.1 (all boxes)={int((~hit[THR]).sum())}')
    return have


# --------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------
def pct_ci(x):
    x = x[np.isfinite(x)]
    return tuple(np.percentile(x, [2.5, 97.5])) if len(x) >= 10 else (np.nan, np.nan)


def metrics(r, idx):
    """r: miss rates (..., n_cols). Returns dict in pct points; probe-cell entries only if the cells exist."""
    clean, lid = r[..., idx['clean']], r[..., idx['lidar']]
    B = (lid - clean) * 100
    out = dict(B=B)
    for s in CELL_AVAIL:
        out['D_' + s] = (r[..., idx[s]] - clean) * 100
    for tag, cell in (('lidar', LIDAR_PROBE), ('cam', CAM_PROBE)):
        if cell in idx:
            out[f'gap_{tag}_probe'] = out['D_' + cell] - B
    if BASE in idx:
        if LIDAR_PROBE in idx:
            out['inc_lidar'] = out['D_' + LIDAR_PROBE] - out['D_' + BASE]
        if CAM_PROBE in idx:
            out['inc_cam'] = out['D_' + CAM_PROBE] - out['D_' + BASE]
    return out


CI_KEYS = ('B', 'gap_lidar_probe', 'gap_cam_probe', 'inc_lidar', 'inc_cam')


def cell_stats(boot, mask, MM, idx, ci=True):
    n = int(mask.sum())
    if n == 0:
        return dict(N_boxes=0)
    n_s, m_s = boot.sums(mask, MM)
    rate = m_s.sum(0) / n_s.sum()
    pt = metrics(rate, idx)
    row = dict(N_boxes=n, miss_clean=float(rate[idx['clean']]), miss_lidar=float(rate[idx['lidar']]))
    row.update({k: float(v) for k, v in pt.items()})
    if ci:
        bn, bm = boot.W @ n_s, boot.W @ m_s
        with np.errstate(invalid='ignore', divide='ignore'):
            br = bm / bn[:, None]
        bt = metrics(br, idx)
        for k in CI_KEYS:
            if k in bt:
                row[k + '_lo'], row[k + '_hi'] = pct_ci(bt[k])
    return row


def flags(row):
    out = []
    for tag, name in (('lidar', 'LiDAR'), ('cam', 'camera')):
        lo = row.get(f'gap_{tag}_probe_lo', np.nan)
        if np.isfinite(lo) and lo > 0:
            out.append(f'BACKFIRE_{name}_probe')
    return ' '.join(out)



# --------------------------------------------------------------------------------------
# 8e  which conditions drive the gap at the probe cells, other conditions held fixed
# --------------------------------------------------------------------------------------
ADJ_FACTORS = [('object motion', 'object_disp_lidar'), ('object motion', 'relative_radial_velocity'),
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


def adjusted_model(g, MM, idx, tdir, fdir):
    """Per-box outcome y = miss(probe cell) - miss(LiDAR-only), regressed on all condition bins at once (linear probability
    model, reference = first bin of each condition, scene-clustered SE).  A coefficient is the change in the gap (pp) when a
    box is in that bin instead of the reference bin, other conditions held fixed.  Conditions that drive B (distance, LiDAR
    points) change sign in this outcome; the degradation outcome miss(cell) - miss(clean) is also saved."""
    out = []
    for probe_name, probe in (('lidar_probe', LIDAR_PROBE), ('camera_probe', CAM_PROBE)):
        if probe not in idx:
            continue
        outcomes = (('gap_vs_lidar', MM[:, idx[probe]] - MM[:, idx['lidar']]),
                    ('degradation_vs_clean', MM[:, idx[probe]] - MM[:, idx['clean']]))
        for grp in ['ALL', 'car']:
            m = np.ones(len(g), bool) if grp == 'ALL' else g.cls.eq('car').to_numpy()
            gg = g[m]
            cols, meta = [np.ones(m.sum())], [('', 'intercept', '')]
            seen = set()
            for fam, cond in ADJ_FACTORS:
                lv = levels(gg, cond)
                b = gg[cond + '_bin'].astype(object).where(gg[cond + '_bin'].notna(), 'missing')
                for l in list(lv[1:]) + (['missing'] if (b == 'missing').any() else []):
                    col = (b == l).to_numpy().astype(float)
                    key = col.tobytes()
                    if col.sum() >= 50 and key not in seen:          # drop identical dummies (boxes without velocity)
                        seen.add(key)
                        cols.append(col)
                        meta.append((fam, cond, l))
            X = np.column_stack(cols)
            for outcome, y in outcomes:
                beta, se = cluster_ols(X, y[m] * 100, gg.scene_token.to_numpy())
                for (fam, cond, l), bb, ss, c in zip(meta, beta, se, X.T):
                    out.append(dict(probe=probe_name, group=grp, outcome=outcome, family=fam, condition=cond, bin=l,
                                    N_boxes=int(c.sum()), coef_pp=bb, se=ss, lo=bb - 1.96 * ss, hi=bb + 1.96 * ss))
    adj = pd.DataFrame(out)
    save_csv(adj, os.path.join(tdir, 'e6b_adjusted_effects.csv'), index=False)
    for probe_name in adj.probe.unique():
        for grp in ['ALL', 'car']:
            a_ = adj[(adj.probe == probe_name) & (adj.group == grp) & (adj.outcome == 'gap_vs_lidar') & (adj.condition != 'intercept')].reset_index(drop=True)
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
            ax.set_xlabel('change in gap to LiDAR-only (pp), other conditions held fixed')
            ax.set_title(f'{grp}, {probe_name}: what pushes fusion below LiDAR-only', fontsize=10)
            ax.legend(fontsize=8, loc='lower right')
            fig.tight_layout()
            fig.savefig(os.path.join(fdir, f'e6b_residual_factors_{probe_name}_{grp}.png'), dpi=130)
            plt.close(fig)
    return adj

# --------------------------------------------------------------------------------------
def add_bins(g):
    g['object_disp_1s'] = g.object_speed * 1.0 / g.width
    g['object_disp_1s_bin'] = pd.cut(g.object_disp_1s, bins=OD1_EDGES, labels=OD1_LABELS, right=True)
    g['object_disp_lidar'] = g.object_speed * (LID_MS[3] / 1000.0) / g.width
    g['object_disp_lidar_bin'] = pd.cut(g.object_disp_lidar, bins=ODL_EDGES, labels=ODL_LABELS, right=True)
    return g


def levels(g, col):
    c = g[col + '_bin']
    return list(c.cat.categories) if hasattr(c, 'cat') else [x for x in sorted(c.dropna().unique())]


def main(a):
    global CELL_AVAIL
    t0 = time.time()
    log('8-0. hit labels for the E6B files')
    have = make_labels(a)
    if not have:
        raise SystemExit(f'No E6B prediction file found in {a.data}. Put e6b_c*_l*_cumulative.json.gz there.')
    CELL_AVAIL = have
    log(f'E6B cells available: {have}')
    for p in (LIDAR_PROBE, CAM_PROBE, BASE):
        if p not in have:
            log(f'  note: probe/base cell {p} is missing, the related columns are skipped')

    gtf = add_bins(load_features(a.out))
    MISS_T1 = load_miss(a.out, THR, 'all')
    zs = {s: np.load(os.path.join(a.out, 'labels', f'{s}.npz')) for s in have}
    pop = np.logical_and.reduce([zs[s]['in_file'] for s in have])
    n_pop_samples = gtf.loc[pop, 'sample_token'].nunique()
    log(f'\nPopulation = samples present in every E6B file: {n_pop_samples} of {gtf.sample_token.nunique()} samples, '
        f'{int(pop.sum()):,} GT boxes (the team reports 5,719 for its E6B mAP subset and 5,712 for the FN note)')
    for s in have:                                   # how many samples does each file lack?
        miss_n = gtf.loc[~zs[s]['in_file'], 'sample_token'].nunique()
        if miss_n:
            log(f'  {s}: lacks {miss_n} samples of the T1 population')

    names = ['clean', 'lidar'] + have
    cols = [MISS_T1[:, SETTINGS.index('fusion_clean')], MISS_T1[:, SETTINGS.index('lidar_only')]]
    cols += [(~zs[s]['hit_0.1']).astype(float) for s in have]
    MM_all = np.column_stack(cols)
    idx = {n: i for i, n in enumerate(names)}

    g = gtf[pop].reset_index(drop=True)
    MM = MM_all[pop]
    boot = SceneBootstrap(g.scene_token)
    tdir, fdir = os.path.join(a.out, 'tables'), os.path.join(a.out, 'figures')

    # ---- aggregate ----
    rate = MM.mean(0)
    agg = pd.DataFrame({'setting': names, 'miss_rate': rate, 'FN': MM.sum(0).astype(int)})
    agg['D_vs_clean_pp'] = (agg.miss_rate - rate[idx['clean']]) * 100
    agg['gap_vs_lidar_pp'] = (agg.miss_rate - rate[idx['lidar']]) * 100
    save_csv(agg, os.path.join(tdir, 'e6b_aggregate_miss.csv'), index=False)
    log('\nAggregate GT miss rate on the population (score >= 0.1); gap > 0 means worse than LiDAR-only:')
    log(agg.round(4).to_string(index=False))

    # ---- 8a class level ----
    rows = []
    for grp in ['ALL'] + CLASSES:
        m = np.ones(len(g), bool) if grp == 'ALL' else g.cls.eq(grp).to_numpy()
        nt, ns, top1, st = support_status(g[m])
        r = cell_stats(boot, m, MM, idx, ci=(st != 'SPARSE'))
        rows.append(dict(group=grp, N_scene=ns, status=st, **r, flag=flags(r)))
    cls = pd.DataFrame(rows)
    save_csv(cls, os.path.join(tdir, 'e6b_class_level.csv'), index=False)
    show = ['group', 'N_boxes', 'B', 'D_' + BASE, 'D_' + CAM_PROBE, 'D_' + LIDAR_PROBE, 'gap_cam_probe', 'gap_lidar_probe',
            'gap_lidar_probe_lo', 'inc_lidar', 'inc_cam', 'flag']
    log('\n8a. Class level (pp). Probe gaps: cam = c8_l1, lidar = c2_l3.  inc_lidar = D(c2_l3)-D(c2_l1), inc_cam = D(c8_l1)-D(c2_l1):')
    log(cls[[c for c in show if c in cls.columns]].round(2).to_string(index=False))

    # ---- 8b 1-D screening ----
    rows = []
    for grp in GROUPS:
        base = np.ones(len(g), bool) if grp == 'ALL' else g.cls.eq(grp).to_numpy()
        for cond in SCREEN_COLS:
            for lev in levels(g, cond):
                m = base & g[cond + '_bin'].eq(lev).to_numpy()
                nt, ns, top1, st = support_status(g[m])
                r = cell_stats(boot, m, MM, idx, ci=(st == 'SUPPORTED'))
                rows.append(dict(group=grp, condition=cond, bin=lev, N_scene=ns, status=st, **r, flag=flags(r)))
    scr = pd.DataFrame(rows)
    save_csv(scr, os.path.join(tdir, 'e6b_screening_1d.csv'), index=False)
    sup = scr[scr.status == 'SUPPORTED'].copy()
    dcol = 'D_' + LIDAR_PROBE
    log(f'\n8b. 1-D screening: {len(scr)} cells, {len(sup)} SUPPORTED. Largest D at the LiDAR probe {LIDAR_PROBE} (SUPPORTED, N>=300):')
    top = sup[sup.N_boxes >= 300].sort_values(dcol, ascending=False).head(15)
    log(top[['group', 'condition', 'bin', 'N_boxes', 'B', dcol, 'gap_lidar_probe', 'gap_lidar_probe_lo', 'inc_lidar', 'inc_cam', 'flag']].round(2).to_string(index=False))
    rb = sup[sup.flag != '']
    log(f'\nCells where fusion is worse than LiDAR-only at a probe cell (95% CI of gap above 0): {len(rb)}')
    if len(rb):
        log(rb[['group', 'condition', 'bin', 'N_boxes', 'B', 'gap_lidar_probe', 'gap_lidar_probe_lo', 'gap_cam_probe', 'gap_cam_probe_lo', 'flag']].round(2).to_string(index=False))
    spread = []
    for cond in SCREEN_COLS:
        s = sup[(sup.group == 'ALL') & (sup.condition == cond)]
        if len(s) >= 2 and dcol in s:
            spread.append(dict(condition=cond, n_bins=len(s), spread_D_lidar_probe=s[dcol].max() - s[dcol].min(),
                               worst_bin_lidar=s.loc[s[dcol].idxmax(), 'bin'],
                               spread_inc_cam=(s['inc_cam'].max() - s['inc_cam'].min()) if 'inc_cam' in s else np.nan))
    spread = pd.DataFrame(spread).sort_values('spread_D_lidar_probe', ascending=False)
    save_csv(spread, os.path.join(tdir, 'e6b_condition_spread_all.csv'), index=False)
    log(f'\nSpread of D at {LIDAR_PROBE} across the bins of each condition (ALL, SUPPORTED bins); spread_inc_cam is the same for the camera increment:')
    log(spread.round(2).to_string(index=False))

    # ---- 8c 2-D controls ----
    rows = []
    for grp in ['ALL', 'car']:
        base = np.ones(len(g), bool) if grp == 'ALL' else g.cls.eq(grp).to_numpy()
        for c1, c2 in PAIRS:
            for l1 in levels(g, c1):
                for l2 in levels(g, c2):
                    m = base & g[c1 + '_bin'].eq(l1).to_numpy() & g[c2 + '_bin'].eq(l2).to_numpy()
                    nt, ns, top1, st = support_status(g[m])
                    r = cell_stats(boot, m, MM, idx, ci=(st == 'SUPPORTED'))
                    rows.append(dict(group=grp, var1=c1, bin1=l1, var2=c2, bin2=l2, N_scene=ns, status=st, **r, flag=flags(r)))
    two = pd.DataFrame(rows)
    save_csv(two, os.path.join(tdir, 'e6b_controls_2d.csv'), index=False)
    log(f'\n8c. 2-D controls saved to tables/e6b_controls_2d.csv. Car, LiDAR-displacement x distance, D at {LIDAR_PROBE}:')
    s = two[(two.group == 'car') & (two.var1 == 'object_disp_lidar') & (two.var2 == 'distance_xy')]
    if dcol in s:
        log(s.pivot(index='bin1', columns='bin2', values=dcol).round(2).to_string())
    for v2 in ('r500', 'visibility'):
        s2 = two[(two.group == 'car') & (two.var1 == 'object_disp_lidar') & (two.var2 == v2)]
        if 'gap_lidar_probe' in s2:
            log(f'\n8c. Does {v2} matter once object displacement is fixed?  Car, gap to LiDAR-only at {LIDAR_PROBE} (pp); rows = object_disp_lidar, columns = {v2}:')
            log(s2.pivot(index='bin1', columns='bin2', values='gap_lidar_probe').round(2).to_string())
            log('   N boxes:')
            log(s2.pivot(index='bin1', columns='bin2', values='N_boxes').to_string())

    # ---- 8e adjusted model ----
    adj = adjusted_model(g, MM, idx, tdir, fdir)
    for probe_name in adj.probe.unique():
        for grp in ['ALL', 'car']:
            a_ = adj[(adj.probe == probe_name) & (adj.group == grp) & (adj.outcome == 'gap_vs_lidar') & (adj.condition != 'intercept') & ((adj.lo > 0) | (adj.hi < 0))]
            log(f'\n8e. {grp}, {probe_name}: bins whose adjusted effect on the gap excludes 0 (pp, vs first bin of each condition):')
            log(a_.sort_values('coef_pp', ascending=False)[['family', 'condition', 'bin', 'N_boxes', 'coef_pp', 'lo', 'hi']].round(2).to_string(index=False)
                if len(a_) else '  none')

    # ---- 8f share of the aggregate gap by group (gaps are additive over boxes) ----
    rows = []
    for probe in (LIDAR_PROBE, CAM_PROBE):
        if probe not in idx:
            continue
        yb = MM[:, idx[probe]] - MM[:, idx['lidar']]
        tot = yb.mean() * 100
        for cond in ['object_disp_lidar', 'visibility', 'num_lidar_pts', 'distance_xy', 'object_speed']:
            for lev in levels(g, cond):
                m = g[cond + '_bin'].eq(lev).to_numpy()
                if m.sum():
                    rows.append(dict(probe=probe, condition=cond, bin=lev, N_boxes=int(m.sum()), share_of_boxes=m.mean(),
                                     gap_pp=yb[m].mean() * 100, contribution_pp=yb[m].sum() / len(yb) * 100, aggregate_gap_pp=tot))
    contrib = pd.DataFrame(rows)
    save_csv(contrib, os.path.join(tdir, 'e6b_gap_contribution.csv'), index=False)
    log('\n8f. Contribution of each bin to the aggregate gap (ALL classes): contribution = share of boxes x gap; bins of one condition add up to the aggregate gap')
    log(contrib[contrib.probe == LIDAR_PROBE].round(3).to_string(index=False))

    # ---- 8d figures ----
    # (1) 3x3 grid of the gap to LiDAR-only for ALL / car / pedestrian
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, grp in zip(axs, GROUPS):
        m = np.ones(len(g), bool) if grp == 'ALL' else g.cls.eq(grp).to_numpy()
        r = MM[m].mean(0)
        grid = np.full((len(KL), len(KC)), np.nan)
        for i, l in enumerate(KL):
            for j, c in enumerate(KC):
                s_ = f'e6b_c{c}_l{l}'
                if s_ in idx:
                    grid[i, j] = (r[idx[s_]] - r[idx['lidar']]) * 100
        lim = np.nanmax(np.abs(grid)) if np.isfinite(grid).any() else 1
        im = ax.imshow(grid, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
        for i in range(len(KL)):
            for j in range(len(KC)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f'{grid[i, j]:+.1f}', ha='center', va='center', fontsize=10)
        ax.set_xticks(range(len(KC)), [f'cam {CAM_MS[c]} ms' for c in KC])
        ax.set_yticks(range(len(KL)), [f'LiDAR {LID_MS[l]} ms' for l in KL])
        ax.set_title(f'{grp}: miss(fusion) - miss(LiDAR-only), pp')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, 'e6b_grid_gap.png'), dpi=130)
    plt.close(fig)

    # (2) car: D against LiDAR delay (camera c2), by object speed / distance / object displacement
    car = g.cls.eq('car').to_numpy()
    lcells = [f'e6b_c2_l{l}' for l in KL if f'e6b_c2_l{l}' in idx]
    if len(lcells) >= 2:
        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
        xs = [LID_MS[int(s.split('_l')[1])] for s in lcells]
        for ax, cond in zip(axs, ['object_speed', 'distance_xy', 'object_disp_lidar']):
            for lev in levels(g, cond):
                m = car & g[cond + '_bin'].eq(lev).to_numpy()
                if m.sum() < 200:
                    continue
                r = MM[m].mean(0)
                line, = ax.plot(xs, [(r[idx[s]] - r[idx['clean']]) * 100 for s in lcells], 'o-', label=f'{lev} (n={int(m.sum())})')
                ax.axhline((r[idx['lidar']] - r[idx['clean']]) * 100, color=line.get_color(), ls=':', lw=0.8)
            ax.set_title(f'car by {cond} (camera {CAM_MS[2]} ms)')
            ax.set_xlabel('LiDAR delay (ms)')
            ax.legend(fontsize=7)
        axs[0].set_ylabel('degradation D (pp); dotted = benefit B')
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, 'e6b_car_degradation.png'), dpi=130)
        plt.close(fig)

    # (3) gap by condition bin, both probe cells (ALL, SUPPORTED, N>=300)
    if 'gap_lidar_probe' in sup:
        s = sup[(sup.group == 'ALL') & (sup.N_boxes >= 300) & sup.gap_lidar_probe.notna()].copy()
    else:
        s = pd.DataFrame()
    if len(s):
        s['label'] = s.condition + ': ' + s.bin.astype(str)
        s = s.sort_values('gap_lidar_probe')
        fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(s))))
        y = np.arange(len(s))
        ax.barh(y + 0.2, s.gap_lidar_probe, 0.4, label=f'LiDAR probe {LIDAR_PROBE}')
        if 'gap_cam_probe' in s:
            ax.barh(y - 0.2, s.gap_cam_probe, 0.4, label=f'camera probe {CAM_PROBE}')
        ax.axvline(0, color='k', lw=0.8)
        ax.set_yticks(y, s.label, fontsize=7)
        ax.set_xlabel('gap to LiDAR-only (pp); > 0 means fusion is worse')
        ax.set_title('All classes: gap by condition')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fdir, 'e6b_gap_by_condition.png'), dpi=130)
        plt.close(fig)

    key = dict(n_samples=int(n_pop_samples), n_boxes=int(pop.sum()), cells=have,
               miss_clean=float(rate[idx['clean']]), miss_lidar=float(rate[idx['lidar']]),
               **{f'miss_{s_}': float(rate[idx[s_]]) for s_ in have})
    with open(os.path.join(a.out, 'e6b_key_numbers.json'), 'w') as f:
        json.dump(key, f, indent=2)
    log(f'\nE6B analysis done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(get_args('E6B conditional analysis (joint ego-consistent camera + LiDAR delay)'))
