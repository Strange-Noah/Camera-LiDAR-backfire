"""Step 6: separate the initial fusion benefit from the misalignment-induced degradation.

For a group of GT boxes (miss rates, main definition, score >= 0.1):
    B      = miss(LiDAR-only) - miss(fusion clean)     initial fusion benefit           (pct points, > 0 = camera helps)
    D(k)   = miss(fusion, k)  - miss(fusion clean)     degradation caused by the delay  (pct points)
    gap(k) = D(k) - B                                  so fusion backfires exactly when D(k) > B
    D(k10) equals the "benefit lost" of steps 3-4.

Analyses
  6a  distance x r500 grid (car, and all classes): B, D(k10), crossover, with scene-bootstrap 95% CIs
  6b  the same grid with r500_ego (r including ego rotation): rotation shifts an object's bearing by
      yaw_rate * dt, i.e. yaw_rate * dt * Z / W body-widths, which grows with distance Z. If the distance
      effect on D shrinks when r includes rotation, part of it is rotation-induced misalignment.
  6c  rotation check (car): D(k10) by distance within r500 bins, split by ego yaw rate (low vs high)
  6d  D(k) curves vs B for car by distance at fixed r (the mechanism of the rank reversal)
Outputs: <out>/tables/benefit_degradation_*.csv, <out>/figures/benefit_degradation_*.png
"""
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (CONDITIONS, DELAYS, R_BINS, SceneBootstrap, crossover, get_args, group_mask, load_features,
                    load_miss, log, save_csv, support_status)

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
DIST_COL = 'distance_E1'                                   # 0-10, 10-20, 20-30, 30-40, >40 m
DIST_BINS = CONDITIONS[DIST_COL][1]
R_EDGES = CONDITIONS['r500'][0]


def bd(boot, mask, MISS):
    """B, D(k10), D(k) curve and crossover for one group, with bootstrap CIs (pct points)."""
    n = int(mask.sum())
    out = dict(N_boxes=n)
    if n == 0:
        return out
    n_s, m_s = boot.sums(mask, MISS)
    rate = m_s.sum(0) / n_s.sum()                          # columns: clean, k1..k10, LiDAR-only
    B = (rate[-1] - rate[0]) * 100
    D = (rate[:-1] - rate[0]) * 100                        # D(clean)=0, D(k1)..D(k10)
    bn, bm = boot.W @ n_s, boot.W @ m_s
    with np.errstate(invalid='ignore', divide='ignore'):
        br = bm / bn[:, None]
    bB = (br[:, -1] - br[:, 0]) * 100
    bD = (br[:, -2] - br[:, 0]) * 100
    ok = np.isfinite(bB) & np.isfinite(bD)
    cx, kind = crossover((rate[:-1] - rate[-1]))
    out.update(B=B, B_lo=np.percentile(bB[ok], 2.5), B_hi=np.percentile(bB[ok], 97.5),
               D_k10=D[-1], D_lo=np.percentile(bD[ok], 2.5), D_hi=np.percentile(bD[ok], 97.5),
               D_minus_B_k10=D[-1] - B, crossover_ms=cx, kind=kind,
               **{f'D_{d:.0f}ms': v for d, v in zip(DELAYS[1:], D[1:])})
    return out


def grid(gtf, boot, MISS, base, r_col):
    rows = []
    for db in DIST_BINS:
        for rb in R_BINS:
            m = base & gtf[DIST_COL + '_bin'].eq(db).to_numpy() & gtf[r_col + '_bin'].eq(rb).to_numpy()
            nt, ns, top1, st = support_status(gtf[m])
            r = bd(boot, m, MISS) if st != 'SPARSE' else dict(N_boxes=int(m.sum()))
            rows.append({'distance': db, r_col: rb, 'N_scene': ns, 'status': st, **r})
    return pd.DataFrame(rows)


def pivot_print(df, r_col, value):
    t = df.pivot(index='distance', columns=r_col, values=value).reindex(index=DIST_BINS, columns=R_BINS)
    log(t.round(1).to_string())


def heatmaps(df, r_col, title, path):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (val, lab, cmap) in zip(axs, [('B', 'initial benefit B (pts)', 'Blues'),
                                          ('D_k10', 'degradation D(k10) (pts)', 'Reds'),
                                          ('D_minus_B_k10', 'gap at k10 = D - B (pts)', 'RdBu_r')]):
        t = df.pivot(index='distance', columns=r_col, values=val).reindex(index=DIST_BINS, columns=R_BINS)
        v = t.to_numpy(dtype=float)
        if val == 'D_minus_B_k10':
            lim = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 1
            im = ax.imshow(v, cmap=cmap, vmin=-lim, vmax=lim, aspect='auto')
        else:
            im = ax.imshow(v, cmap=cmap, aspect='auto')
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                if np.isfinite(v[i, j]):
                    ax.text(j, i, f'{v[i, j]:.1f}', ha='center', va='center', fontsize=8)
        ax.set_xticks(range(len(R_BINS)), R_BINS, fontsize=8)
        ax.set_yticks(range(len(DIST_BINS)), DIST_BINS, fontsize=8)
        ax.set_xlabel(r_col)
        ax.set_ylabel('distance')
        ax.set_title(lab, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(a):
    t0 = time.time()
    gtf = load_features(a.out)
    if 'r500_ego' not in gtf.columns:
        raise SystemExit('gt_features.csv.gz is from an older version: delete it and re-run step 1 (about 2 min).')
    gtf['r500_ego_bin'] = pd.cut(gtf.r500_ego, bins=R_EDGES, labels=R_BINS, right=True)
    boot = SceneBootstrap(gtf.scene_token)
    MISS = load_miss(a.out, 0.1, 'all')
    tdir, fdir = os.path.join(a.out, 'tables'), os.path.join(a.out, 'figures')

    # ---- 6a / 6b ----
    grids = {}
    for g in ['car', 'ALL']:
        for r_col in ['r500', 'r500_ego']:
            df = grid(gtf, boot, MISS, group_mask(gtf, g), r_col)
            grids[(g, r_col)] = df
            save_csv(df, os.path.join(tdir, f'benefit_degradation_{g}_distance_x_{r_col}.csv'), index=False)
            heatmaps(df, r_col, f'{g}: distance x {r_col} (score >= 0.1)',
                     os.path.join(fdir, f'benefit_degradation_{g}_{r_col}.png'))
            log(f'\n6{"a" if r_col == "r500" else "b"}. {g}: distance x {r_col}')
            log('  initial benefit B = miss(LiDAR-only) - miss(clean), pct points:')
            pivot_print(df, r_col, 'B')
            log('  degradation D(k10) = miss(k10) - miss(clean), pct points:')
            pivot_print(df, r_col, 'D_k10')
            log('  crossover (ms; fusion backfires when D(k) > B):')
            pivot_print(df, r_col, 'crossover_ms')

    # How much does distance still matter for D once r is fixed? (far minus near, per r bin)
    log('\nDistance effect on D(k10) within r bins: D(>40m) - D(0-10m), pct points')
    rows = []
    for g in ['car', 'ALL']:
        for r_col in ['r500', 'r500_ego']:
            t = grids[(g, r_col)].pivot(index='distance', columns=r_col, values='D_k10').reindex(index=DIST_BINS, columns=R_BINS)
            rows.append(dict(group=g, r_definition=r_col, **{rb: t.loc['>40m', rb] - t.loc['0-10m', rb] for rb in R_BINS}))
    eff = pd.DataFrame(rows)
    save_csv(eff, os.path.join(tdir, 'benefit_degradation_distance_effect.csv'), index=False)
    log(eff.round(1).to_string(index=False))
    log('If the far-minus-near difference shrinks with r500_ego, part of the distance effect is ego rotation.')

    # ---- 6c rotation check ----
    car = group_mask(gtf, 'car')
    rows = []
    for yb in ['yaw low', 'yaw high']:
        for rb in R_BINS:
            for db in DIST_BINS:
                m = car & gtf.abs_ego_yaw_rate_bin.eq(yb).to_numpy() & gtf.r500_bin.eq(rb).to_numpy() & gtf[DIST_COL + '_bin'].eq(db).to_numpy()
                nt, ns, top1, st = support_status(gtf[m])
                r = bd(boot, m, MISS) if st == 'SUPPORTED' else dict(N_boxes=int(m.sum()))
                rows.append(dict(yaw=yb, r500=rb, distance=db, status=st, **r))
    rot = pd.DataFrame(rows)
    save_csv(rot, os.path.join(tdir, 'benefit_degradation_car_yaw_x_r500_x_distance.csv'), index=False)
    log('\n6c. car: D(k10) by distance within r500 bins, low vs high ego yaw rate (SUPPORTED cells only)')
    for yb in ['yaw low', 'yaw high']:
        log(f'  {yb}:')
        t = rot[rot.yaw == yb].pivot(index='distance', columns='r500', values='D_k10').reindex(index=DIST_BINS, columns=R_BINS)
        log(t.round(1).to_string())
    log('If distance matters mainly through rotation, the far-vs-near difference should be much smaller at low yaw.')

    # ---- 6d curves ----
    df_all = []
    fig, axs = plt.subplots(1, len(R_BINS), figsize=(19, 4.5), sharey=True)
    for ax, rb in zip(axs, R_BINS):
        for db in DIST_BINS:
            m = car & gtf.r500_bin.eq(rb).to_numpy() & gtf[DIST_COL + '_bin'].eq(db).to_numpy()
            if m.sum() < 200:
                continue
            rate = MISS[m].mean(axis=0)
            D = (rate[:-1] - rate[0]) * 100
            B = (rate[-1] - rate[0]) * 100
            line, = ax.plot(DELAYS, D, marker='o', ms=3, label=f'{db} (n={int(m.sum())})')
            ax.axhline(B, color=line.get_color(), ls='--', lw=1)
            df_all.append(dict(r500=rb, distance=db, N=int(m.sum()), B=B, **{f'D_{d:.0f}ms': v for d, v in zip(DELAYS, D)}))
        ax.set_title(f'car, {rb}', fontsize=10)
        ax.set_xlabel('camera delay (ms)')
        ax.legend(fontsize=7)
    axs[0].set_ylabel('degradation D(k) (solid) vs initial benefit B (dashed), pct pts')
    fig.suptitle('Fusion backfires where the solid line D(k) rises above its dashed line B')
    fig.tight_layout()
    fig.savefig(os.path.join(fdir, 'benefit_degradation_car_curves.png'), dpi=130)
    plt.close(fig)
    save_csv(pd.DataFrame(df_all), os.path.join(tdir, 'benefit_degradation_car_curves.csv'), index=False)
    log(f'\nstep 6 done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main(get_args('Step 6: initial benefit vs misalignment-induced degradation'))
