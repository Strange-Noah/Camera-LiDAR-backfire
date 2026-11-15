"""Step 5: figures (main miss definition, score >= 0.1).

  figures/car_gap_panels.png      car gap curves by r500, lateral velocity, ego yaw rate, ego speed, object speed, distance
  figures/r500_gap_by_class.png   gap curves by r500 for traffic_cone and pedestrian
  figures/car_gap_by_distance.png car gap curves by distance
  figures/fp_vs_delay.png         total false positives vs camera delay, per score threshold
Gap = miss_rate(fusion, k) - miss_rate(LiDAR-only) in percentage points; > 0 means fusion is worse.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import DELAYS, DELAY_MS, FUSION_SETTINGS, SCORE_THRESHOLDS, SETTINGS, get_args, load_features, load_miss, log


def gap_panel(ax, gtf, MISS, cls_name, cond, min_n=50):
    base = gtf.cls.eq(cls_name).to_numpy()
    col = gtf[cond + '_bin']
    for b in col.cat.categories:
        m = base & col.eq(b).to_numpy()
        if m.sum() < min_n:
            continue
        rate = MISS[m].mean(axis=0)
        ax.plot(DELAYS, (rate[:-1] - rate[-1]) * 100, marker='o', ms=3, label=f'{b} (n={int(m.sum())})')
    ax.axhline(0, color='k', lw=1)
    ax.set_title(f'{cls_name} by {cond}', fontsize=10)
    ax.set_xlabel('camera delay (ms)')
    ax.set_ylabel('miss-rate gap vs LiDAR-only (pct pts)')
    ax.legend(fontsize=7)


def main(a):
    gtf = load_features(a.out)
    MISS = load_miss(a.out, 0.1, 'all')
    fig_dir = os.path.join(a.out, 'figures')

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    for ax, cond in zip(axs.ravel(), ['r500', 'relative_lateral_velocity', 'abs_ego_yaw_rate', 'ego_speed', 'object_speed', 'distance_xy']):
        gap_panel(ax, gtf, MISS, 'car', cond)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'car_gap_panels.png'), dpi=130)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    gap_panel(axs[0], gtf, MISS, 'traffic_cone', 'r500')
    gap_panel(axs[1], gtf, MISS, 'pedestrian', 'r500')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'r500_gap_by_class.png'), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    gap_panel(ax, gtf, MISS, 'car', 'distance_xy')
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'car_gap_by_distance.png'), dpi=130)
    plt.close(fig)

    fp = pd.concat([pd.read_csv(os.path.join(a.out, 'labels', f'{s}_fp.csv')) for s in SETTINGS], ignore_index=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for thr in SCORE_THRESHOLDS:
        t = fp[fp.threshold == thr].groupby('setting').n_fp.sum()
        line, = ax.plot(DELAYS, [t.get(s, np.nan) for s in FUSION_SETTINGS], marker='o', ms=3, label=f'fusion, score >= {thr}')
        ax.axhline(t.get('lidar_only', np.nan), color=line.get_color(), ls='--', lw=1, label=f'LiDAR-only, score >= {thr}')
    ax.set_xlabel('camera delay (ms)')
    ax.set_ylabel('false positives (count)')
    ax.set_title('False positives vs camera delay')
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fp_vs_delay.png'), dpi=130)
    plt.close(fig)
    log('figures saved to', fig_dir, ':', sorted(os.listdir(fig_dir)))


if __name__ == '__main__':
    main(get_args('Step 5: figures'))
