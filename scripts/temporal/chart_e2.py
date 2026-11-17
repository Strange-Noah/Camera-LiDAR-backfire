# E2 decay chart: stale-LiDAR (E2) vs stale-camera (T1), with baselines.
# Regenerates results/temporal_misalignment/chart_e2_decay.svg (and local PNG).
import matplotlib.pyplot as plt
import numpy as np

# n=5856 common subset (E2 eval); T1 values n=5712 (T1 eval, k*100ms nominal)
delay_e2 = [50.1, 100.5, 150.6, 200.7, 251.1, 301.2, 351.4, 401.6, 451.7]
map_e2   = [0.6247, 0.5394, 0.4744, 0.4133, 0.3606, 0.3187, 0.2869, 0.2448, 0.1787]
delay_t1 = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
map_t1   = [0.6869, 0.6823, 0.6712, 0.6640, 0.6593, 0.6557, 0.6515, 0.6488, 0.6470, 0.6448, 0.6432]

fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.axhline(0.6855, color='tab:green', ls='--', lw=1.4, label='Fusion clean (0.6855)')
ax.axhline(0.6480, color='tab:gray', ls='--', lw=1.4, label='LiDAR-only (0.6480)')
ax.plot(delay_t1, map_t1, 'o-', color='tab:blue', lw=2, ms=5,
        label='T1: stale camera (k·100 ms)')
ax.plot(delay_e2, map_e2, 's-', color='tab:red', lw=2, ms=5,
        label='E2: stale LiDAR (k·50 ms)')
ax.axvspan(0, 50, color='tab:red', alpha=0.06)
ax.annotate('E2 already below\nLiDAR-only at ~50 ms',
            xy=(50, 0.6247), xytext=(150, 0.60),
            arrowprops=dict(arrowstyle='->', color='tab:red'), color='tab:red', fontsize=9)
ax.annotate('T1 crossover ~570 ms', xy=(570, 0.652), xytext=(620, 0.56),
            arrowprops=dict(arrowstyle='->', color='tab:blue'), color='tab:blue', fontsize=9)
ax.set_xlabel('Temporal misalignment delay (ms)')
ax.set_ylabel('mAP (nuScenes detection, paired subset)')
ax.set_title('E2: stale LiDAR destroys fusion far faster than stale camera (T1)')
ax.legend(loc='lower left')
ax.grid(alpha=0.3)
fig.savefig('chart_e2_decay.svg', bbox_inches='tight')
print('saved chart_e2_decay.svg')
