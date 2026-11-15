# E6 chart: joint camera+LiDAR misalignment grid (kc x kl) as a heatmap,
# plus marginal comparison vs T1 (camera-only) and E2 (LiDAR-only) curves.
# NOTE: fill GRID from run_e6_eval.sh output before running.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()
import matplotlib.pyplot as plt
import numpy as np

KCS = [2, 5, 8]
KLS = [1, 3, 6]
# Paired-subset mAP from e6_subset_eval.log (2026-11-04, common subset n=5719).
# GRID[i, j] = mAP for (kc=KCS[j], kl=KLS[i])
GRID = np.array([
    [0.6028, 0.5823, 0.5723],   # kl=1 (~50 ms lidar)
    [0.4410, 0.4222, 0.4102],   # kl=3 (~151 ms lidar)
    [0.2922, 0.2745, 0.2656],   # kl=6 (~301 ms lidar)
])

fig, ax = plt.subplots(figsize=(7.2, 5.4))
im = ax.imshow(GRID, cmap='RdYlGn', vmin=0.25, vmax=0.69)
ax.set_xticks(range(len(KCS)), [f'$k_c$={k}\n(~{k*100} ms cam)' for k in KCS])
ax.set_yticks(range(len(KLS)), [f'$k_l$={k}\n(~{k*50} ms lidar)' for k in KLS])
for i in range(len(KLS)):
    for j in range(len(KCS)):
        if not np.isnan(GRID[i, j]):
            color = 'white' if GRID[i, j] > 0.48 else 'black'
            ax.text(j, i, f'{GRID[i, j]:.4f}', ha='center', va='center', fontsize=11, color=color)
fig.colorbar(im, ax=ax, label='mAP (paired subset)')
ax.set_title('E6: joint camera+LiDAR temporal misalignment\n'
             f'(fusion clean 0.6870, LiDAR-only 0.6497, subset n=5719)')
fig.savefig('chart_e6_grid.png', bbox_inches='tight', dpi=150)
fig.savefig('chart_e6_grid.svg', bbox_inches='tight')
print('saved chart_e6_grid.png + .svg')
