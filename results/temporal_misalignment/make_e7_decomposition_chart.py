import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
setup_plot()
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['svg.fonttype'] = 'none'

labels = ['kl = 3\n(~151 ms)', 'kl = 6\n(~301 ms)']
density = [-0.0090, -0.0291]
staleness = [-0.0229, -0.0338]
interaction = [-0.0430, -0.0972]
total = [-0.0749, -0.1601]

fig, ax = plt.subplots(figsize=(8, 5.2))
x = np.arange(2)
w = 0.5
b1 = ax.bar(x, density, w, label='Density (history truncation only)', color='#4C9F70')
b2 = ax.bar(x, staleness, w, bottom=density, label='Staleness (stale main cloud, full history)', color='#E8A33D')
b3 = ax.bar(x, interaction, w, bottom=[d+s for d, s in zip(density, staleness)],
            label='Interaction (super-additive)', color='#C0504D')
for xi, t in zip(x, total):
    ax.text(xi, t - 0.006, f'total {t:+.4f}', ha='center', va='top', fontsize=11, fontweight='bold')
for bars in (b1, b2, b3):
    for r in bars:
        ax.text(r.get_x() + r.get_width()/2, r.get_y() + r.get_height()/2,
                f'{r.get_height():+.4f}', ha='center', va='center', fontsize=9,
                color='white', fontweight='bold')
ax.axhline(-0.0376, ls='--', lw=1.2, color='gray')
ax.text(1.32, -0.0376, 'LiDAR-only\ngap −0.0376', va='center', fontsize=9, color='gray')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel('mAP change vs clean fusion (0.6855)', fontsize=11)
ax.set_title('E7: stale-LiDAR mAP loss decomposition (paired subset n = 5,869)', fontsize=12, fontweight='bold')
ax.legend(loc='lower left', fontsize=9.5)
ax.set_ylim(-0.185, 0.005)
fig.savefig('D:/Camera-LIDAR/results/chart_e7_decomposition.png', bbox_inches='tight', dpi=160)
fig.savefig('D:/Camera-LIDAR/results/chart_e7_decomposition.svg', bbox_inches='tight')
print('saved')
