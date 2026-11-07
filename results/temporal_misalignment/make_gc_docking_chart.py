import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.executable).parent.parent.parent))
from daimon_runtime import setup_plot
import matplotlib.pyplot as plt
import numpy as np

setup_plot()

# ---- Panel A: per-class conditional backfire gap G_c at k10 (miss-rate basis, E4 corrected FN) ----
classes = ["traffic_cone", "motorcycle", "bicycle", "pedestrian", "car",
           "construction_vehicle", "trailer", "bus", "barrier", "truck"]
G = [8.9, 4.8, 4.2, 3.0, 2.7, 1.8, 0.6, -0.4, -1.1, -1.2]  # percentage points
B = [4.4, 5.3, 6.0, 1.1, 2.3, 2.9, 0.9, 3.9, 3.3, 5.4]
D = [13.3, 10.1, 10.2, 4.1, 5.0, 4.7, 1.5, 3.5, 2.2, 4.2]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

y = np.arange(len(classes))
colors = ["#c0392b" if g > 0 else "#27ae60" for g in G]
ax1.barh(y, G, color=colors, alpha=0.85)
ax1.set_yticks(y, classes)
ax1.invert_yaxis()
ax1.axvline(0, color="black", lw=0.8)
ax1.set_xlabel("Conditional backfire gap $G_c$ = FN%$_{k10}$ - FN%$_{LiDAR}$ (pt)")
ax1.set_title("A. Per-class backfire gap at k10 (~958 ms)\nmiss-rate basis, nuScenes val common subset n=5712")
for i, (g, b, d) in enumerate(zip(G, B, D)):
    ax1.text(g + (0.15 if g >= 0 else -0.15), i, f"G={g:+.1f}  (B={b:.1f}, D={d:.1f})",
             va="center", ha="left" if g >= 0 else "right", fontsize=8)
ax1.set_xlim(-4.5, 16)

# ---- Panel B: distance-dependent gap s_hat(d) for car / truck / bus (E1 range bins, k8) ----
bins = ["0-10", "10-20", "20-30", "30-40", "40-60"]
miss = {
    "car":    {"clean": [0.3, 0.7, 1.6, 2.9, 6.1], "k8": [0.4, 0.7, 2.2, 5.4, 10.3], "lidar": [0.0, 0.7, 2.0, 4.4, 9.2]},
    "truck":  {"clean": [1.9, 4.0, 6.1, 5.9, 15.9], "k8": [1.8, 4.2, 6.1, 8.7, 20.0], "lidar": [1.5, 4.5, 6.2, 7.9, 20.7]},
    "bus":    {"clean": [0.9, 1.7, 3.1, 8.3, 10.3], "k8": [0.9, 1.2, 3.9, 9.9, 12.4], "lidar": [0.0, 1.5, 4.1, 12.6, 14.3]},
}
x = np.arange(len(bins))
for name, m in miss.items():
    g = [a - b for a, b in zip(m["k8"], m["lidar"])]
    ax2.plot(x, g, marker="o", label=name)
ax2.axhline(0, color="black", lw=0.8)
ax2.set_xticks(x, bins)
ax2.set_xlabel("Object distance (m)")
ax2.set_ylabel("$\\hat{s}(d)$ = miss%$_{k8}$ - miss%$_{LiDAR}$ (pt)")
ax2.set_title("B. Distance-dependent gap (candidate weighting proxy)\nrecall@2m custom matching, k8 (~700 ms)")
ax2.legend(title="class")
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(r"D:\Camera-LIDAR\results\chart_gc_docking.png", bbox_inches="tight")
print("SAVED")
