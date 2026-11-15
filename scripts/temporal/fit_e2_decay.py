# Fit exponential decay mAP(d) = a + b*exp(-d/tau) for T1 (stale camera) and
# E2 (stale LiDAR). Prints fitted parameters and R^2.
import numpy as np
from scipy.optimize import curve_fit

def f(d, a, b, tau):
    return a + b * np.exp(-d / tau)

# T1: stale camera, k=0..10 (n=5712 paired subset)
d_t1 = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], float)
m_t1 = np.array([0.6869, 0.6823, 0.6712, 0.6640, 0.6593, 0.6557, 0.6515, 0.6488, 0.6470, 0.6448, 0.6432])

# E2: stale LiDAR, k=1..9 (n=5856 paired subset, mean delays from e2_delay_table.md)
d_e2 = np.array([50.1, 100.5, 150.6, 200.7, 251.1, 301.2, 351.4, 401.6, 451.7])
m_e2 = np.array([0.6247, 0.5394, 0.4744, 0.4133, 0.3606, 0.3187, 0.2869, 0.2448, 0.1787])

for name, d, m in (('T1', d_t1, m_t1), ('E2', d_e2, m_e2)):
    p, _ = curve_fit(f, d, m, p0=(m.min(), m.max() - m.min(), 300), maxfev=20000)
    pred = f(d, *p)
    r2 = 1 - np.sum((m - pred) ** 2) / np.sum((m - m.mean()) ** 2)
    print(f'{name}: mAP = {p[0]:.4f} + {p[1]:.4f}*exp(-d/{p[2]:.0f} ms)  R2={r2:.4f}')
