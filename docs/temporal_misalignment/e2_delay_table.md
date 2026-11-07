# E2 delay distribution (stale LiDAR = own k-th past sweep, 20 Hz sensor)

Computed from nuscenes_infos_val.pkl: delay = sample timestamp − sweeps[k-1] timestamp.

| k | n eligible | min (ms) | mean (ms) | median (ms) | max (ms) |
|---|-----------|----------|-----------|-------------|----------|
| 1 | 5869 | 39.4 | 50.1 | 49.8 | 100.1 |
| 2 | 5869 | 91.8 | 100.5 | 100.1 | 199.7 |
| 3 | 5869 | 144.3 | 150.6 | 149.9 | 250.0 |
| 4 | 5869 | 193.9 | 200.7 | 200.2 | 299.7 |
| 5 | 5869 | 240.8 | 251.1 | 249.9 | 348.4 |
| 6 | 5869 | 290.6 | 301.2 | 300.2 | 399.8 |
| 7 | 5869 | 341.7 | 351.4 | 350.0 | 450.1 |
| 8 | 5869 | 393.7 | 401.6 | 399.8 | 499.9 |
| 9 | 5860 | 438.7 | 451.7 | 450.1 | 550.2 |

Notes:
- 9 fewer eligible samples at k=9 (sweep-count histogram {0:150, 8:9, 9:5860}).
- 4 additional tokens (20 camera frames) are excluded entirely: their current camera
  images do not exist in the public nuScenes blobs (2 scenes affected).
- E2 eligible universe = 6019 − 150 (0 sweeps) − 4 (bad camera) = 5865 unique tokens;
  per-k eligible n shown above (k ≤ 8: 5865; k = 9: 5856).
