"""T1B settings (ego-consistent delayed camera)."""
# T1B: ego-consistent delayed camera (each historical frame is delivered with its own capture-time
# calibration; LiDAR current). Same historical images as T1, so T1 and T1B can be compared per frame index k.
T1B_K = [2, 5, 8, 10, 12]
T1B_SETTINGS = [f't1b_k{k}' for k in T1B_K]
T1B_FILES = {f't1b_k{k}': f't1b_k{k}_cumulative.json' for k in T1B_K}
T1B_RELEASE_URL = 'https://github.com/Strange-Noah/Camera-LiDAR-backfire/releases/download/predictions-t1b'
T1B_RELEASE_ASSETS = [f't1b_k{k}_cumulative.json.gz' for k in T1B_K]
