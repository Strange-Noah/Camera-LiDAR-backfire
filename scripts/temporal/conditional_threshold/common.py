"""Shared configuration, I/O helpers and statistics for the conditional-threshold pipeline."""
import argparse
import gc
import gzip
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Experiment constants (T1: stale camera, current LiDAR)
# --------------------------------------------------------------------------------------
# Mean camera delay per setting in ms (heyz convention; T1 and T1B use identical replacement frames).
# k9/k10 corrected from 851/958 (old estimates) to 758/834; k8 677 -> 672.
DELAY_MS = {'fusion_clean': 0, 'k1': 77, 'k2': 168, 'k3': 254, 'k4': 332, 'k5': 423,
            'k6': 507, 'k7': 586, 'k8': 672, 'k9': 758, 'k10': 834}
FUSION_SETTINGS = list(DELAY_MS)
SETTINGS = FUSION_SETTINGS + ['lidar_only']          # column order of every miss matrix
DELAYS = np.array([DELAY_MS[s] for s in FUSION_SETTINGS], dtype=float)

PRED_FILES = {'fusion_clean': 'fusion_cumulative.json', 'lidar_only': 'lidar_cumulative.json'}
PRED_FILES.update({f'k{k}': f'k{k}_cumulative.json' for k in range(1, 11)})
GT_FILE = 'gt_common.json'

RELEASE_URL = 'https://github.com/Strange-Noah/Camera-LiDAR-backfire/releases/download/predictions-v1'
RELEASE_ASSETS = ['gt_common.json', 'fusion_cumulative.json.gz', 'lidar_cumulative.json'] + \
                 [f'k{k}_cumulative.json.gz' for k in range(1, 11)]

CLASSES = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle',
           'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']

# nuScenes category -> detection class (official mapping)
DET_NAME = {
    'vehicle.bicycle': 'bicycle', 'vehicle.bus.bendy': 'bus', 'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car', 'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle', 'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian', 'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian', 'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer', 'vehicle.truck': 'truck', 'movable_object.barrier': 'barrier'}

# Matching rule (same as E4): greedy by score within (sample, class), ground-plane center distance.
MATCH_DIST_M = 2.0
SCORE_THRESHOLDS = [0.1, 0.3, 0.5]       # 0.1 is the main threshold; 0.3/0.5 are sensitivity checks
MAIN_THRESHOLD = 0.1

# Reference numbers from the team (common subset, score >= 0.1) used for reconciliation.
EXPECTED = {
    'n_samples': 5712,
    'n_boxes': 116027,
    'boxes_per_class': {'car': 50509, 'truck': 8800, 'bus': 1952, 'trailer': 2217, 'construction_vehicle': 1449,
                        'pedestrian': 22419, 'motorcycle': 1767, 'bicycle': 1750, 'traffic_cone': 10015, 'barrier': 15149},
    'night_samples': 566,
    'intersection_samples': 1982,
    'fn_all_unmatched': {'fusion_clean': 14892, 'k10': 20903, 'lidar_only': 18015},   # corrected E4
    'fn_old_e4': {'fusion_clean': 12607, 'k10': 17400, 'lidar_only': 15380},          # original E4 definition
    'fp': {'fusion_clean': 25907, 'k10': 22997, 'lidar_only': 28146},                 # E4 false positives (score >= 0.1)
}

# Condition bins. Tertile edges from the metadata audit (bin_definitions.json), right-closed.
INF = np.inf
CONDITIONS = {
    'distance_xy':               ([-INF, 21.44, 38.584, INF], ['near <21m', 'mid 21-39m', 'far >39m']),
    'distance_E1':               ([-INF, 10, 20, 30, 40, INF], ['0-10m', '10-20m', '20-30m', '30-40m', '>40m']),
    'relative_lateral_velocity': ([-INF, 0.855, 3.098, INF], ['lateral slow', 'lateral mid', 'lateral fast']),
    'object_speed':              ([-INF, 0.013, 0.176, INF], ['static', 'slow', 'moving']),
    'ego_speed':                 ([-INF, 2.869, 6.573, INF], ['ego slow', 'ego mid', 'ego fast']),
    'num_lidar_pts':             ([-INF, 2, 10, INF], ['pts 0-2', 'pts 3-10', 'pts >10']),
    'abs_ego_yaw_rate':          ([-INF, 0.003, 0.018, INF], ['yaw low', 'yaw mid', 'yaw high']),
    'relative_radial_velocity':  ([-INF, -1.616, 1.328, INF], ['approaching', 'parallel', 'receding']),
    'r500':                      ([-INF, 0.1, 0.5, 1.0, INF], ['r<0.1', '0.1<=r<0.5', '0.5<=r<1', 'r>=1']),
}
R_BINS = CONDITIONS['r500'][1]

# Support rule from the metadata audit.
SUPPORT = dict(min_traj=30, min_scene=10, max_top1=0.30, weak_traj=5, weak_scene=3)

BOOTSTRAP_B = 500
BOOTSTRAP_SEED = 2026

# Screening rule for "robust" conditional findings.
SCREEN = dict(min_shift_ms=50.0, max_ci_width_ms=400.0)


# --------------------------------------------------------------------------------------
# Paths / CLI
# --------------------------------------------------------------------------------------
def get_args(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--root', default='/content/drive/MyDrive/nuscenes',
                   help='folder that contains v1.0-trainval/ and team_data/ (default: Colab Drive layout)')
    p.add_argument('--meta', default=None, help='nuScenes metadata folder (default: <root>/v1.0-trainval)')
    p.add_argument('--data', default=None, help='team prediction files folder (default: <root>/team_data)')
    p.add_argument('--out', default=None, help='output folder (default: <root>/ct_results)')
    a = p.parse_args()
    a.meta = a.meta or os.path.join(a.root, 'v1.0-trainval')
    a.data = a.data or os.path.join(a.root, 'team_data')
    a.out = a.out or os.path.join(a.root, 'ct_results')
    for sub in ('', 'labels', 'tables', 'figures'):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    return a


def log(*msg):
    print(*msg, flush=True)


def find_file(folder, name):
    """Return <folder>/<name> or <folder>/<name>.gz, whichever exists."""
    for p in (os.path.join(folder, name), os.path.join(folder, name + '.gz')):
        if os.path.exists(p):
            return p
    return None


def open_text(path):
    return gzip.open(path, 'rt') if path.endswith('.gz') else open(path, 'r')


def load_json(path):
    with open_text(path) as f:
        return json.load(f)


def save_csv(df, path, **kw):
    """UTF-8 with BOM so Excel shows symbols correctly."""
    df.to_csv(path, encoding='utf-8-sig', **kw)


def _get_ijson():
    try:
        import ijson
        return ijson
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ijson'], capture_output=True)
        try:
            import ijson
            return ijson
        except ImportError:
            return None


def stream_rows(path, keep=None):
    """Read a large JSON array and keep only rows where keep(row) is True (streams with ijson if available)."""
    t0 = time.time()
    ijson = _get_ijson()
    out = []
    if ijson is not None:
        with open(path, 'rb') as f:
            try:
                it = ijson.items(f, 'item', use_float=True)
            except TypeError:
                it = ijson.items(f, 'item')
            for r in it:
                if keep is None or keep(r):
                    out.append(r)
    else:
        with open(path) as f:
            data = json.load(f)
        out = [r for r in data if keep is None or keep(r)]
        del data
        gc.collect()
    log(f'  {os.path.basename(path)}: kept {len(out):,} rows in {time.time() - t0:.0f}s')
    return out


# --------------------------------------------------------------------------------------
# Geometry (identical to the metadata audit, src/metadata_audit/core.py)
# --------------------------------------------------------------------------------------
def yaw_of(q):
    """Yaw angle from quaternion [w, x, y, z]."""
    w, x, y, z = q
    return math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


def differences(times_us, values, max_gap_s):
    """Central differences on an irregular time grid (one-sided at the ends).
    NaN when any adjacent gap inside the window exceeds max_gap_s."""
    t = (np.asarray(times_us, dtype=np.int64) - int(times_us[0])) / 1e6
    x = np.asarray(values, dtype=float)
    n = len(t)
    v = np.full_like(x, np.nan, dtype=float)
    if n > 1 and np.any(np.diff(t) <= 0):
        raise ValueError('timestamps are not strictly increasing')
    for i in range(n):
        lo, hi = max(0, i - 1), min(n - 1, i + 1)
        if lo == hi or np.any(np.diff(t[lo:hi + 1]) > max_gap_s):
            continue
        v[i] = (x[hi] - x[lo]) / (t[hi] - t[lo])
    return v


def relative_components(position_xy, ego_xy, velocity_xy, ego_velocity_xy):
    """Relative velocity, radial component (positive = receding) and signed lateral component."""
    d = np.asarray(position_xy) - np.asarray(ego_xy)
    rel = np.asarray(velocity_xy) - np.asarray(ego_velocity_xy)
    z = np.linalg.norm(d, axis=-1)
    unit = np.divide(d, z[..., None], out=np.full_like(d, np.nan, dtype=float), where=z[..., None] > 0)
    radial = np.sum(rel * unit, axis=-1)
    signed_lateral = unit[..., 0] * rel[..., 1] - unit[..., 1] * rel[..., 0]
    return rel, radial, signed_lateral


# --------------------------------------------------------------------------------------
# Loading the per-box tables produced by step 1 and step 2
# --------------------------------------------------------------------------------------
def load_features(out):
    gtf = pd.read_csv(os.path.join(out, 'gt_features.csv.gz')).sort_values('box_id').reset_index(drop=True)
    for col, (edges, labels) in CONDITIONS.items():
        src = 'distance_xy' if col == 'distance_E1' else col
        gtf[col + '_bin'] = pd.cut(gtf[src], bins=edges, labels=labels, right=True)
    gtf['visibility_bin'] = gtf.visibility.map({1: 'v1:0-40%', 2: 'v2:40-60%', 3: 'v3:60-80%', 4: 'v4:80-100%'})
    gtf['night_bin'] = gtf.night.map({0: 'day', 1: 'night'})
    gtf['intersection_bin'] = gtf.intersection.map({0: 'non-intersection', 1: 'intersection'})
    return gtf


COND_COLS = [c + '_bin' for c in CONDITIONS] + ['visibility_bin', 'night_bin', 'intersection_bin']


def load_miss(out, threshold=MAIN_THRESHOLD, definition='all'):
    """Miss matrix (n_boxes x 12), columns in SETTINGS order (last column = LiDAR-only).
    definition='all'    : every unmatched GT box is a miss (corrected E4, main result)
    definition='old_e4' : unmatched AND the sample has >=1 same-class prediction (original E4)"""
    cols = []
    for s in SETTINGS:
        z = np.load(os.path.join(out, 'labels', f'{s}.npz'))
        hit = z[f'hit_{threshold}']
        if definition == 'all':
            cols.append(~hit)
        elif definition == 'old_e4':
            cols.append(~hit & z[f'present_{threshold}'])
        else:
            raise ValueError(definition)
    return np.column_stack(cols).astype(float)


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def support_status(g):
    """SUPPORTED / WEAK / SPARSE following the metadata-audit rule."""
    nt, ns = g.instance_token.nunique(), g.scene_token.nunique()
    top1 = g.groupby('scene_token').size().max() / len(g) if len(g) else np.nan
    if nt >= SUPPORT['min_traj'] and ns >= SUPPORT['min_scene'] and top1 <= SUPPORT['max_top1']:
        st = 'SUPPORTED'
    elif nt >= SUPPORT['weak_traj'] and ns >= SUPPORT['weak_scene']:
        st = 'WEAK'
    else:
        st = 'SPARSE'
    return nt, ns, top1, st


def crossover(gap, delays=DELAYS):
    """Delay (ms) from which gap stays > 0, linearly interpolated.
    Returns (ms, kind): kind = 'cross' | 'always' (already worse at clean) | 'none' (never sustained)."""
    pos = gap > 0
    if not pos[-1]:
        return np.nan, 'none'
    i = len(gap) - 1
    while i > 0 and pos[i - 1]:
        i -= 1
    if i == 0:
        return 0.0, 'always'
    g0, g1 = gap[i - 1], gap[i]
    return float(delays[i - 1] + (0 - g0) * (delays[i] - delays[i - 1]) / (g1 - g0)), 'cross'


class SceneBootstrap:
    """Scene-level bootstrap with one fixed set of resampling weights (reproducible, shared by all cells)."""

    def __init__(self, scene_tokens, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED):
        self.codes, self.ids = pd.factorize(pd.Series(scene_tokens))
        self.S = len(self.ids)
        self.W = np.random.default_rng(seed).multinomial(self.S, np.ones(self.S) / self.S, size=B).astype(float)

    def sums(self, mask, MISS):
        idx = np.flatnonzero(mask)
        sc = self.codes[idx]
        n_s = np.bincount(sc, minlength=self.S).astype(float)
        m_s = np.column_stack([np.bincount(sc, weights=MISS[idx, j], minlength=self.S) for j in range(MISS.shape[1])])
        return n_s, m_s

    def boot_gaps(self, n_s, m_s):
        bn, bm = self.W @ n_s, self.W @ m_s
        with np.errstate(invalid='ignore', divide='ignore'):
            rate = bm / bn[:, None]
        return rate[:, :-1] - rate[:, [-1]]           # (B, 11)

    def analyze(self, mask, MISS):
        """Point estimates + 95% CIs for one group of GT boxes.
        gap = miss_rate(fusion, k) - miss_rate(LiDAR-only); benefit_lost = gap(k10) - gap(clean), in pct points."""
        n = int(np.sum(mask))
        if n == 0:
            return empty_result(0, 'empty')
        n_s, m_s = self.sums(mask, MISS)
        rate = m_s.sum(0) / n_s.sum()
        gap = rate[:-1] - rate[-1]
        est, kind = crossover(gap)
        bgap = self.boot_gaps(n_s, m_s)
        bx = np.array([crossover(g)[0] if np.all(np.isfinite(g)) else np.nan for g in bgap])
        fin = np.isfinite(bx)
        lo, hi = (np.percentile(bx[fin], [2.5, 97.5]) if fin.sum() >= 10 else (np.nan, np.nan))
        bl = (bgap[:, -1] - bgap[:, 0]) * 100
        bl = bl[np.isfinite(bl)]
        return dict(N_boxes=n, crossover_ms=est, kind=kind, ci_lo=lo, ci_hi=hi, frac_cross=float(fin.mean()),
                    miss_clean=float(rate[0]), miss_k10=float(rate[-2]), miss_lidar=float(rate[-1]),
                    gap_clean=float(gap[0] * 100), gap_k10=float(gap[-1] * 100),
                    benefit_lost=float((gap[-1] - gap[0]) * 100),
                    bl_lo=float(np.percentile(bl, 2.5)), bl_hi=float(np.percentile(bl, 97.5)))


def empty_result(n, kind):
    return dict(N_boxes=int(n), crossover_ms=np.nan, kind=kind, ci_lo=np.nan, ci_hi=np.nan, frac_cross=np.nan,
                miss_clean=np.nan, miss_k10=np.nan, miss_lidar=np.nan, gap_clean=np.nan, gap_k10=np.nan,
                benefit_lost=np.nan, bl_lo=np.nan, bl_hi=np.nan)


def group_mask(gtf, group):
    return np.ones(len(gtf), bool) if group == 'ALL' else gtf.cls.eq(group).to_numpy()


def fmt_cross(r):
    if not np.isfinite(r['crossover_ms']):
        return r['kind']
    if np.isfinite(r['ci_lo']):
        return f"{r['crossover_ms']:.0f} [{r['ci_lo']:.0f}, {r['ci_hi']:.0f}]"
    return f"{r['crossover_ms']:.0f}"
