"""Separate paper-method reconstruction. Frozen minimal adapter is never edited.

Reference: Moravec & Sara, TRO 2024, Sections III-C/G and Appendix A.
Dataset conventions and unpublished choices are recorded in FULL_METHOD.md.
"""
import numpy as np


def classify_final(initial, final, valid_updates, tolerance=1e-8):
    """A roundoff-sized change or a held estimate is not calibration recovery."""
    if valid_updates == 0:
        return 'NO_VALID_UPDATE'
    if abs(final-initial) <= tolerance:
        return 'UNCHANGED'
    return 'IMPROVED' if final < initial else 'DEGRADED'


def exp_rotation(v):
    v = np.asarray(v, dtype=float)
    angle = np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    a = np.sinc(angle / np.pi)
    b = .5 * np.sinc(angle / (2 * np.pi)) ** 2
    return np.eye(3) + a * k + b * (k @ k)


def rotation_error(matrix):
    # atan2 is accurate near zero, unlike trace/arccos alone.
    v = np.array([matrix[2, 1]-matrix[1, 2], matrix[0, 2]-matrix[2, 0],
                  matrix[1, 0]-matrix[0, 1]]) / 2
    return float(np.arctan2(np.linalg.norm(v), (np.trace(matrix)-1)/2))


def rigid(record):
    w, x, y, z = np.asarray(record['rotation'], float)
    assert np.isclose(w*w+x*x+y*y+z*z, 1, atol=1e-5)
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = record['translation']
    return t


def camera_points(points, camera_to_lidar, theta):
    """Right camera-local correction; truth theta=-omega stays fixed in time."""
    r = camera_to_lidar[:3, :3] @ exp_rotation(theta)
    return (points - camera_to_lidar[:3, 3]) @ r


def project(points_camera, intrinsics, width, height):
    good = points_camera[:, 2] > 0
    uvh = points_camera @ intrinsics.T
    uv = np.full((len(points_camera), 2), np.nan)
    uv[good] = uvh[good, :2] / uvh[good, 2:3]
    good &= (uv[:, 0] >= 0) & (uv[:, 0] < width)
    good &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, good


def loss(tree, corners, camera_to_lidar, intrinsics, shape, theta):
    uv, good = project(camera_points(corners, camera_to_lidar, theta),
                       intrinsics, shape[1], shape[0])
    distances, _ = tree.query(uv[good], k=10)
    return float(-np.exp(-distances**2 / (2*9**2)).sum())


class PaperTracker:
    """Published MATLAB SGDSchaul rotational tracking, with a callable loss.

    Rejected observations do not enter this class. No new optimizer rule, line
    search, Hessian damping, per-sigma tuning or outcome-dependent reset.
    """
    def __init__(self):
        self.theta = np.zeros(3)
        self.g = np.zeros(3)
        self.v = np.zeros(3)
        self.h = np.zeros(3)
        self.memory = np.ones(3)
        self.pending_g = np.zeros(3)
        self.pending_v = np.zeros(3)
        self.pending_h = np.zeros(3)
        self.pending_count = 0
        self.accepted = 0
        self.updates = 0

    def update(self, objective):
        theta = self.theta
        step = .001 * np.eye(3)
        center = objective(theta)
        plus = np.array([objective(theta+d) for d in step])
        minus = np.array([objective(theta-d) for d in step])
        gradient = (plus-minus)/.002
        curvature = np.abs((plus-2*center+minus)/.001**2)
        if not np.isfinite(np.r_[center, plus, minus]).all():
            raise FloatingPointError('Nonfinite objective; do not accept this run')
        self.accepted += 1
        self.pending_g += gradient
        self.pending_v += gradient**2
        self.pending_h += curvature
        self.pending_count += 1
        if self.accepted <= 10:
            return {'status': 'BURN_IN', 'delta': [0., 0., 0.]}
        gradient = self.pending_g/self.pending_count
        squared_gradient = self.pending_v/self.pending_count
        curvature = self.pending_h/self.pending_count
        rate = 1/self.memory
        self.g = (1-rate)*self.g + rate*gradient
        self.v = (1-rate)*self.v + rate*squared_gradient
        self.h = (1-rate)*self.h + rate*curvature
        epsilon = np.finfo(float).eps  # MATLAB eps, not Python example's 1e-7.
        ratio = self.g**2/(self.v+epsilon)
        self.memory = np.minimum((1-ratio)*self.memory+1, 5)
        self.pending_g[:] = 0
        self.pending_v[:] = 0
        self.pending_h[:] = 0
        self.pending_count = 0
        # Undefined division is reported, never quietly made into a successful run.
        if np.any(self.h <= 0) or not np.isfinite(self.h).all():
            return {'status': 'INVALID_CURVATURE', 'delta': [0., 0., 0.]}
        proposal = gradient/self.h
        delta = -ratio*np.minimum(np.abs(proposal), .0024)*proposal/(np.abs(proposal)+epsilon)
        self.theta = theta + delta
        self.updates += 1
        return {'status': 'UPDATED', 'delta': delta.tolist(),
                'gradient': gradient.tolist(), 'curvature': curvature.tolist()}


def _response(values):
    mask = -np.arange(-5, 6)*np.exp(-np.arange(-5, 6)**2/2)
    denominator = np.sqrt(np.convolve(values**2, np.ones(11), mode='same'))
    return np.divide(np.abs(np.convolve(values, mask, mode='same')), denominator,
                     out=np.zeros(len(values)), where=denominator > 0)


def _nms(response, radius, threshold):
    good = response > threshold
    for offset in range(1, radius+1):
        good[offset:] &= response[offset:] >= response[:-offset]
        good[:-offset] &= response[:-offset] >= response[offset:]
    # Entire derivative and suppression neighborhoods must be real, not padded.
    good[:5+radius] = False
    good[-5-radius:] = False
    return good


def extract_corners(points):
    """32 independent azimuth-ordered scanlines; range, intensity and gap union.

    nuScenes intensity is 0..255; scale to 0..1 before the dimensionless response.
    Circular padding makes the -pi/pi seam an ordinary scanline neighborhood.
    """
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 5 or not np.isfinite(points).all():
        raise ValueError('Expected finite x,y,z,intensity,ring records')
    rings = points[:, 4]
    if not (np.allclose(rings, np.round(rings)) and
            np.all((rings >= 0) & (rings < 32))):
        raise ValueError('Invalid nuScenes ring IDs')
    selected = set()
    stats = {'range': 0, 'intensity': 0, 'gap': 0}
    for ring in np.unique(rings):
        ids = np.flatnonzero((rings == ring) & (np.linalg.norm(points[:, :3], axis=1) > 0))
        ids = ids[np.argsort(np.arctan2(points[ids, 1], points[ids, 0]), kind='stable')]
        if len(ids) < 17:
            continue
        az = np.arctan2(points[ids, 1], points[ids, 0])
        # Drop exact duplicate azimuths deterministically; each azimuth must be unique.
        ids = ids[np.r_[True, np.diff(az) > 1e-12]]
        if len(ids) < 17:
            continue
        az = np.arctan2(points[ids, 1], points[ids, 0])
        distance = np.linalg.norm(points[ids, :3], axis=1)
        pad = 8
        intensity = np.clip(points[ids, 3]/255., 0, 1)
        for name, values, radius, threshold in [
                ('range', distance, 2, .01), ('intensity', intensity, 3, .05)]:
            candidates = np.flatnonzero(_nms(_response(np.pad(values, pad, mode='wrap')),
                                             radius, threshold)[pad:-pad])
            for index in candidates:
                neighbors = [(index-1) % len(ids), index, (index+1) % len(ids)]
                # Match official closer-neighbor displacement rule (0.5m), also
                # for reflectance corners as Appendix A says repeat steps a-c.
                closer = [j for j in neighbors if distance[j] < distance[index]-.5]
                chosen = min(closer, key=lambda j: distance[j]) if closer else index
                selected.add(int(ids[chosen]))
            stats[name] += len(candidates)
        gaps = np.mod(np.roll(az, -1)-az, 2*np.pi) > .1
        for index in np.flatnonzero(gaps):
            selected.update([int(ids[index]), int(ids[(index+1) % len(ids)])])
        stats['gap'] += int(2*gaps.sum())
    ids = np.array(sorted(selected), dtype=int)
    return points[ids, :3], ids, stats


def camera_edges(image_bgr):
    import cv2
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Published MATLAB uses grayscale + automatic Canny. OpenCV is an explicit
    # fixed Python-port approximation; no claim of MATLAB pixel identity.
    edges = cv2.Canny(gray, 50, 100).astype(bool)
    objective_edges = edges.copy()
    objective_edges[:image_bgr.shape[0]//3] = False
    return edges, objective_edges


def quality_features(corners, camera_to_lidar, intrinsics, edges):
    """30 radial/left-right corner counts + 6*(edge count, anisotropy mean)."""
    pc = camera_points(corners, camera_to_lidar, np.zeros(3))
    _, good = project(pc, intrinsics, edges.shape[1], edges.shape[0])
    distance = np.linalg.norm(corners, axis=1)
    values = []
    for bin_index in range(15):
        inside = good & (distance >= 5*bin_index) & (distance < 5*(bin_index+1))
        values.extend([np.count_nonzero(inside & (pc[:, 0] < 0)),
                       np.count_nonzero(inside & (pc[:, 0] >= 0))])
    # 20 neighbors: offsets -10..-1,+1..+10, excluding center, zero outside image.
    nv = np.zeros(edges.shape, dtype=np.int16)
    nh = np.zeros(edges.shape, dtype=np.int16)
    for offset in range(1, 11):
        nv[offset:] += edges[:-offset]; nv[:-offset] += edges[offset:]
        nh[:, offset:] += edges[:, :-offset]; nh[:, :-offset] += edges[:, offset:]
    # Zero-denominator convention is unpublished; explicit finite count floor=1.
    quotient = np.maximum(nv, nh)/np.maximum(np.minimum(nv, nh), 1)
    ys = np.linspace(edges.shape[0]//4, edges.shape[0], 3, dtype=int)
    xs = np.linspace(0, edges.shape[1], 4, dtype=int)
    for row in range(2):
        for col in range(3):
            e = edges[ys[row]:ys[row+1], xs[col]:xs[col+1]]
            q = quotient[ys[row]:ys[row+1], xs[col]:xs[col+1]]
            values.extend([int(e.sum()), float(q[e].mean()) if e.any() else 0.])
    return np.asarray(values, dtype=float)


def automatic_label(objective):
    grid = np.arange(-10, 11)*.005
    profiles = np.array([[objective(np.eye(3)[axis]*value) for value in grid]
                         for axis in range(3)])
    minima = grid[np.argmin(profiles, axis=1)]
    # All tied minima must lie within tolerance; flat/uninformative is unsuitable.
    good = all(np.all(np.abs(grid[np.isclose(p, p.min(), rtol=0, atol=1e-10)])
                         <= .01+1e-12) for p in profiles)
    return int(good), minima.tolist()
