"""Coordinate and finite-difference primitives with explicit missingness."""
import numpy as np
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, atol=1e-3):
        raise ValueError('Invalid unit quaternion')
    return Quaternion(q).rotation_matrix


def yaw(q):
    r = rotation(q)
    return np.arctan2(r[1, 0], r[0, 0])


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def differences(times_us, values, max_gap_s):
    """Secant central difference for unequal spacing, one-sided only at endpoints.

    Each adjacent gap must pass, including both gaps of a central stencil.
    This is deliberately stricter than devkit's total-span criterion.
    """
    t = (np.asarray(times_us, dtype=np.int64) - int(times_us[0])) / 1e6
    x = np.asarray(values, dtype=float)
    n = len(t)
    v = np.full_like(x, np.nan, dtype=float)
    span = np.full(n, np.nan)
    neighbors = np.zeros(n, dtype=int)
    method = np.full(n, 'unavailable', dtype=object)
    if n > 1 and np.any(np.diff(t) <= 0):
        raise ValueError('Non-increasing trajectory time')
    for i in range(n):
        lo, hi = max(0, i - 1), min(n - 1, i + 1)
        gaps = np.diff(t[lo:hi + 1])
        neighbors[i] = np.sum((gaps > 0) & (gaps <= max_gap_s))
        if lo == hi or np.any(gaps > max_gap_s):
            continue
        span[i] = t[hi] - t[lo]
        v[i] = (x[hi] - x[lo]) / span[i]
        method[i] = 'central_secant' if lo < i < hi else 'one_sided'
    return v, span, neighbors, method


def relative_components(position_xy, ego_xy, velocity_xy, ego_velocity_xy):
    d = np.asarray(position_xy) - np.asarray(ego_xy)
    rel = np.asarray(velocity_xy) - np.asarray(ego_velocity_xy)
    z = np.linalg.norm(d, axis=-1)
    unit = np.divide(d, z[..., None], out=np.full_like(d, np.nan, dtype=float), where=z[..., None] > 0)
    radial = np.sum(rel * unit, axis=-1)
    signed_lateral = unit[..., 0] * rel[..., 1] - unit[..., 1] * rel[..., 0]
    return rel, radial, signed_lateral


def corners_world(centers, sizes, quaternions):
    # nuScenes size order [width, length, height], local x=length, y=width.
    signs = np.array([[1,1,1],[1,-1,1],[1,-1,-1],[1,1,-1],[-1,1,1],[-1,-1,1],[-1,-1,-1],[-1,1,-1]])
    scales = np.asarray(sizes)[:, [1, 0, 2]] / 2
    local = scales[:, None, :] * signs[None, :, :]
    rotations = np.array([rotation(q) for q in quaternions])
    return np.einsum('nij,nkj->nki', rotations, local) + np.asarray(centers)[:, None, :]


def project_boxes(corners, ego, calibrated, width, height, near=0.1):
    """World -> camera via full SE(3); no object-time interpolation.

    Use all-positive-depth corners and convex hull/image intersection. Width is
    UNCLIPPED projected 3D-box envelope, never an occlusion-aware 2D detection.
    Near-plane crossings rejected instead of inventing finite widths.
    """
    cam = ((corners - np.asarray(ego['translation'])) @ rotation(ego['rotation'])
           - np.asarray(calibrated['translation'])) @ rotation(calibrated['rotation'])
    k = np.asarray(calibrated['camera_intrinsic'])
    if k.shape != (3, 3) or not np.isfinite(k).all() or width <= 0 or height <= 0:
        raise ValueError('Invalid camera intrinsics/image dimensions')
    depth_valid = np.isfinite(cam).all(axis=(1, 2)) & (cam[:, :, 2] > near).all(axis=1)
    uvh = cam @ k.T
    uv = np.divide(uvh[:, :, :2], uvh[:, :, 2:3], out=np.full_like(uvh[:, :, :2], np.nan), where=uvh[:, :, 2:3] > near)
    visible = np.zeros(len(cam), dtype=bool)
    partial = np.zeros(len(cam), dtype=bool)
    w = np.full(len(cam), np.nan)
    h = w.copy()
    canvas = box(0, 0, width, height)
    for i in np.flatnonzero(depth_valid):
        hull = MultiPoint(uv[i]).convex_hull
        if hull.area > 0 and hull.intersection(canvas).area > 0:
            visible[i] = True
            lo, hi = uv[i].min(axis=0), uv[i].max(axis=0)
            w[i], h[i] = hi - lo
            partial[i] = bool((lo < 0).any() or hi[0] > width or hi[1] > height)
    return {'camera_visible': visible, 'projection_valid': visible & depth_valid,
            'projection_depth_valid': depth_valid, 'partially_outside_image': partial,
            'projected_width_px': w, 'projected_height_px': h,
            'near_plane_crossing': (cam[:, :, 2].min(axis=1) <= near) & (cam[:, :, 2].max(axis=1) > near)}
