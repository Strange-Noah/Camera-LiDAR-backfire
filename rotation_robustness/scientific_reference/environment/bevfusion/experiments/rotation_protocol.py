"""Camera-local persistent rotations, independent of all global RNG streams."""
import hashlib
import math
import numpy as np

VERSION = 'rotation-v1'
CHANNELS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')


def stable_seed(run_seed, channel):
    if channel not in CHANNELS or not isinstance(run_seed, int):
        raise ValueError('Invalid seed/channel')
    return int.from_bytes(hashlib.sha256(f'{VERSION}:{run_seed}:{channel}'.encode()).digest()[:4], 'little')


def parameters(sigma, seed, channel):
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError('Sigma must be finite nonnegative degrees per axis')
    local_seed = stable_seed(seed, channel)
    z = np.random.default_rng(local_seed).normal(size=3)
    omega = np.deg2rad(sigma) * z
    theta = np.linalg.norm(omega)
    x, y, zc = omega
    skew = np.array([[0., -zc, y], [zc, 0., -x], [-y, x, 0.]])
    # Stable Rodrigues coefficients, including the exact zero-angle limit.
    q = np.eye(3) + np.sinc(theta / np.pi) * skew + .5 * np.sinc(theta / (2*np.pi))**2 * (skew @ skew)
    return dict(channel=channel, sigma_R_degrees_per_axis=float(sigma), run_seed=seed,
                channel_seed=local_seed, z=z.tolist(), omega_radians=omega.tolist(),
                Q=q.tolist(), angle_degrees=float(np.rad2deg(theta)), version=VERSION)


def calibrated_matrices(camera2lidar, intrinsic, param):
    t = np.array(camera2lidar, dtype=np.float64, copy=True)
    k = np.array(intrinsic, dtype=np.float64, copy=True)
    if t.shape != (4, 4) or k.shape != (4, 4) or not np.isfinite(t).all() or not np.isfinite(k).all():
        raise ValueError('Expected finite padded 4x4 calibration')
    if not np.allclose(t[3], [0, 0, 0, 1]) or not np.allclose(t[:3,:3].T @ t[:3,:3], np.eye(3), atol=1e-6):
        raise ValueError('Invalid clean rigid transform')
    if abs(np.linalg.det(t[:3,:3])-1)>1e-6:
        raise ValueError('Clean rotation has improper determinant')
    q = np.asarray(param['Q'], dtype=np.float64)
    if q.shape != (3, 3) or not np.isfinite(q).all():
        raise ValueError('Expected finite 3x3 SO(3) perturbation')
    if np.linalg.norm(q.T @ q - np.eye(3)) >= 1e-6 or abs(np.linalg.det(q)-1) >= 1e-6:
        raise ValueError('Invalid SO(3) perturbation')
    t[:3,:3] = t[:3,:3] @ q
    inverse = np.linalg.inv(t)
    return t, inverse, k @ inverse
