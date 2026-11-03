"""Admission of already authenticated perturbations; never generates rotations.

Only the 24 identities in the reviewed original production plan use this path.
The frozen scientific source files and the matrix application code stay untouched.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from environment.bevfusion.experiments import experiment_manifest as original
from environment.bevfusion.experiments.rotation_protocol import CHANNELS, VERSION, stable_seed

LEGACY_VALIDATE = original.validate_manifest
REGISTRY_PATH = Path(__file__).with_name('frozen_manifest_registry.json')
# Trust is rooted in the reviewed package/registry, not a manifest's own self-hash.
REGISTRY = json.loads(REGISTRY_PATH.read_text())
SO3_BOUND = 1e-6  # Unchanged physical-validity bound in calibrated_matrices().


def validate_structure(body, row):
    if (body['schema_version'] != 1 or body['experiment'] != 'rotation'
            or body['intervention_version'] != VERSION or body['model'] != 'fusion'
            or body['dataset_version'] != 'v1.0-trainval' or body['split'] != 'val'):
        raise ValueError('Wrong frozen rotation contract')
    if type(body['run_seed']) is not int or body['run_seed'] != row['seed']:
        raise ValueError('Wrong frozen seed')
    sigma = body['sigma_R_degrees_per_axis']
    if type(sigma) is not float or not math.isfinite(sigma) or sigma != row['sigma']:
        raise ValueError('Wrong frozen severity')
    if any(body[k] != v for k,v in REGISTRY['immutable'].items()):
        raise ValueError('Frozen scientific identity changed')
    if body['matching'] != original.MATCHING or body['scientific_references'] != json.loads(Path(original.__file__).with_name('specification-lock.json').read_text()):
        raise ValueError('Frozen scientific contract metadata changed')
    subset = body['subset'];tokens = subset['sample_tokens']
    if (len(tokens) != REGISTRY['frames'] or len(set(tokens)) != len(tokens)
            or len(subset['scene_tokens']) != len(tokens)
            or original.identity(subset) != body['subset_sha256']
            or body['subset_sha256'] != REGISTRY['subset_sha256']):
        raise ValueError('Wrong frozen ordered population')
    if set(body['cameras']) != set(CHANNELS):
        raise ValueError('Exactly six Camera channels required')
    for channel in CHANNELS:
        p = body['cameras'][channel]
        if (p['channel'] != channel or p['version'] != VERSION
                or p['sigma_R_degrees_per_axis'] != sigma
                or type(p['run_seed']) is not int or p['run_seed'] != row['seed']
                or type(p['channel_seed']) is not int
                or p['channel_seed'] != stable_seed(row['seed'],channel)):
            raise ValueError('Frozen Camera identity mismatch: '+channel)
        z = np.asarray(p['z'],dtype=np.float64)
        omega = np.asarray(p['omega_radians'],dtype=np.float64)
        q = np.asarray(p['Q'],dtype=np.float64)
        if (z.shape != (3,) or omega.shape != (3,) or q.shape != (3,3)
                or not all(np.isfinite(v).all() for v in (z,omega,q))
                or not math.isfinite(p['angle_degrees']) or p['angle_degrees'] < 0):
            raise ValueError('Malformed frozen Camera arrays: '+channel)
        # Same SO(3) checks used by the unchanged physical injection routine.
        if np.linalg.norm(q.T @ q-np.eye(3)) >= SO3_BOUND or abs(np.linalg.det(q)-1) >= SO3_BOUND:
            raise ValueError('Invalid frozen SO(3): '+channel)
        # Mathematical consistency, not a bitwise re-realization. Hash admission
        # above already forbids changing even one saved scalar by one ULP.
        if np.linalg.norm(omega-np.deg2rad(sigma)*z) >= SO3_BOUND:
            raise ValueError('Inconsistent frozen rotation vector: '+channel)
        theta = float(np.linalg.norm(omega))
        if abs(math.radians(p['angle_degrees'])-theta) >= SO3_BOUND:
            raise ValueError('Inconsistent frozen angle diagnostic: '+channel)
        x,y,zc=omega;skew=np.array([[0.,-zc,y],[zc,0.,-x],[-y,x,0.]])
        # Use the scalar exponential map only as a mathematical validity check;
        # NEVER return it or replace Q. No random sampling is involved.
        q_check=np.eye(3)+np.sinc(theta/np.pi)*skew+.5*np.sinc(theta/(2*np.pi))**2*(skew@skew)
        if np.linalg.norm(q-q_check) >= SO3_BOUND:
            raise ValueError('Frozen Q/vector inconsistency: '+channel)
    return body


def validate_authenticated(manifest):
    cid=manifest.get('condition_id')
    if cid not in REGISTRY['conditions']:
        raise ValueError('Condition is not in the authenticated frozen production registry')
    row=REGISTRY['conditions'][cid]
    if original.identity(manifest['condition']) != cid:
        raise ValueError('Manifest condition identity changed')
    if hashlib.sha256(original.canonical(manifest)+b'\n').hexdigest() != row['manifest_sha256']:
        raise ValueError('Authenticated canonical manifest bytes changed')
    return validate_structure(manifest['condition'],row)


def validate_manifest(manifest):
    if manifest.get('condition_id') in REGISTRY['conditions']:
        return validate_authenticated(manifest)
    return LEGACY_VALIDATE(manifest)


def read_authenticated(path):
    data=Path(path).read_bytes();manifest=json.loads(data)
    body=validate_authenticated(manifest)
    row=REGISTRY['conditions'][manifest['condition_id']]
    if hashlib.sha256(data).hexdigest() != row['manifest_sha256']:
        raise ValueError('Authenticated original file bytes changed')
    return manifest,body


def install_before_consumers():
    """Explicit process-local compatibility bridge, before importing Runner B.

    Frozen modules bind validate_manifest at import time. Fail closed if a
    consumer already captured the legacy function; no partial installation.
    The bridge accepts only package-pinned IDs and falls back for every other ID.
    It changes no on-disk frozen source or matrix application function.
    """
    import sys
    if original.validate_manifest is validate_manifest:
        return
    consumers=('environment.bevfusion.experiments.handoff','analysis.throughput.runner',
               'analysis.throughput.writer','environment.bevfusion.experiments.run_rotation')
    if any(name in sys.modules for name in consumers):
        raise RuntimeError('Install authenticated admission before importing consumers')
    if original.validate_manifest is not LEGACY_VALIDATE:
        raise RuntimeError('Unexpected prior manifest validator override')
    original.validate_manifest=validate_manifest
