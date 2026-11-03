"""Canonical condition identities, atomic records and strict cache admission."""
import hashlib
import json
import os
import math
from pathlib import Path
import tempfile
from .rotation_protocol import CHANNELS, VERSION, parameters

MATCHING = dict(version='object-match-v1', confidence_cutoff=.1, xy_distance_strict_m=2.,
                prediction_tie='original_index', gt_tie='annotation_token')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2), b''):
            h.update(b)
    return h.hexdigest()


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.pending-', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path, value):
    atomic_bytes(path, canonical(value) + b'\n')


def make_manifest(subset, sigma, seed, checkpoint_sha256, config_sha256, source_sha256,
                  dataset_sha256, input_protocol_sha256, mode='rotation'):
    tokens = subset['sample_tokens']
    if len(subset.get('scene_tokens',[]))!=len(tokens):
        raise ValueError('Missing scene identities')
    if len(tokens) != len(set(tokens)) or not tokens:
        raise ValueError('Empty/duplicate sample set')
    if mode not in ('rotation', 'clean_repeat') or (mode == 'clean_repeat' and sigma != 0):
        raise ValueError('Invalid intervention mode')
    fields = [checkpoint_sha256, config_sha256, source_sha256, dataset_sha256, input_protocol_sha256]
    if any(len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in fields):
        raise ValueError('Immutable SHA256 identities required')
    body = dict(runtime_sha256='5d803136dbf26b68baedaf307e55ef55e9956feec590b7b8e940caf78684a3e7', scientific_references=json.loads(Path(__file__).with_name('specification-lock.json').read_text()), schema_version=1, experiment=mode, intervention_version=VERSION, model='fusion',
        checkpoint_sha256=checkpoint_sha256, config_sha256=config_sha256, source_sha256=source_sha256,
        dataset_version='v1.0-trainval', split='val', dataset_sha256=dataset_sha256,
        subset=subset, subset_sha256=identity(subset), sigma_R_degrees_per_axis=float(sigma), run_seed=seed,
        cameras={c:parameters(sigma,seed,c) for c in CHANNELS}, input_protocol_sha256=input_protocol_sha256,
        matching=MATCHING.copy())
    return dict(condition=body, condition_id=identity(body))


def validate_manifest(manifest):
    body = manifest['condition']
    if identity(body) != manifest['condition_id'] or identity(body['subset']) != body['subset_sha256']:
        raise ValueError('Manifest hash mismatch')
    expected = make_manifest(body['subset'], body['sigma_R_degrees_per_axis'], body['run_seed'],
        body['checkpoint_sha256'],body['config_sha256'],body['source_sha256'],body['dataset_sha256'],
        body['input_protocol_sha256'],body['experiment'])
    # BLAS norm reduction can move the descriptive angle by one ULP across
    # macOS NumPy2 and pinned Linux NumPy1.23. Freeze z/omega/Q exactly; tolerate
    # only 1e-12 degrees in this redundant diagnostic. Canonical hash still
    # protects every original byte and conditions never share identities.
    for channel in CHANNELS:
        got=body['cameras'][channel]['angle_degrees']
        want=expected['condition']['cameras'][channel]['angle_degrees']
        if not math.isclose(got,want,rel_tol=0,abs_tol=1e-12):
            raise ValueError('Invalid actual-angle diagnostic')
        expected['condition']['cameras'][channel]['angle_degrees']=got
    expected['condition_id']=identity(expected['condition'])
    if expected != manifest:
        raise ValueError('Noncanonical or inconsistent condition parameters')
    return body


def cache_directory(root, manifest, token):
    validate_manifest(manifest)
    if token not in manifest['condition']['subset']['sample_tokens']:
        raise ValueError('Unexpected sample')
    return Path(root)/'rotation-v1'/manifest['condition_id']/token


def validate_cache(directory, manifest, token):
    validate_manifest(manifest)
    directory = Path(directory)
    try:
        r = json.loads((directory/'complete.json').read_text())
        if r['status'] != 'complete' or r['condition_id'] != manifest['condition_id'] or r['token'] != token:
            raise ValueError('Cache condition/token mismatch')
        if digest(directory/'prediction.pkl') != r['sha256']:
            raise ValueError('Corrupt prediction payload')
        return r
    except (KeyError, OSError, json.JSONDecodeError) as e:
        raise ValueError('Incomplete or invalid cache record') from e
