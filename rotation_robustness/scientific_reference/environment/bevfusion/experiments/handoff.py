"""Integrity-first GPU→CPU contract. Validator never unpickles predictions."""
import json
import pickle
from pathlib import Path
from .experiment_manifest import atomic_bytes, atomic_json, digest, validate_manifest
from .subset_evaluation import exact_tokens

FILES=('manifest.json','sample_tokens.json','raw_predictions.pkl','input_trace.jsonl','geometry_trace.jsonl','execution_report.json')


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def validate_handoff(directory, expected_condition=None, _candidate_completion=None, _staging=False):
    directory=Path(directory)
    if not _staging and (directory.name.endswith('.pending') or 'incomplete' in directory.parts):
        raise ValueError('Unpublished/incomplete directory')
    try:
        completion=_candidate_completion if _candidate_completion is not None else json.loads((directory/'completion.json').read_text())
        if completion['status']!='complete' or set(completion['files'])!=set(FILES): raise ValueError('Incomplete handoff')
        for name in FILES:
            if digest(directory/name)!=completion['files'][name]: raise ValueError('Corrupt '+name)
        manifest=json.loads((directory/'manifest.json').read_text());body=validate_manifest(manifest)
        cid=manifest['condition_id']
        if completion['condition_id']!=cid or (expected_condition is not None and expected_condition!=cid):
            raise ValueError('Condition mismatch')
        tokens=json.loads((directory/'sample_tokens.json').read_text())
        if tokens!=body['subset']['sample_tokens']: raise ValueError('Sample manifest mismatch')
        report=json.loads((directory/'execution_report.json').read_text())
        if report['condition_id']!=cid or report['model']!='fusion' or report['status']!='PASS' or report['samples']!=len(tokens):
            raise ValueError('Execution metadata mismatch')
        if completion['samples']!=len(tokens): raise ValueError('Completion sample count mismatch')
        for name in ('input_trace.jsonl','geometry_trace.jsonl'):
            records=read_jsonl(directory/name)
            if len(records)!=len(tokens) or {r['token'] for r in records}!=set(tokens): raise ValueError('Trace coverage mismatch')
            if any(r['condition_id']!=cid for r in records): raise ValueError('Trace identity mismatch')
            if name=='input_trace.jsonl' and any(not r.get('points_sha256') or 'sweep_paths' not in r for r in records):
                raise ValueError('Missing LiDAR input evidence')
            if name=='geometry_trace.jsonl':
                for r in records:
                    if len(r['cameras'])!=6 or {c['channel'] for c in r['cameras']}!=set(body['cameras']):
                        raise ValueError('Incomplete Camera trace')
                    if any(c['parameters']!=body['cameras'][c['channel']] for c in r['cameras']): raise ValueError('Camera parameter mismatch')
                    import numpy as np
                    from .rotation_protocol import calibrated_matrices
                    for c in r['cameras']:
                        expected=calibrated_matrices(c['clean_camera2lidar'],c['camera_intrinsics'],c['parameters'])
                        for key,matrix in zip(('camera2lidar','lidar2camera','lidar2image'),expected):
                            if not np.allclose(c[key],matrix,atol=1e-4,rtol=1e-5): raise ValueError('Inconsistent effective '+key)
        return manifest
    except (OSError,KeyError,json.JSONDecodeError) as e:
        raise ValueError('Invalid handoff') from e


def complete_handoff(directory,manifest,predictions,tokens,input_trace,geometry_trace,execution):
    directory=Path(directory);validate_manifest(manifest)
    exact_tokens({t:[] for t in tokens},manifest['condition']['subset']['sample_tokens'])
    if len(tokens)!=len(set(tokens)) or len(predictions)!=len(tokens) or tokens!=manifest['condition']['subset']['sample_tokens']:
        raise ValueError('Prediction count/order mismatch')
    # A token-bearing envelope prevents positional ambiguity after transfer.
    atomic_bytes(directory/'raw_predictions.pkl',pickle.dumps(dict(condition_id=manifest['condition_id'],
        model='fusion',sample_tokens=tokens,predictions=predictions),protocol=4))
    atomic_json(directory/'manifest.json',manifest);atomic_json(directory/'sample_tokens.json',tokens)
    for name,records in [('input_trace.jsonl',input_trace),('geometry_trace.jsonl',geometry_trace)]:
        from .experiment_manifest import canonical
        atomic_bytes(directory/name,b''.join(canonical(r)+b'\n' for r in records))
    atomic_json(directory/'execution_report.json',execution)
    completion=dict(status='complete',condition_id=manifest['condition_id'],
        samples=len(tokens),files={name:digest(directory/name) for name in FILES})
    validate_handoff(directory,manifest['condition_id'],_candidate_completion=completion)
    atomic_json(directory/'completion.json',completion)



def load_predictions(directory,expected_condition=None):
    manifest=validate_handoff(directory,expected_condition)
    # Only project-created trusted pickle files, AFTER hash/manifest validation.
    with (Path(directory)/'raw_predictions.pkl').open('rb') as f: raw=pickle.load(f)
    tokens=manifest['condition']['subset']['sample_tokens']
    if raw.get('condition_id')!=manifest['condition_id'] or raw.get('model')!='fusion' or raw.get('sample_tokens')!=tokens or len(raw.get('predictions',[]))!=len(tokens):
        raise ValueError('Corrupted prediction metadata')
    return manifest,raw['predictions']
