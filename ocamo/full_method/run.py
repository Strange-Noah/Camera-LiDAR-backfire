"""Bounded Gate A driver. It has no Gate B / weighting / download mode."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from core import (PaperTracker, automatic_label, camera_edges, camera_points, classify_final,
                  exp_rotation, extract_corners, loss, quality_features, rotation_error)
from data import load_inventory, perturbation
from preselection import scores, train_scene_disjoint


PIN = '1ac55de3c0eabb02f155cd44f7481117536c9526'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare(root, obs):
    import cv2
    from scipy.spatial import KDTree
    image = cv2.imread(str(root/obs['camera']['filename']))
    if image is None:
        raise ValueError('Unreadable image')
    if image.shape[:2] != (obs['camera']['height'], obs['camera']['width']):
        raise ValueError('Image dimensions disagree with sensor metadata')
    raw = np.fromfile(root/obs['lidar']['filename'], np.float32).reshape(-1, 5)
    corners, _, feature_stats = extract_corners(raw)
    if not len(corners):
        raise ValueError('No physical scanline corners')
    edges, objective_edges = camera_edges(image)
    y, x = np.where(objective_edges)
    if len(x) < 10:
        raise ValueError('Insufficient Canny edges')
    tree = KDTree(np.column_stack([x, y]))
    lc = np.array(obs['camera_to_lidar']); k = np.array(obs['intrinsics'])
    return {'image': image, 'raw': raw, 'corners': corners, 'tree': tree,
            'edges': edges, 'lc': lc, 'k': k, 'shape': image.shape[:2],
            'feature_stats': feature_stats, 'objective_edge_count': len(x)}


def selected_training(rows):
    if len(rows) < 8:
        raise ValueError('Fewer than eight genuine keyframes in training scene')
    return [rows[i] for i in np.linspace(0, len(rows)-1, 8, dtype=int)]


def training_records(inventory, root):
    records = {}
    for population in ['train', 'validation']:
        result = []
        for scene in inventory['populations'][population]:
            rows = inventory['scenes'][scene]['keyframes']['paired']
            for obs in selected_training(rows):
                frame = prepare(root, obs)
                objective = lambda theta: loss(frame['tree'], frame['corners'], frame['lc'],
                                               frame['k'], frame['shape'], theta)
                label, minima = automatic_label(objective)
                features = quality_features(frame['corners'], frame['lc'], frame['k'], frame['edges'])
                result.append({'scene': scene, 'camera_token': obs['camera']['token'],
                               'features': features.tolist(), 'label': label, 'minima_radians': minima})
            print('LABELED', population, scene, len(result), flush=True)
        records[population] = result
    return records


class MinimalReference:
    """Replay frozen minimal feature/objective/SGDSchaul conventions, no edits."""
    def __init__(self, baseline, upstream):
        frozen = baseline/'ocamo/tools/nuscenes_minimal.py'
        # Hash of the immutable d8b86b5 source; also supports a file-only remote
        # execution package without duplicating the repository's entire history.
        expected = '4c851022ea71b06a50857ce26f32b15ff7d7a6188b29ffb89ec896f928225904'
        if expected != hashlib.sha256(frozen.read_bytes()).hexdigest():
            raise ValueError('Frozen minimal source differs')
        spec = importlib.util.spec_from_file_location('frozen_minimal', frozen)
        self.module = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)
        sys.path.insert(0, str(upstream/'python/inference'))
        from include.feature_extraction import get_corners
        import include.SGDSchaul as optimizer
        self.corners = get_corners; self.optimizer = optimizer

    def frame(self, frame, corrupt):
        import cv2
        from scipy.spatial import KDTree
        image = cv2.cvtColor(frame['image'], cv2.COLOR_BGR2RGB)
        y, x = np.where(cv2.Canny(image, 50, 100))
        keep = y > image.shape[0]/3
        tree = KDTree(np.column_stack([x[keep], y[keep]]))
        corners, _ = self.module.features(frame['raw'].astype(float), self.corners)
        centered = corners-corrupt[:3, 3]
        rc = corrupt[:3, :3]; k = frame['k']; height, width = frame['shape']
        from scipy.spatial.transform import Rotation
        def objective(kdtree, pcl, params, sigma, neighbors):
            pc = rc.T@Rotation.from_rotvec(params[:3]).as_matrix()@pcl[:, :3].T
            uv = k@pc; uv = uv[:2, uv[2] > 0]/uv[2, uv[2] > 0]
            inside = (uv[0] >= 0) & (uv[0] < width) & (uv[1] >= 0) & (uv[1] < height)
            distance, _ = kdtree.query(uv[:, inside].T, k=neighbors)
            return -np.exp(-distance**2/(2*sigma**2)).sum()
        return centered, tree, objective


def evaluate(inventory, root, model, baseline, upstream, out):
    reference = MinimalReference(baseline, upstream)
    summaries = []; eval_pop = inventory['populations']['evaluation_previous']+inventory['populations']['evaluation_heldout']
    with (out/'trajectories.jsonl').open('x') as stream:
        for cadence in ['keyframes', 'higher_frequency']:
            for scene in eval_pop:
                rows = inventory['scenes'][scene][cadence]['paired']
                if not rows:
                    raise ValueError('Missing predetermined evaluation sequence')
                if cadence == 'higher_frequency' and [r['camera']['token'] for r in rows] == [
                        r['camera']['token'] for r in inventory['scenes'][scene]['keyframes']['paired']]:
                    summaries.append({'scene': scene, 'cadence': cadence,
                                      'status': 'UNAVAILABLE_NO_ADDITIONAL_OBSERVATIONS'})
                    continue
                states = {}; histories = {}
                for seed in [0, 1]:
                    omega = perturbation(scene, seed); q = exp_rotation(omega)
                    for method in ['minimal', 'full_no_preselection', 'full_preselection']:
                        tracker = reference.optimizer.SGDSchaul(np.array([0, 1, 2])) if method == 'minimal' else PaperTracker()
                        states[(seed, method)] = {'omega': omega, 'q': q, 'tracker': tracker, 'accepted': 0, 'updates': 0}
                        histories[(seed, method)] = []
                first = rows[0]['camera']['timestamp']
                for index, obs in enumerate(rows):
                    frame = prepare(root, obs)
                    for (seed, method), state in states.items():
                        corrupt = frame['lc'].copy(); corrupt[:3, :3] = frame['lc'][:3, :3]@state['q']
                        # Preselector sees nominal available reference (corrupted), never GT residual.
                        accepted = True; score = None
                        if method == 'full_preselection':
                            x = quality_features(frame['corners'], corrupt, frame['k'], frame['edges'])
                            score = float(scores(model, x[None])[0]); accepted = score >= model['threshold']
                        tracker = state['tracker']; update = {'status': 'REJECTED'}
                        if accepted:
                            state['accepted'] += 1
                            if method == 'minimal':
                                pcl, tree, objective = reference.frame(frame, corrupt)
                                reference.optimizer.estimate_kernel_correlation = objective
                                tracker.update(pcl, tree)
                                valid = state['accepted'] > 10
                                if not np.isfinite(tracker.theta_).all():
                                    raise FloatingPointError('Frozen minimal estimator nonfinite')
                                update = {'status': 'UPDATED' if valid else 'BURN_IN'}
                            else:
                                objective = lambda theta: loss(frame['tree'], frame['corners'], corrupt,
                                                               frame['k'], frame['shape'], theta)
                                update = tracker.update(objective); valid = update['status'] == 'UPDATED'
                            state['updates'] += int(valid)
                        if method == 'minimal':
                            estimated = exp_rotation(tracker.theta_).T@corrupt[:3, :3]
                            correction = corrupt[:3, :3].T@estimated
                        else:
                            correction = exp_rotation(tracker.theta)
                            estimated = corrupt[:3, :3]@correction
                        residual = np.rad2deg(rotation_error(frame['lc'][:3, :3].T@estimated))
                        assert np.allclose(correction.T@correction, np.eye(3), atol=1e-10)
                        assert np.isclose(np.linalg.det(correction), 1, atol=1e-10)
                        record = {'scene': scene, 'cadence': cadence, 'seed': seed, 'method': method,
                                  'index': index, 'camera_token': obs['camera']['token'],
                                  'timestamp': obs['camera']['timestamp'],
                                  'elapsed_seconds': (obs['camera']['timestamp']-first)/1e6,
                                  'initial_error_degrees': float(np.rad2deg(np.linalg.norm(state['omega']))),
                                  'residual_degrees': float(residual), 'accepted': bool(accepted),
                                  'accepted_count': state['accepted'], 'valid_updates': state['updates'],
                                  'update': update, 'preselection_score': score,
                                  'right_correction': correction.tolist(),
                                  'feature_counts': frame['feature_stats'],
                                  'corners': len(frame['corners']), 'edges': frame['objective_edge_count']}
                        histories[(seed, method)].append(record)
                        stream.write(json.dumps(record, allow_nan=False)+'\n')
                    stream.flush()
                    if index % 20 == 0:
                        print('EVALUATED', cadence, scene, index, flush=True)
                for (seed, method), records in histories.items():
                    post = [r['residual_degrees'] for r in records if r['accepted_count'] > 10 and r['accepted']]
                    initial = records[0]['initial_error_degrees']; final = records[-1]['residual_degrees']
                    summaries.append({'scene': scene, 'cadence': cadence, 'seed': seed, 'method': method,
                                      'status': 'EXECUTED', 'observed': len(records),
                                      'accepted': records[-1]['accepted_count'],
                                      'rejected': len(records)-records[-1]['accepted_count'],
                                      'valid_updates': records[-1]['valid_updates'],
                                      'initial_degrees': initial, 'final_degrees': final,
                                      'post_burn_mean_degrees': float(np.mean(post)) if post else None,
                                      'elapsed_seconds': records[-1]['elapsed_seconds'],
                                      'final_improved': classify_final(initial, final, records[-1]['valid_updates']) == 'IMPROVED',
                                      'final_outcome': classify_final(initial, final, records[-1]['valid_updates']),
                                      'final_comparison_tolerance_degrees': 1e-8,
                                      'prefixes': [dict(observed=n, residual_degrees=records[n-1]['residual_degrees'],
                                                        accepted=records[n-1]['accepted_count'],
                                                        seconds=records[n-1]['elapsed_seconds'])
                                                   for n in [10, 20, 40, 80, 160, len(records)]
                                                   if n <= len(records)]})
    write_json(out/'summary.json', {'sequences': summaries, 'gate_a': 'PENDING_SCIENTIFIC_REVIEW',
                                   'gate_b': 'NOT_RUN', 'sigma_degrees_per_axis': .5})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataroot', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--inventory-only', action='store_true')
    args = p.parse_args()
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.upstream, text=True).strip() == PIN
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = load_inventory(args.dataroot)
    write_json(args.output/'inventory.json', inventory)
    if args.inventory_only:
        return
    records = training_records(inventory, args.dataroot)
    write_json(args.output/'training.json', records)
    evaluation = inventory['populations']['evaluation_previous']+inventory['populations']['evaluation_heldout']
    model = train_scene_disjoint(records['train'], records['validation'], evaluation)
    write_json(args.output/'preselection.json', model)
    evaluate(inventory, args.dataroot, model, args.baseline, args.upstream, args.output)


if __name__ == '__main__':
    main()
