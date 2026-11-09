"""Six fixed real-frame checks before training; small overlays stay private."""
import argparse
import json
from pathlib import Path
import numpy as np
from core import camera_points, exp_rotation, loss, project, rotation_error
from data import load_inventory, perturbation
from run import prepare, write_json


def main():
    from scipy.spatial import KDTree
    import cv2
    p = argparse.ArgumentParser()
    p.add_argument('--dataroot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    inventory = load_inventory(a.dataroot)
    results = []
    for scene in inventory['populations']['evaluation_previous'][1:3]:
        rows = inventory['scenes'][scene]['keyframes']['paired']
        if len(rows) < 3:
            raise ValueError('Fixed prior scene unavailable')
        for index in [0, (len(rows)-1)//2, len(rows)-1]:
            frame = prepare(a.dataroot, rows[index])
            clean = frame['lc']; corrupt = clean.copy(); omega = perturbation(scene, 0)
            corrupt[:3, :3] = clean[:3, :3]@exp_rotation(omega)
            corners = frame['corners']
            np.testing.assert_allclose(camera_points(corners, corrupt, -omega),
                                       camera_points(corners, clean, np.zeros(3)), atol=1e-9)
            corrected = corrupt.copy(); corrected[:3, :3] = corrupt[:3, :3]@exp_rotation(-omega)
            independent = (np.linalg.inv(corrected)@np.column_stack([corners, np.ones(len(corners))]).T).T[:, :3]
            np.testing.assert_allclose(independent, camera_points(corners, corrupt, -omega), atol=1e-9)
            # Brute-force kNN independent of SciPy query; small fixed corner subset.
            subset = corners[np.linspace(0, len(corners)-1, min(24, len(corners)), dtype=int)]
            for theta in [np.zeros(3), -omega, -omega+np.array([.001, 0, 0])]:
                candidate = corrupt.copy(); candidate[:3, :3] = corrupt[:3, :3]@exp_rotation(theta)
                pc = (np.linalg.inv(candidate)@np.column_stack([subset, np.ones(len(subset))]).T).T[:, :3]
                uv, good = project(pc, frame['k'], frame['shape'][1], frame['shape'][0])
                brute = 0.
                for xy in uv[good]:
                    ds = np.sum((frame['tree'].data-xy)**2, axis=1)
                    brute -= np.exp(-np.partition(ds, 9)[:10]/162.).sum()
                measured = loss(frame['tree'], subset, corrupt, frame['k'], frame['shape'], theta)
                np.testing.assert_allclose(measured, brute, atol=1e-8, rtol=1e-12)
            pc = camera_points(corners, clean, np.zeros(3))
            uv, good = project(pc, frame['k'], frame['shape'][1], frame['shape'][0])
            nearest, _ = frame['tree'].query(uv[good], k=1)
            overlay = frame['image'].copy()
            for xy in uv[good].astype(int):
                cv2.circle(overlay, tuple(xy), 2, (0, 0, 255), -1)
            cv2.imwrite(str(a.output/(scene+'-'+str(index)+'-corners.jpg')), overlay)
            results.append({'scene': scene, 'index': index, 'corners': len(corners),
                            'projected': int(good.sum()), 'edge_count': frame['objective_edge_count'],
                            'feature_counts': frame['feature_stats'],
                            'radial_quantiles': np.quantile(np.linalg.norm(corners, axis=1), [0, .1, .5, .9, 1]).tolist(),
                            'clean_nearest_edge_quantiles_px': np.quantile(nearest, [0, .5, .9, 1]).tolist(),
                            'matrix_objective_checks': 'PASS', 'physical_overlay_review': 'PENDING'})
    write_json(a.output/'verification.json', {'checks': results, 'physical_review': 'PENDING',
                                             'gate_a': 'UNVERIFIED'})


if __name__ == '__main__':
    main()
