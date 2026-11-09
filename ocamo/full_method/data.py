"""Metadata-only sensor pairing, with no fabricated point firing timestamps."""
import bisect
import json
from pathlib import Path
import numpy as np
from core import rigid


def load_inventory(root):
    root = Path(root)
    folder = root/'v1.0-trainval'
    def load(name):
        return json.loads((folder/(name+'.json')).read_text())
    samples = load('sample')
    scene_of = {s['token']: s['scene_token'] for s in samples}
    scenes = sorted(set(scene_of.values()))
    if len(scenes) < 22:
        raise ValueError('Predetermined 22 scene ranks unavailable; no favorable substitution')
    populations = {'train': scenes[3:15], 'validation': scenes[15:19],
                   'evaluation_previous': scenes[:3], 'evaluation_heldout': scenes[19:22]}
    selected = set(sum(populations.values(), []))
    cal = {r['token']: r for r in load('calibrated_sensor')}
    sensors = {r['token']: r for r in load('sensor')}
    grouped = {scene: {'CAM_FRONT': [], 'LIDAR_TOP': []} for scene in selected}
    records = load('sample_data')
    needed = set()
    for r in records:
        scene = scene_of.get(r['sample_token'])
        if scene not in selected:
            continue
        channel = sensors[cal[r['calibrated_sensor_token']]['sensor_token']]['channel']
        if channel in grouped[scene]:
            grouped[scene][channel].append(r)
            needed.add(r['ego_pose_token'])
    del records
    poses = {r['token']: r for r in load('ego_pose') if r['token'] in needed}
    result = {'populations': populations, 'scenes': {}, 'schema': 1,
              'pairing_max_gap_us': 60000, 'unique_lidar_per_cadence': True,
              'point_timestamp_column': 'absent; fifth float is ring, not time'}
    for scene, channels in grouped.items():
        camera = sorted(channels['CAM_FRONT'], key=lambda r: r['timestamp'])
        lidar = sorted(channels['LIDAR_TOP'], key=lambda r: r['timestamp'])
        rows = {}
        for cadence in ['keyframes', 'higher_frequency']:
            cams = [r for r in camera if r['is_key_frame'] or cadence != 'keyframes']
            lids = [r for r in lidar if r['is_key_frame'] or cadence != 'keyframes']
            timestamps = [r['timestamp'] for r in lids]
            used = set(); paired = []; missing = 0; unmatched = 0
            for cam in cams:
                index = bisect.bisect_left(timestamps, cam['timestamp'])
                candidates = [i for i in [index-1, index] if 0 <= i < len(lids)]
                if cadence == 'keyframes':
                    candidates = [i for i in candidates if lids[i]['sample_token'] == cam['sample_token']]
                candidates = [i for i in candidates if lids[i]['token'] not in used]
                if not candidates:
                    unmatched += 1; continue
                index = min(candidates, key=lambda i: abs(timestamps[i]-cam['timestamp']))
                lid = lids[index]
                if abs(lid['timestamp']-cam['timestamp']) > 60000:
                    unmatched += 1; continue
                if not (root/cam['filename']).is_file() or not (root/lid['filename']).is_file():
                    missing += 1; continue
                used.add(lid['token'])
                wc = rigid(poses[cam['ego_pose_token']]) @ rigid(cal[cam['calibrated_sensor_token']])
                wl = rigid(poses[lid['ego_pose_token']]) @ rigid(cal[lid['calibrated_sensor_token']])
                paired.append({'camera': cam, 'lidar': lid,
                               'camera_to_lidar': (np.linalg.inv(wl)@wc).tolist(),
                               'intrinsics': cal[cam['calibrated_sensor_token']]['camera_intrinsic']})
            rows[cadence] = {'paired': paired, 'camera_records': len(cams),
                             'lidar_records': len(lids), 'missing_files': missing,
                             'unmatched': unmatched,
                             'elapsed_acquisition_seconds': ((paired[-1]['camera']['timestamp']-
                                  paired[0]['camera']['timestamp'])/1e6) if paired else 0.}
        result['scenes'][scene] = rows
    return result


def perturbation(scene, seed, sigma=.5):
    import hashlib
    derived = int.from_bytes(hashlib.sha256((scene+':CAM_FRONT:'+str(seed)).encode()).digest()[:8], 'little')
    # Identical standard-normal vector and persistent camera-local protocol to minimal.
    return np.random.default_rng(derived).normal(0, np.deg2rad(sigma), 3)
