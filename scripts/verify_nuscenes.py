#!/usr/bin/env python3
"""Bounded pre-audit data-access checks. No statistical audit or inference."""
import argparse
import datetime as dt
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import numpy as np
from PIL import Image
from nuscenes.nuscenes import NuScenes

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--root', default=os.environ.get('NUSCENES_ROOT', '/workspace/datasets/nuscenes'))
p.add_argument('--output', default='/workspace/results/smoke_test')
p.add_argument('--render-once', action='store_true')
a = p.parse_args()
root, out = Path(a.root).resolve(), Path(a.output)
out.mkdir(parents=True, exist_ok=True)
# Remove a stale success marker so a failed rerun cannot look successful.
(out / 'PASS').unlink(missing_ok=True)
for rel in ('maps', 'samples/CAM_FRONT', 'samples/LIDAR_TOP', 'sweeps', 'v1.0-mini'):
    assert any(f.is_file() and f.stat().st_size > 0 for f in (root / rel).rglob('*')), rel
nusc = NuScenes(version='v1.0-mini', dataroot=str(root), verbose=True)
print('Scenes:', len(nusc.scene))
print('Samples:', len(nusc.sample))
print('Annotations:', len(nusc.sample_annotation))
assert len(nusc.scene) == 10
scene = nusc.scene[0]
sample = nusc.get('sample', scene['first_sample_token'])
print('Scene:', scene)
print('Sample timestamp:', sample['timestamp'])
print('Sensor channels:', sorted(sample['data']))
assert {'CAM_FRONT', 'LIDAR_TOP'} <= sample['data'].keys()

def transform(record):
    t, q = record['translation'], record['rotation']
    assert len(t) == 3 and len(q) == 4
    assert np.isfinite(t).all() and np.isfinite(q).all()
    assert abs(np.linalg.norm(q) - 1) < 1e-4
    return {'translation': t, 'rotation': q}

sensors = {}
for channel in ('CAM_FRONT', 'LIDAR_TOP'):
    sd = nusc.get('sample_data', sample['data'][channel])
    assert sd['sample_token'] == sample['token']
    path = (root / sd['filename']).resolve()
    assert path.is_relative_to(root) and path.is_file() and path.stat().st_size > 0
    cs = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])
    pose = nusc.get('ego_pose', sd['ego_pose_token'])
    sensor = nusc.get('sensor', cs['sensor_token'])
    assert sensor['channel'] == channel
    assert isinstance(sd['timestamp'], int) and isinstance(pose['timestamp'], int)
    entry = {'sample_data_token': sd['token'], 'filename': sd['filename'],
             'file_bytes': path.stat().st_size, 'timestamp_us': sd['timestamp'],
             'calibrated_sensor_token': cs['token'], 'extrinsics': transform(cs),
             'ego_pose_token': pose['token'], 'ego_pose': transform(pose),
             'ego_timestamp_us': pose['timestamp'], 'prev': sd['prev'], 'next': sd['next']}
    if channel == 'CAM_FRONT':
        K = np.asarray(cs['camera_intrinsic'])
        assert K.shape == (3, 3) and np.isfinite(K).all()
        entry['camera_intrinsic'] = K.tolist()
        with Image.open(path) as image:
            entry['image_size'] = list(image.size)
            image.verify()
    else:
        cloud = np.fromfile(path, dtype=np.float32)
        assert cloud.size > 0 and cloud.size % 5 == 0
        assert np.isfinite(cloud.reshape(-1, 5)[:, :3]).all()
        entry['point_count'] = int(cloud.size // 5)
    links = {}
    for direction, opposite in (('prev', 'next'), ('next', 'prev')):
        token = sd[direction]
        if token:
            adjacent = nusc.get('sample_data', token)
            assert adjacent[opposite] == sd['token']
            adjacent_cs = nusc.get('calibrated_sensor', adjacent['calibrated_sensor_token'])
            assert nusc.get('sensor', adjacent_cs['sensor_token'])['channel'] == channel
            assert (adjacent['timestamp'] < sd['timestamp']) if direction == 'prev' else (adjacent['timestamp'] > sd['timestamp'])
            links[direction] = {'token': token, 'timestamp_us': adjacent['timestamp'], 'is_key_frame': adjacent['is_key_frame']}
    assert links, f'{channel}: no temporal link could be resolved'
    entry['resolved_links'] = links
    sensors[channel] = entry

# One annotated object's next links, bounded to three records.
ann = next((nusc.get('sample_annotation', t) for t in sample['anns'] if nusc.get('sample_annotation', t)['next']), None)
assert ann is not None, 'First sample has no linked annotation'
instance = nusc.get('instance', ann['instance_token'])
trajectory = []
current = ann
for _ in range(3):
    required = ('translation','size','rotation','visibility_token','num_lidar_pts','instance_token','prev','next')
    assert all(k in current for k in required)
    assert current['instance_token'] == instance['token']
    assert len(current['size']) == 3 and np.all(np.asarray(current['size']) > 0)
    assert current['num_lidar_pts'] >= 0
    transform(current)
    visibility = nusc.get('visibility', current['visibility_token'])
    linked_sample = nusc.get('sample', current['sample_token'])
    assert linked_sample['scene_token'] == scene['token']
    trajectory.append({k: current[k] for k in ('token','sample_token',*required)} | {'timestamp_us': linked_sample['timestamp'], 'visibility': visibility})
    if not current['next']: break
    nxt = nusc.get('sample_annotation', current['next'])
    assert nxt['prev'] == current['token']
    current = nxt
assert len(trajectory) >= 2
assert all(b['timestamp_us'] > a['timestamp_us'] for a,b in zip(trajectory, trajectory[1:]))
first = nusc.get('sample_annotation', instance['first_annotation_token'])
last = nusc.get('sample_annotation', instance['last_annotation_token'])
assert first['instance_token'] == last['instance_token'] == instance['token']
rendered = None
if a.render_once:
    rendered = 'camera_lidar_projection.png'
    nusc.render_pointcloud_in_image(sample['token'], pointsensor_channel='LIDAR_TOP', camera_channel='CAM_FRONT', out_path=str(out / rendered))
    assert (out / rendered).stat().st_size > 0
    import matplotlib.pyplot as plt
    plt.close('all')
try:
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=Path(__file__).resolve().parents[1], text=True).strip()
except subprocess.CalledProcessError:
    commit = None
report = {
    'status': 'PASS', 'scope': 'Pre-audit bounded schema/file-access validation only',
    'time_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'git_commit': commit,
    'dataroot': str(root), 'version': 'v1.0-mini', 'python': platform.python_version(),
    'packages': {x: version(x) for x in ('nuscenes-devkit','numpy','pandas','matplotlib','pyquaternion')},
    'counts': {'scenes': len(nusc.scene), 'samples': len(nusc.sample), 'annotations': len(nusc.sample_annotation)},
    'scene': scene, 'sample_token': sample['token'], 'sample_timestamp_us': sample['timestamp'],
    'sensor_channels': sorted(sample['data']), 'sensors': sensors,
    'camera_minus_lidar_timestamp_us': sensors['CAM_FRONT']['timestamp_us'] - sensors['LIDAR_TOP']['timestamp_us'],
    'instance': instance, 'annotation_chain': trajectory, 'visualization': rendered,
    'limitations': ['Only one sample and one bounded object trajectory checked.',
                    'Counts are loading diagnostics, not a scene-level statistical audit.',
                    'No inference, corruption, temporal resampling, or suitability conclusions.']
}
(out / 'smoke_test.json').write_text(json.dumps(report, indent=2) + '\n')
(out / 'PASS').write_text(report['time_utc'] + '\n')
print(json.dumps(report, indent=2))
