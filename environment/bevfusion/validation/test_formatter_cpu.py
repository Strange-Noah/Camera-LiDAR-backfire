"""Round-trip existing real one-frame smoke predictions through CPU formatter.

This validates software/coordinate conventions, not model accuracy.
"""
import json
from pathlib import Path
import numpy as np
import mmcv
from pyquaternion import Quaternion
from evaluate_cpu import format_raw

out = Path('/workspace/results/validation-baseline-preparation/formatter-selftest')
out.mkdir(parents=True, exist_ok=True)
source = Path('/workspace/results/deployment/bevfusion-abxl6fke26bh5s/predictions/fusion-1/predictions.pkl')
infos = Path('/workspace/datasets/nuscenes-trainval-smoke/nuscenes_infos_val.pkl')
info = mmcv.load(str(infos))['infos'][0]
tokens = out / 'sample_tokens.json'
tokens.write_text(json.dumps([info['token']]))
formatted = format_raw(source, tokens, infos, 'fusion', out / 'formatted')
actual = json.loads(formatted.read_text())['results'][info['token']]
prediction = mmcv.load(str(source))[0]
raw = prediction['boxes_3d'].tensor.numpy()
labels = prediction['labels_3d'].numpy()
scores = prediction['scores_3d'].numpy()
classes = ['car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone']
ranges = dict(car=50,truck=50,construction_vehicle=50,bus=50,trailer=50,barrier=30,motorcycle=40,bicycle=40,pedestrian=40,traffic_cone=30)
expected = []
lidar_q = Quaternion(info['lidar2ego_rotation'])
global_q = Quaternion(info['ego2global_rotation'])
for box, label, score in zip(raw, labels, scores):
    center = box[:3].copy()
    center[2] += box[5] / 2
    ego = lidar_q.rotate(center) + info['lidar2ego_translation']
    if np.linalg.norm(ego[:2]) > ranges[classes[label]]:
        continue
    expected.append(dict(translation=global_q.rotate(ego)+info['ego2global_translation'],
        size=box[3:6], rotation=(global_q*lidar_q*Quaternion(axis=[0,0,1],radians=-box[6]-np.pi/2)).elements,
        velocity=global_q.rotate(lidar_q.rotate([box[7],box[8],0]))[:2],
        score=float(score), name=classes[label]))
assert len(expected) == len(actual) and actual
for a,b in zip(actual,expected):
    for key in ['translation','size','velocity']:
        np.testing.assert_allclose(a[key],b[key],rtol=1e-5,atol=1e-5)
    # Quaternion sign q/-q represents the same rotation.
    assert abs(abs(np.dot(a['rotation'],b['rotation']))-1) < 1e-5
    assert a['detection_name'] == b['name'] and abs(a['detection_score']-b['score']) < 1e-6
report=dict(status='PASS', purpose='SOFTWARE_SELF_TEST', input='Existing real formal trainval one-frame smoke',
    sample_token=info['token'], checked_boxes=len(actual), checks=['bottom_to_gravity_center','yaw_convention','lidar_to_ego_to_global',
    'ego_class_range_filter','box_dimensions','global_velocity','class_mapping','scores','raw_validation_hashes'])
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
