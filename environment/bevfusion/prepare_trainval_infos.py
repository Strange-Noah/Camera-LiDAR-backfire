"""Use the pinned official converter for one selected real trainval frame on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys

assert os.geteuid() != 0 and os.environ.get('PROJECT_DATASET_WRITE_GUARD') == 'landlock'
root = Path('/workspace/datasets/nuscenes-trainval-smoke')
manifest = json.loads((root/'subset-manifest.json').read_text())
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import val, mini_train, mini_val
assert manifest['scene'] in val and manifest['scene'] not in mini_train + mini_val
nusc = NuScenes(version='v1.0-trainval', dataroot=str(root), verbose=True)
sample = nusc.get('sample', manifest['sample_token'])
class OneSample:
    sample = [sample]
    def __getattr__(self, name):
        return getattr(nusc, name)
sys.path.insert(0, '/opt/bevfusion/tools')
from data_converter.nuscenes_converter import _fill_trainval_infos
_, infos = _fill_trainval_infos(OneSample(), set(), {manifest['scene_token']}, False, max_sweeps=10)
assert len(infos) == 1 and len(infos[0]['cams']) == 6 and len(infos[0]['sweeps']) == 10
import mmcv
import numpy as np
from PIL import Image
for name in manifest['requested_files']:
    path=root/name
    assert path.is_file() and path.stat().st_size == manifest['files'][name]['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files'][name]['sha256']
    if name.endswith('.jpg'):
        with Image.open(path) as image:
            image.verify()
    else:
        points=np.fromfile(path,dtype=np.float32).reshape(-1,5)
        assert len(points)>0 and np.isfinite(points).all()
output=Path('/home/bfbaseline/trainval-infos');output.mkdir()
mmcv.dump(dict(infos=infos, metadata=dict(version='v1.0-trainval')),
          str(output/'nuscenes_infos_val.pkl'))
(output/'input-check.json').write_text(json.dumps(dict(status='passed', version='v1.0-trainval',
    scene=manifest['scene'], sample_token=sample['token'], mini_overlap=False,
    checked_files=17, camera_count=6, stored_previous_sweeps=10,
    note='Official test pipeline samples 9 previous sweeps plus current LiDAR frame'),indent=2)+'\n')
print('TRAINVAL_INFOS_AND_INPUTS_PASSED')
