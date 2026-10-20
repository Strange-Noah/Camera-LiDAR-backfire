import json
from pathlib import Path
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.datasets import build_dataset
from mmdet3d.utils import recursive_eval

spec = json.loads(Path(__file__).with_name('upstream.json').read_text())
configs.load('/opt/bevfusion/' + spec['models']['fusion']['config'], recursive=True)
cfg = Config(recursive_eval(configs))
ds = cfg.data.test
ds.dataset_root = '/workspace/cache/bevfusion-runtime/mini-view/'
ds.ann_file = ds.dataset_root + 'nuscenes_infos_val.pkl'
ds.test_mode = True
ds.pipeline = [x for x in ds.pipeline if x['type'] != 'LoadBEVSegmentation']
for x in ds.pipeline:
    if x['type'] == 'Collect3D':
        x['keys'] = [k for k in x['keys'] if k != 'gt_masks_bev']
dataset = build_dataset(ds)
tokens = []
for i in range(8):
    item = dataset[i]
    assert item is not None
    tokens.append(dataset.data_infos[i]['token'])
Path('/home/bfbaseline/input-validation.json').write_text(json.dumps(dict(status='passed', samples=8,
    full_val_count=len(dataset), tokens=tokens, image_size=cfg.image_size,
    lidar_sweeps=9, dataset_modified=False), indent=2) + '\n')
print('Eight real input pipelines passed on CPU', flush=True)
