"""Validate checkpoint keys/shapes without allocating a GPU."""
import gc
import json
from pathlib import Path
import traceback
import torch
from mmcv import Config
from torchpack.utils.config import configs
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


spec = json.loads(Path(__file__).with_name('upstream.json').read_text())
records = []
try:
    assert not torch.cuda.is_available(), 'Run this check on the CPU preparation Pod'
    for name, item in spec['models'].items():
        configs.clear()
        configs.load('/opt/bevfusion/' + item['config'], recursive=True)
        cfg = Config(recursive_eval(configs))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        if cfg.model.encoders.get('camera'):
            cfg.model.encoders.camera.backbone.init_cfg = None
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        raw = torch.load('/workspace/checkpoints/bevfusion/' + item['file'], map_location='cpu')
        state = raw.get('state_dict', raw)
        state = {(k[7:] if k.startswith('module.') else k): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        records.append(dict(model=name, status='passed', strict=True, tensors=len(state)))
        del model, raw, state
        gc.collect()
except Exception:
    records.append(dict(status='failed', traceback=traceback.format_exc()))
    raise
finally:
    path = Path('/workspace/cache/bevfusion-runtime/model-loads-cpu.json')
    path.write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2), flush=True)
