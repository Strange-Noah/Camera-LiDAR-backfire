"""Strict CPU checkpoint + unchanged official pipeline on available val frames.

Missing full payload is an explicit BLOCKED gate, even when the partial check
passes. Run under the existing non-root Landlock experiment wrapper.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import tempfile
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['fusion', 'lidar'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=0, help='Software probe only; zero validates all complete samples')
    p.add_argument('--workers', type=int, default=4, help='CPU data-loading workers; token RNG is independent of workers')
    args = p.parse_args()
    assert os.geteuid() != 0 and os.environ.get('PROJECT_DATASET_WRITE_GUARD') == 'landlock'
    import torch
    import mmcv
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from mmdet3d.utils import recursive_eval
    from input_protocol import traced_dataset, POLICY
    assert not torch.cuda.is_available()
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    spec = json.loads(Path(__file__).parent.parent.joinpath('upstream.json').read_text())
    manifest = json.loads(Path('/workspace/results/validation-baseline-preparation/validation-file-manifest.json').read_text())
    available = {e['relative_path'] for e in manifest['files'] if e['available_locations']}
    complete = {s['token'] for s in manifest['samples'] if
        set(s['camera_files']+[s['lidar_keyframe']]+s['sweep_candidate_files']).issubset(available)}
    assert complete, 'No complete real val frame available'
    assert args.limit >= 0
    if args.limit:
        complete = set([s['token'] for s in manifest['samples'] if s['token'] in complete][:args.limit])
    configs.clear()
    configs.load('/opt/bevfusion/'+spec['models'][args.model]['config'], recursive=True)
    cfg = Config(recursive_eval(configs))
    root = '/workspace/datasets/nuscenes-validation/'
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if cfg.model.encoders.get('camera'):
        cfg.model.encoders.camera.backbone.init_cfg = None
    ds_cfg = cfg.data.test
    ds_cfg.dataset_root = root
    ds_cfg.ann_file = root+'nuscenes_infos_val.pkl'
    ds_cfg.test_mode = True
    for stage in ds_cfg.pipeline:
        if stage['type'] == 'LoadBEVSegmentation':
            stage['dataset_root'] = root
    # No transforms/Collect keys are removed, including the unused BEV map GT.
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'resolved_config.py').write_text(cfg.pretty_text)
    dataset = build_dataset(ds_cfg)
    assert len(dataset) == 6019
    assert {r['token'] for r in dataset.data_infos} == {s['token'] for s in manifest['samples']}
    dataset.data_infos = [r for r in dataset.data_infos if r['token'] in complete]
    if hasattr(dataset, 'flag'):
        dataset.flag = dataset.flag[:len(dataset.data_infos)]
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    raw = torch.load('/workspace/checkpoints/bevfusion/'+spec['models'][args.model]['file'], map_location='cpu')
    meta_classes = raw.get('meta', {}).get('CLASSES')
    assert meta_classes is None or list(meta_classes) == list(dataset.CLASSES), 'Checkpoint class order mismatch'
    state = raw.get('state_dict', raw)
    state = {(k[7:] if k.startswith('module.') else k): v for k,v in state.items()}
    model.load_state_dict(state, strict=True)
    del raw, state, model
    gc.collect()
    trace_path = args.output/'input-selection.jsonl'
    expected = None
    if args.model == 'lidar':
        reference = args.output.parent/'validation-full-preflight-fusion'/'input-selection.jsonl'
        if reference.exists():
            expected = {r['token']:r for r in map(json.loads, reference.read_text().splitlines())}
            assert len(expected) == len(dataset)
    dataset = traced_dataset(dataset, trace_path, expected)
    done = []
    start = time.time()
    with tempfile.TemporaryDirectory() as folder:
        try:
            torch.distributed.init_process_group('gloo', init_method='file://'+folder+'/init', rank=0, world_size=1)
            loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=args.workers, dist=False, shuffle=False)
            for index, batch in enumerate(loader):
                assert batch is not None and 'gt_masks_bev' in batch
                done.append(dataset.data_infos[index]['token'])
                if index % 10 == 0:
                    print('OFFICIAL_PIPELINE_CPU', args.model, len(done), '/', len(complete), flush=True)
                    progress = args.output.parent/'results'/'progress.json'
                    progress.parent.mkdir(parents=True, exist_ok=True)
                    progress.write_text(json.dumps(dict(model=args.model,completed=len(done),time=time.time())))
        finally:
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()
    assert not torch.distributed.is_initialized() and set(done) == complete
    report = dict(status='PASS' if len(done)==6019 else 'PARTIAL_PASS_FULL_VAL_BLOCKED',
        model=args.model, strict_checkpoint_load=True, official_dataset_samples=6019,
        available_complete_samples=len(done), missing_complete_samples=6019-len(done),
        checked_sample_tokens=done, pipeline_transforms_removed=[], process_group_cleanup='PASS',
        checkpoint_init_download_disabled=True, path_relocation_only=True,
        class_names=list(dataset.CLASSES), checkpoint_meta_classes_present=meta_classes is not None,
        gpu_inference_executed=False, seconds=time.time()-start)
    report.update(input_protocol=POLICY, input_identity_comparison='PASS' if expected is not None else 'REFERENCE',
        input_selection_path=str(trace_path), cpu_workers_per_gpu=args.workers)
    (args.output/'preflight.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
