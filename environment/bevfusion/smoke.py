"""Single-GPU clean inference. Run each model in its own process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time
import traceback


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, choices=['fusion', 'lidar'])
    p.add_argument('--data-view', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--limit', type=int, default=8, help='0 means all mini_val samples')
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--raw-only', action='store_true', help='Save predictions/metrics; defer plots and nuScenes formatting to CPU')
    a = p.parse_args()
    if a.limit < 0 or (a.evaluate and a.limit != 0):
        p.error('Evaluation requires the complete split: --limit 0')
    if a.raw_only and a.evaluate:
        p.error('Evaluation belongs to the CPU postprocessing phase')
    if os.geteuid() == 0:
        p.error('Run through the existing protected non-root experiment shell')
    if os.environ.get('PROJECT_DATASET_WRITE_GUARD') != 'landlock':
        p.error('Existing Landlock experiment wrapper is required')
    a.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'failed', 'model': a.model, 'completed': 0}
    started = time.perf_counter()
    try:
        import mmcv
        import torch
        from mmcv import Config
        from mmcv.parallel import MMDataParallel
        from mmcv.runner import wrap_fp16_model
        from torchpack.utils.config import configs
        from mmdet.apis import set_random_seed
        from mmdet3d.datasets import build_dataset, build_dataloader
        from mmdet3d.models import build_model
        from mmdet3d.utils import recursive_eval
        # Imports exercise extension loading; actual model inference exercises kernels.
        import mmdet3d.ops

        spec = json.loads(Path(__file__).with_name('upstream.json').read_text())
        commit = subprocess.check_output(['git', '-C', '/opt/bevfusion', 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == spec['commit'], 'Unexpected upstream revision'
        assert torch.cuda.is_available()
        torch.cuda.set_device(0)
        # Some upstream norm layers query world_size even in eval mode.
        with tempfile.TemporaryDirectory() as dist_dir:
            torch.distributed.init_process_group('nccl', init_method='file://' + dist_dir + '/init', rank=0, world_size=1)
        set_random_seed(0, deterministic=True)
        torch.backends.cudnn.benchmark = False
        model_spec = spec['models'][a.model]
        config_path = '/opt/bevfusion/' + model_spec['config']
        configs.load(config_path, recursive=True)
        cfg = Config(recursive_eval(configs))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        if cfg.model.encoders.get('camera'):
            # Full detector checkpoint supplies this backbone; avoid a third download.
            cfg.model.encoders.camera.backbone.init_cfg = None
        ds_cfg = cfg.data.test
        ds_cfg.dataset_root = str(a.data_view.resolve()) + '/'
        ds_cfg.ann_file = str(a.data_view.resolve() / 'nuscenes_infos_val.pkl')
        ds_cfg.test_mode = True
        # Detection does not consume segmentation labels. Keep all sensor transforms.
        ds_cfg.pipeline = [x for x in ds_cfg.pipeline if x['type'] != 'LoadBEVSegmentation']
        for stage in ds_cfg.pipeline:
            if stage['type'] == 'Collect3D':
                stage['keys'] = [key for key in stage['keys'] if key != 'gt_masks_bev']
        cfg.dump(str(a.output / 'resolved_config.py'))
        dataset = build_dataset(ds_cfg)
        full_count = len(dataset)
        if a.limit:
            dataset.data_infos = dataset.data_infos[:a.limit]
            if hasattr(dataset, 'flag'):
                dataset.flag = dataset.flag[:a.limit]
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        checkpoint = Path('/workspace/checkpoints/bevfusion') / model_spec['file']
        raw = torch.load(str(checkpoint), map_location='cpu')
        state = raw.get('state_dict', raw)
        state = {key.removeprefix('module.') if hasattr(key, 'removeprefix') else
                 (key[7:] if key.startswith('module.') else key): value for key, value in state.items()}
        # Strict loading fails on unexplained missing/unexpected keys and shape mismatches.
        model.load_state_dict(state, strict=True)
        del raw, state
        if cfg.get('fp16') is not None:
            wrap_fp16_model(model)
        model = MMDataParallel(model.cuda(), device_ids=[0]).eval()
        torch.cuda.reset_peak_memory_stats()
        predictions, times, records = [], [], []
        report.update(code_commit=commit, torch=torch.__version__, cuda=torch.version.cuda,
                      mmcv=mmcv.__version__, gpu=torch.cuda.get_device_name(0),
                      checkpoint_sha256=sha256(checkpoint), infos_sha256=sha256(ds_cfg.ann_file),
                      requested=len(dataset), full_split_count=full_count, batch_size=1, workers=0,
                      coordinate_frame='LiDAR at keyframe; boxes.tensor uses bottom-center origin',
                      classes=list(dataset.CLASSES), checkpoint_load='strict')
        with torch.no_grad():
            for i, data in enumerate(loader):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                result = model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - tick)
                predictions.extend(result)
                item = result[0]
                boxes = item['boxes_3d'].tensor
                scores = item['scores_3d']
                assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
                records.append(dict(token=dataset.data_infos[i]['token'], boxes=boxes.tolist(),
                                    labels=item['labels_3d'].tolist(), scores=scores.tolist()))
                report['completed'] = i + 1
        mmcv.dump(predictions, str(a.output / 'predictions.pkl'))
        (a.output / 'predictions.json').write_text(json.dumps(records))
        if not a.raw_only:
            dataset.format_results(predictions, jsonfile_prefix=str(a.output / 'nuscenes'))
        if a.evaluate:
            report['mini_metrics_not_full_validation'] = dataset.evaluate(
                predictions, metric='bbox', jsonfile_prefix=str(a.output / 'evaluation'))
        report.update(status='passed', success_rate=1.0, forward_seconds=times,
                      first_sample_seconds=times[0],
                      steady_mean_seconds=sum(times[1:]) / len(times[1:]) if len(times) > 1 else None,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(), oom=False)
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc(), oom='out of memory' in str(exc).lower())
        raise
    finally:
        report.update(wall_seconds=time.perf_counter()-started,
                      process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (a.output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
