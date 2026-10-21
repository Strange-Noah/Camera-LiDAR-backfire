"""Single-GPU validation with unchanged transforms and resumable atomic per-token outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time
import traceback

from input_protocol import traced_dataset, POLICY
from evaluate_cpu import digest

ROOT = Path('/workspace/results/validation-baseline-preparation')
VIEW = Path('/workspace/datasets/nuscenes-validation')
PERSISTED = ROOT / 'clean-predictions'


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['fusion', 'lidar'], required=True)
    p.add_argument('--phase', choices=['pilot', 'full'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert os.geteuid() != 0 and os.environ.get('PROJECT_DATASET_WRITE_GUARD') == 'landlock'
    a.output.mkdir(parents=True, exist_ok=False)
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
    report = dict(model=a.model, phase=a.phase, pod_id=os.environ['RUNPOD_POD_ID'], status='failed', exit_code=1, started_at=time.time(), completed=0,
        input_protocol=POLICY, upstream_model_source_modified=False)
    progress = Path('/home/bfbaseline/results/progress.json')
    def tick(stage, **details):
        atomic_json(progress, dict(stage=stage, time=time.time(), model=a.model, **details))
        print(json.dumps(dict(stage=stage, model=a.model, **details)), flush=True)
    group = None
    try:
        assert torch.cuda.is_available()
        shm = os.statvfs('/dev/shm')
        assert shm.f_bavail * shm.f_frsize >= 1_000_000_000, 'Insufficient shared memory for four official loader workers'
        torch.cuda.set_device(0)
        set_random_seed(0, deterministic=False)
        # This is the flag explicitly set in the pinned official tools/test.py.
        torch.backends.cudnn.benchmark = True
        spec = json.loads(Path(__file__).parent.parent.joinpath('upstream.json').read_text())
        readiness = json.loads((ROOT/'baseline-preflight-report.json').read_text())
        assert readiness['status'] == 'PASS' and readiness['full_gpu_admission']
        assert Path('/opt/bevfusion/.git/HEAD').read_text().strip() == spec['commit']
        item = spec['models'][a.model]
        checkpoint = Path('/workspace/checkpoints/bevfusion') / item['file']
        configs.clear()
        configs.load('/opt/bevfusion/'+item['config'], recursive=True)
        cfg = Config(recursive_eval(configs))
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        if cfg.model.encoders.get('camera'):
            cfg.model.encoders.camera.backbone.init_cfg = None
        ds = cfg.data.test
        ds.dataset_root = str(VIEW)+'/'
        ds.ann_file = str(VIEW/'nuscenes_infos_val.pkl')
        ds.test_mode = True
        for stage in ds.pipeline:
            if stage['type'] == 'LoadBEVSegmentation':
                stage['dataset_root'] = str(VIEW)+'/'
        config_path = a.output/'resolved_config.py'
        config_path.write_text(cfg.pretty_text)
        dataset = build_dataset(ds)
        assert len(dataset) == 6019
        identity = dict(model=a.model, checkpoint_sha256=digest(checkpoint), upstream_commit=spec['commit'],
            resolved_config_sha256=digest(config_path), infos_sha256=digest(ds.ann_file),
            manifest_sha256=digest(ROOT/'validation-file-manifest.json'),
            runtime_sha256=Path('/workspace/cache/bevfusion-runtime/runtime.sha256').read_text().split()[0],
            runner_sha256=digest(Path(__file__)),
            input_protocol_source_sha256=digest(Path(__file__).with_name('input_protocol.py')),
            input_protocol=POLICY, cudnn_benchmark=True, deterministic=False, batch_size=1,
            parallel='MMDataParallel, one-rank Gloo group; official transforms and fp16 retained')
        assert identity['manifest_sha256'] == readiness['manifest_sha256']
        assert identity['infos_sha256'] == readiness['official_infos_sha256']
        assert identity['checkpoint_sha256'] == readiness['checkpoint_sha256'][a.model]
        all_tokens = [info['token'] for info in dataset.data_infos]
        assert len(set(all_tokens)) == 6019
        chosen = set(all_tokens)
        if a.phase == 'pilot':
            # One middle frame per official val scene: 150 scenes and lighting/motion conditions.
            chosen = set(json.loads((ROOT/'pilot-sample-tokens.json').read_text()))
            assert len(chosen) == 150
        reused = []
        for token in list(chosen):
            directory = PERSISTED/a.model/token
            marker = directory/'complete.json'
            if marker.exists():
                record = json.loads(marker.read_text())
                assert record['token'] == token and record['identity'] == identity
                assert digest(directory/'prediction.pkl') == record['raw_sha256']
                chosen.remove(token)
                reused.append(token)
        dataset.data_infos = [i for i in dataset.data_infos if i['token'] in chosen]
        if hasattr(dataset, 'flag'):
            dataset.flag = dataset.flag[:len(dataset)]
        expected = {r['token']: r for r in map(json.loads,
            (ROOT/'validation-full-preflight-fusion/input-selection.jsonl').read_text().splitlines())}
        assert len(expected) == 6019
        dataset = traced_dataset(dataset, a.output/'input-selection.jsonl', expected)
        loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=POLICY['workers_per_gpu'], dist=False, shuffle=False)
        tick('strict_model_load', requested=len(dataset), reused=len(reused))
        model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
        raw = torch.load(str(checkpoint), map_location='cpu')
        meta_classes = raw.get('meta', {}).get('CLASSES')
        assert meta_classes is None or list(meta_classes) == list(dataset.CLASSES)
        model.CLASSES = meta_classes if meta_classes is not None else dataset.CLASSES
        state = raw.get('state_dict', raw)
        state = {(k[7:] if k.startswith('module.') else k):v for k,v in state.items()}
        model.load_state_dict(state, strict=True)
        del raw, state
        report.update(identity=identity, strict_checkpoint_load=True, requested_samples=len(dataset),
            reused_tokens=reused, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
            cuda=torch.version.cuda, mmcv=mmcv.__version__)
        group = tempfile.TemporaryDirectory()
        torch.distributed.init_process_group('gloo', init_method='file://'+group.name+'/init', rank=0, world_size=1)
        if cfg.get('fp16') is not None:
            wrap_fp16_model(model)
        model = MMDataParallel(model.cuda(), device_ids=[0]).eval()
        torch.cuda.reset_peak_memory_stats()
        forward_times = []
        completed = []
        with torch.no_grad():
            for index, data in enumerate(loader):
                token = dataset.data_infos[index]['token']
                torch.cuda.synchronize()
                started = time.perf_counter()
                prediction = model(return_loss=False, rescale=True, **data)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                assert len(prediction) == 1
                result = prediction[0]
                boxes = result['boxes_3d'].tensor
                scores, labels = result['scores_3d'], result['labels_3d']
                assert boxes.device.type == scores.device.type == labels.device.type == 'cpu'
                assert len(boxes) == len(scores) == len(labels)
                assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
                assert (boxes[:,3:6] > 0).all() and ((scores >= 0) & (scores <= 1)).all()
                assert ((labels >= 0) & (labels < 10)).all()
                directory = a.output/token
                directory.mkdir()
                temporary = directory/'prediction.tmp.pkl'
                mmcv.dump(result, str(temporary))
                temporary.replace(directory/'prediction.pkl')
                atomic_json(directory/'complete.json', dict(token=token, pod_id=os.environ['RUNPOD_POD_ID'], identity=identity,
                    raw_sha256=digest(directory/'prediction.pkl'), input=expected[token],
                    forward_seconds=elapsed, boxes=len(boxes)))
                completed.append(token)
                forward_times.append(elapsed)
                report['completed'] = len(completed)
                tick('inference', completed=len(completed), requested=len(dataset), token=token)
        report.update(status='PASS', completed_tokens=completed, forward_seconds=forward_times,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        traced = [json.loads(line) for line in (a.output/'input-selection.jsonl').read_text().splitlines()]
        assert len(traced) == len(completed) and {r['token'] for r in traced} == set(completed)
        assert all(r == expected[r['token']] for r in traced)
        report['exit_code'] = 0
    except Exception as exc:
        report.update(status='FAIL', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_at=time.time(), process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(a.output/'execution.json', report)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        if group:
            group.cleanup()


if __name__ == '__main__':
    main()
