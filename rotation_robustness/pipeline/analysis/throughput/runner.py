"""Isolated throughput candidate: unchanged model and frozen scientific helpers.

Benchmark qualification is required before production promotion. Outputs have an
additional execution-regime namespace; never mix them with formal cached runs.
"""
import argparse
import resource
import uuid
import json
import os
from pathlib import Path
import time
import tempfile
from environment.bevfusion.experiments.experiment_manifest import atomic_json, digest, validate_manifest
from environment.bevfusion.experiments.handoff import complete_handoff, load_predictions
from environment.bevfusion.experiments import experiment_manifest

# Runner can live in an overlay; frozen environment files live with their module.
BASE=Path(experiment_manifest.__file__).resolve().parents[1]
from .pipeline import PreparedDataset, ordered_loader, array_hash, tensor_array, restore_forward_rng
from .writer import RunCache, OrderedWriter
from .provenance import execution_regime, bind_regime, capture_inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--execute-gpu',action='store_true')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--prefetch',type=int,default=2)
    p.add_argument('--writer-depth',type=int,default=2);p.add_argument('--capture-inputs',action='store_true')
    a=p.parse_args()
    if a.workers not in (0,2,4) or a.prefetch<1 or a.writer_depth<0:raise ValueError('Invalid bounded loader/writer settings')
    invocation_start=time.perf_counter()
    if not a.execute_gpu: raise SystemExit('GPU execution requires explicit --execute-gpu')
    if os.geteuid()==0 or os.environ.get('PROJECT_DATASET_WRITE_GUARD')!='landlock': raise RuntimeError('Restricted readonly workload required')
    manifest=json.loads(a.manifest.read_text());body=validate_manifest(manifest)
    if digest(a.config)!=body['config_sha256']: raise ValueError('Config hash mismatch')
    from environment.bevfusion.experiments.source_identity import source_identity
    if source_identity()!=body['source_sha256']: raise ValueError('Experiment source hash mismatch')
    from environment.bevfusion.validation.input_protocol import POLICY
    if digest(BASE/'validation/input_protocol.py')!=body['input_protocol_sha256']: raise ValueError('Input protocol hash mismatch')
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import wrap_fp16_model
    import mmcv
    import numpy as np
    import torch
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    root=Path('/workspace/results/validation-baseline-preparation'); view=Path('/workspace/datasets/nuscenes-validation')
    spec=json.loads((BASE/'upstream.json').read_text())
    if Path('/opt/bevfusion/.git/HEAD').read_text().strip()!=spec['commit']: raise ValueError('Upstream mismatch')
    if digest(root/'validation-file-manifest.json')!=body['dataset_sha256']: raise ValueError('Dataset manifest mismatch')
    checkpoint=Path('/workspace/checkpoints/bevfusion/bevfusion-det.pth')
    if digest(checkpoint)!=body['checkpoint_sha256']: raise ValueError('Checkpoint mismatch')
    if Path('/workspace/cache/bevfusion-runtime/runtime.sha256').read_text().split()[0]!=body['runtime_sha256']: raise ValueError('Runtime identity mismatch')
    if torch.__version__!='1.10.1+cu113' or mmcv.__version__!='1.4.0': raise ValueError('Software version mismatch')
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    shm=os.statvfs('/dev/shm')
    if shm.f_bavail*shm.f_frsize<1_000_000_000: raise RuntimeError('Insufficient shared memory')
    torch.manual_seed(0)
    cfg=Config.fromfile(str(a.config)); ds=cfg.data.test
    if ds.ann_file!=str(view/'nuscenes_infos_val.pkl'): raise ValueError('Unexpected infos path')
    readiness=json.loads((root/'baseline-preflight-report.json').read_text())
    if digest(ds.ann_file)!=readiness['official_infos_sha256']: raise ValueError('Infos hash mismatch')
    dataset=build_dataset(ds)
    tokens=body['subset']['sample_tokens']; by_token={i['token']:i for i in dataset.data_infos}
    if not set(tokens)<=set(by_token): raise ValueError('Subset missing')
    dataset.data_infos=[by_token[t] for t in tokens]
    if hasattr(dataset,'flag'): dataset.flag=np.zeros(len(tokens),dtype=np.uint8)
    expected={r['token']:r for r in map(json.loads,(root/'validation-full-preflight-fusion/input-selection.jsonl').read_text().splitlines())}
    regime=execution_regime(a.workers,a.prefetch,a.writer_depth,a.capture_inputs)
    effective_root=a.output_root/'throughput-v1'/regime['sha256']
    bind_regime(effective_root,regime)
    cache=RunCache(effective_root,manifest,regime['sha256']);cid=cache.cid;out=cache.out
    if (out/'completion.json').exists():
        load_predictions(out,cid);return
    # Worker append files are private per PID/attempt; canonical traces use returned
    # token-bearing envelopes, not callbacks trapped in forked worker memory.
    evidence=out/'worker-evidence'/uuid.uuid4().hex
    prepared=PreparedDataset(dataset,body,{t:expected[t] for t in tokens},evidence)
    loader=ordered_loader(prepared,a.workers,a.prefetch)
    iterator=iter(loader)  # Start CPU-only workers before model construction/CUDA.
    init=None
    try:
        cfg.model.pretrained=None;cfg.model.train_cfg=None
        if cfg.model.encoders.get('camera'): cfg.model.encoders.camera.backbone.init_cfg=None
        model=build_model(cfg.model,test_cfg=cfg.get('test_cfg'))
        raw=torch.load(str(checkpoint),map_location='cpu');state=raw.get('state_dict',raw)
        state={(k[7:] if k.startswith('module.') else k):v for k,v in state.items()}
        classes=raw.get('meta',{}).get('CLASSES')
        if classes is not None and list(classes)!=list(dataset.CLASSES): raise ValueError('Checkpoint classes mismatch')
        model.load_state_dict(state,strict=True);model.CLASSES=dataset.CLASSES
        del raw,state
        torch.cuda.set_device(0);torch.backends.cudnn.benchmark=True
        init=tempfile.TemporaryDirectory()
        torch.distributed.init_process_group('gloo',init_method='file://'+init.name+'/init',rank=0,world_size=1)
        if cfg.get('fp16') is not None: wrap_fp16_model(model)
        model=MMDataParallel(model.cuda(),device_ids=[0]).eval()
        torch.cuda.reset_peak_memory_stats()
        predictions=[];inputs=[];geometry=[];rows=[];started=time.time();loop_start=time.perf_counter()
        writer=OrderedWriter(cache,a.writer_depth)
        succeeded=False
        writer_drain_seconds=0.;handoff_seconds=0.
        try:
            for index,token in enumerate(tokens):
                tick=time.perf_counter();packet=next(iterator);wait=time.perf_counter()-tick
                if packet['index']!=index or packet['token']!=token or packet['input']!=expected[token]:
                    raise ValueError('Loader order/token/input mismatch')
                data=packet['data'];g=packet['geometry'];g['condition_id']=cid
                from mmcv.parallel import collate
                tick=time.perf_counter();batch=collate([data],samples_per_gpu=1)
                point_hash=array_hash(data['points'])
                if point_hash!=expected[token]['points_sha256']:raise ValueError('LiDAR input/IPC mismatch')
                g['image_tensor_sha256']=array_hash(data['img'])
                g['img_aug_matrix']=tensor_array(data['img_aug_matrix']).tolist()
                g['lidar_aug_matrix']=tensor_array(data['lidar_aug_matrix']).tolist()
                parent_prepare=time.perf_counter()-tick
                # Never use an extra scatter/replay/clone to execute the model.
                # Descriptor extraction is optional benchmark evidence only.
                capture_seconds=0.
                if a.capture_inputs:
                    tick=time.perf_counter();capture_inputs(out/'benchmark-inputs.jsonl',token,batch)
                    capture_seconds=time.perf_counter()-tick
                restore_forward_rng(packet)
                tick=time.perf_counter();result=cache.load(index,token);cache_load=time.perf_counter()-tick
                reused=result is not None;forward=0.
                if not reused:
                    tick=time.perf_counter()
                    with torch.no_grad():result=model(return_loss=False,rescale=True,**batch)[0]
                    forward=time.perf_counter()-tick  # Includes H2D, kernels and upstream blocking CPU outputs.
                    boxes,scores,labels=result['boxes_3d'].tensor,result['scores_3d'],result['labels_3d']
                    if not all(x.device.type=='cpu' for x in (boxes,scores,labels)):
                        raise ValueError('Writer accepts CPU results only')
                    if not (len(boxes)==len(scores)==len(labels) and torch.isfinite(boxes).all() and torch.isfinite(scores).all()
                            and (boxes[:,3:6]>0).all() and ((scores>=0)&(scores<=1)).all() and ((labels>=0)&(labels<10)).all()):
                        raise ValueError('Invalid prediction tensors')
                tick=time.perf_counter()
                if array_hash(data['points'])!=point_hash:raise ValueError('Model mutated LiDAR tensor')
                mutation_seconds=time.perf_counter()-tick
                write=writer.submit(index,token,result,reused)
                predictions.append(result);inputs.append(dict(packet['input'],condition_id=cid));geometry.append(g)
                rows.append(dict(index=index,token=token,loader_wait_seconds=wait,
                    worker_prepare_seconds=packet['prepare_seconds'],parent_prepare_seconds=parent_prepare,
                    forward_seconds=forward,cache_load_seconds=cache_load,mutation_check_seconds=mutation_seconds,
                    input_capture_seconds=capture_seconds,reused=reused,**write))
            drain_start=time.perf_counter()
            writer.finish()  # Completion cannot overtake a pending/failed fsync.
            writer_drain_seconds=time.perf_counter()-drain_start
            loop_seconds=time.perf_counter()-loop_start
            handoff_start=time.perf_counter()
            complete_handoff(out,manifest,predictions,tokens,inputs,geometry,
                dict(status='PASS',condition_id=cid,model='fusion',samples=len(tokens),started_at=started,finished_at=time.time(),
                    worker_count=a.workers,baseline_worker_count=4,input_protocol=POLICY,gpu=torch.cuda.get_device_name(0),
                    execution_regime_sha256=regime['sha256'],scientific_admission='THROUGHPUT_CANDIDATE_NOT_QUALIFIED'))
            handoff_seconds=time.perf_counter()-handoff_start
            succeeded=True
        finally:
            try:
                writer.finish()
            finally:
                atomic_json(out/'throughput-profile.json',dict(status='PASS' if succeeded else 'FAILED',
                    execution_regime=regime,frames_observed=len(rows),frames_fresh=sum(not r['reused'] for r in rows),
                    loop_seconds=time.perf_counter()-loop_start if not succeeded else loop_seconds,
                    invocation_seconds=time.perf_counter()-invocation_start,rows=rows,writer=writer.times,
                    phases=dict(setup_seconds=loop_start-invocation_start,
                        writer_drain_seconds=writer_drain_seconds,handoff_seconds=handoff_seconds,
                        forward_seconds_sum=sum(r['forward_seconds'] for r in rows),
                        serialization_seconds_sum=sum(r['serialization_seconds'] for r in writer.times),
                        writer_commit_seconds_sum=sum(r['commit_seconds'] for r in writer.times),
                        writer_queue_wait_seconds_sum=sum(r['writer_queue_wait_seconds'] for r in rows),
                        interpretation='Thread/loader timings overlap; summed writer time is not additive wall time.'),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    main_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    finally:
        # Cover setup failures as well as forward/writer/profile failures.
        if hasattr(iterator,'_shutdown_workers'):iterator._shutdown_workers()
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
        if init is not None:init.cleanup()



if __name__=='__main__': main()
