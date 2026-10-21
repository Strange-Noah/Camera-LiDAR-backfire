"""Verify per-token durable outputs, merge in official order, format/evaluate on CPU."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from evaluate_cpu import digest

ROOT = Path('/workspace/results/validation-baseline-preparation')


def main():
    import torch
    import mmcv
    assert not torch.cuda.is_available()
    p = argparse.ArgumentParser()
    p.add_argument('--cleanup-proof', type=Path, required=True)
    a = p.parse_args()
    proof = json.loads(a.cleanup_proof.read_text())
    assert proof and all(r['account_control_plane_status'] == 404 and r['compute_released'] for r in proof)
    verified_pods = {r['pod_id'] for r in proof}
    manifest_path = ROOT/'validation-file-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    tokens = [r['token'] for r in manifest['samples']]
    assert len(tokens) == len(set(tokens)) == 6019
    expected = {r['token']:r for r in map(json.loads,
        (ROOT/'validation-full-preflight-fusion/input-selection.jsonl').read_text().splitlines())}
    paths = {}
    for model in ['fusion','lidar']:
        directory = ROOT/'clean-predictions'/model
        markers = {p.parent.name:p for p in directory.glob('*/complete.json')}
        assert set(markers) == set(tokens), (model, 'Incomplete/extra raw token set')
        runs = list((ROOT/'clean-predictions/runs').glob(model+'-*/execution.json'))
        run_records = [json.loads(path.read_text()) for path in runs]
        assert run_records and all(r['pod_id'] in verified_pods for r in run_records)
        # Previous failed attempts may have contributed valid atomic samples.
        # They stay in evidence; a successful full execution must still exist.
        full_success = []
        for record in run_records:
            state = json.loads((ROOT/('gpu-'+record['pod_id'])/'job-state.json').read_text())
            if (record['phase']=='full' and record['status']=='PASS' and record['exit_code']==0
                    and state['status']=='completed' and state['exit_code']==0):
                full_success.append(record)
        assert full_success and all(r.get('strict_checkpoint_load') for r in run_records if r['completed'] > 0)
        combined = ROOT/('baseline-'+model)
        combined.mkdir(exist_ok=True)
        raw = []
        identity = None
        for index, token in enumerate(tokens):
            marker = markers[token]
            record = json.loads(marker.read_text())
            assert record['pod_id'] in verified_pods
            assert record['token'] == token and record['input'] == expected[token]
            if identity is None:
                identity = record['identity']
            assert record['identity'] == identity
            source = marker.parent/'prediction.pkl'
            assert digest(source) == record['raw_sha256']
            raw.append(mmcv.load(str(source)))
            if index % 100 == 0:
                print('CPU_MERGE',model,index+1,'/6019',flush=True)
                (ROOT/'prepare-progress.json').write_text(json.dumps(dict(stage='cpu_merge',model=model,completed=index+1,time=time.time())))
        raw_path = combined/'raw-predictions.pkl'
        temporary = combined/'raw-predictions.tmp.pkl'
        mmcv.dump(raw,str(temporary))
        temporary.replace(raw_path)
        del raw
        token_path = combined/'sample-tokens.json'
        token_path.write_text(json.dumps(tokens))
        matching_run = next(path.parent for path in runs if
            digest(path.parent/'resolved_config.py') == identity['resolved_config_sha256'])
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate_cpu.py')),
            '--model',model,'--raw-pickle','--predictions',str(raw_path),'--tokens',str(token_path),
            '--infos','/workspace/datasets/nuscenes-validation/nuscenes_infos_val.pkl',
            '--dataroot','/workspace/datasets/nuscenes-validation','--output',str(combined/'evaluation')],check=True)
        artifacts = {}
        for name,path in [('raw_predictions',raw_path),('sample_tokens',token_path),
            ('resolved_config',matching_run/'resolved_config.py'),
            ('nuscenes_predictions',combined/'evaluation/formatted/results_nusc.json'),
            ('raw_validation',combined/'evaluation/formatted/raw-validation.json')]:
            artifacts[name] = dict(path=str(path),sha256=digest(path))
        execution = dict(model=model,strict_checkpoint_load=True,exit_code=0,completed_samples=6019,
            manifest_sha256=digest(manifest_path),dataset_version='v1.0-trainval',split='val',
            unhandled_errors=[],unexplained_deviations=[],gpu_cleanup_verified=True,
            checkpoint_sha256=identity['checkpoint_sha256'],upstream_commit=identity['upstream_commit'],
            resolved_config_sha256=identity['resolved_config_sha256'],runtime_sha256=identity['runtime_sha256'],
            started_at=min(r['started_at'] for r in run_records),finished_at=max(r['finished_at'] for r in run_records),
            artifacts=artifacts,input_protocol=identity['input_protocol'],gpu_pod_ids=sorted({r['pod_id'] for r in run_records}),
            cleanup_proof_sha256=digest(a.cleanup_proof),gpu_execution_runs=[dict(path=str(path),sha256=digest(path)) for path in runs],
            declared_operational_differences=['Single rank instead of README eight ranks; four official loader workers retained.',
                'Token-derived RNG states; original random-nine-of-ten loader unchanged; LiDAR tensor hashes match across modalities.',
                'MMDataParallel with one-rank Gloo instead of MMDistributedDataParallel; fp16, model, NMS and evaluation unchanged.'])
        execution_path = combined/'execution-report.json'
        execution_path.write_text(json.dumps(execution,indent=2)+'\n')
        paths[model] = [str(execution_path),str(combined/'evaluation/evaluation-report.json')]
    command = [sys.executable,str(Path(__file__).with_name('baseline-acceptance.py')),
        '--manifest',str(manifest_path),'--output',str(ROOT/'baseline-results.json')]
    for model,pair in paths.items():
        command.extend(['--'+model+'-execution',pair[0],'--'+model+'-evaluation',pair[1]])
    subprocess.run(command,check=True)


if __name__ == '__main__':
    main()
