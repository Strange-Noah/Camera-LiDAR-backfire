"""Admit a GPU pilot only from complete, persisted CPU evidence and real free quota."""
import json
from pathlib import Path
import subprocess
import time
from evaluate_cpu import digest
from input_protocol import POLICY

ROOT = Path('/workspace/results/validation-baseline-preparation')


def read(name):
    return json.loads((ROOT/name).read_text())


def main():
    import torch
    assert not torch.cuda.is_available()
    manifest = read('validation-file-manifest.json')
    acquisition = read('validation-acquisition-report.json')
    infos = read('official-infos-report.json')
    assert manifest['scene_count'] == 150 and len(manifest['samples']) == 6019
    for name, item in manifest['metadata'].items():
        assert digest(item['path']) == item['sha256'], ('Original metadata changed', name)
    assert acquisition['status'] == 'PASS' and acquisition['required_files'] == 94522
    assert acquisition['verified_files'] == 94522 and acquisition['missing_files'] == acquisition['failed_files'] == 0
    assert infos['status'] == 'PASS' and infos['samples'] == 6019
    assert infos['infos_sha256'] == digest(infos['infos_path'])
    tokens = [r['token'] for r in manifest['samples']]
    scenes = {}
    for sample in manifest['samples']:
        scenes.setdefault(sample['scene_token'], []).append(sample['token'])
    pilot_tokens = [items[len(items)//2] for items in scenes.values()]
    assert len(pilot_tokens) == len(set(pilot_tokens)) == 150
    (ROOT/'pilot-sample-tokens.json').write_text(json.dumps(pilot_tokens))
    (ROOT/'validation-sample-tokens.json').write_text(json.dumps(tokens))
    pipelines = {}
    inputs = {}
    for model in ['fusion', 'lidar']:
        directory = 'validation-full-preflight-'+model
        report = read(directory+'/preflight.json')
        assert report['status'] == 'PASS' and report['available_complete_samples'] == 6019
        assert report['strict_checkpoint_load'] and report['checked_sample_tokens'] == tokens
        assert report['pipeline_transforms_removed'] == [] and report['process_group_cleanup'] == 'PASS'
        pipelines[model] = dict(status='PASS', samples=6019, strict_checkpoint_load=True,
            input_selection_sha256=digest(ROOT/directory/'input-selection.jsonl'), cpu_workers_per_gpu=report['cpu_workers_per_gpu'])
        lines = (ROOT/directory/'input-selection.jsonl').read_text().splitlines()
        inputs[model] = {r['token']:r for r in map(json.loads,lines)}
        assert len(lines) == len(inputs[model]) == 6019 and set(inputs[model]) == set(tokens)
    assert inputs['fusion'] == inputs['lidar']
    real = read('real-prediction-evaluator-selftest/report.json')
    assert real['status'] == 'PASS' and real['sample_tokens'] == 6019 and real['nonempty_samples'] > 0
    assert not real['baseline_accuracy_result']
    checkpoint_hashes = {'fusion': 'ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122',
        'lidar': '7f3f1f83754752ea2091543d010a87109d724dc48512a6258c31ef6545619a0e'}
    spec = json.loads(Path(__file__).parent.parent.joinpath('upstream.json').read_text())
    for model, item in spec['models'].items():
        assert digest('/workspace/checkpoints/bevfusion/'+item['file']) == checkpoint_hashes[model]
    assert Path('/opt/bevfusion/.git/HEAD').read_text().strip() == spec['commit']
    runtime_hash = Path('/workspace/cache/bevfusion-runtime/runtime.sha256').read_text().split()[0]
    assert runtime_hash == '5d803136dbf26b68baedaf307e55ef55e9956feec590b7b8e940caf78684a3e7'
    import mmcv, mmdet3d.ops
    tests = (ROOT/'acceptance-tests.log').read_text()
    assert 'Ran 11 tests' in tests and '\nOK\n' in tests
    used = int(subprocess.check_output(['du','-sx','--block-size=1','/workspace'], text=True).split()[0])
    assert used+5_000_000_000+3_000_000_000 <= 90_000_000_000, 'Quota cannot cover output budget and safety reserve'
    result = dict(status='PASS', full_gpu_admission=True, time=time.time(), dataset_version='v1.0-trainval',split='val',
        expected_samples=6019, scenes=150, verified_sensor_files=94522, manifest_sha256=digest(ROOT/'validation-file-manifest.json'),
        official_infos_sha256=infos['infos_sha256'], official_pipeline_checks=pipelines, checkpoint_sha256=checkpoint_hashes,
        input_protocol=POLICY, identical_lidar_inputs=True, evaluator_software_test=real,
        runtime_sha256=runtime_hash, upstream_commit=spec['commit'], torch=torch.__version__,mmcv=mmcv.__version__,
        volume_capacity_bytes=90_000_000_000, actual_volume_used_bytes=used,
        quota_remaining_bytes=90_000_000_000-used, output_allowance_bytes=3_000_000_000, safety_reserve_bytes=5_000_000_000,
        gpu_job_guard='Pod-side no-progress watchdog, exit cleanup, DELETE retries, zero GPU handoff wait',
        inference_status='NOT_RUN')
    (ROOT/'baseline-preflight-report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
