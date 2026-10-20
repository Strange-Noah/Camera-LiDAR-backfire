"""Aggregate CPU evidence without confusing preparation with model accuracy."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path('/workspace/results/validation-baseline-preparation')


def read(name):
    return json.loads((ROOT/name).read_text())


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    phase=read('cpu-phase-exit.json')
    assert phase['exit_code']==0, 'CPU phase did not pass'
    manifest=read('validation-file-manifest.json')
    acquisition=read('validation-acquisition-report.json')
    infos=read('official-infos-report.json')
    maps=read('map-acquisition-report.json')
    formatter=read('formatter-selftest/report.json')
    runtime=read('cpu-runtime-identity.json')
    partial={m:read('official-pipeline-partial-preflight-'+m+'.json') for m in ['fusion','lidar']}
    evaluator={m:read('evaluator-selftest-'+m+'/evaluation-report.json') for m in ['fusion','lidar']}
    assert manifest['archive_audit_status']=='passed' and infos['status']=='PASS' and infos['samples']==6019
    assert maps['status']=='PASS' and formatter['status']=='PASS'
    assert acquisition['failed_files']==0
    assert all(r['purpose']=='SOFTWARE_SELF_TEST' for r in evaluator.values())
    assert all(r['strict_checkpoint_load'] and r['process_group_cleanup']=='PASS' and r['pipeline_transforms_removed']==[] for r in partial.values())
    assert partial['fusion']['checked_sample_tokens']==partial['lidar']['checked_sample_tokens']
    expected={'fusion':'ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122',
              'lidar':'7f3f1f83754752ea2091543d010a87109d724dc48512a6258c31ef6545619a0e'}
    spec=json.loads(Path(__file__).parent.parent.joinpath('upstream.json').read_text())
    checkpoints={}
    for model,item in spec['models'].items():
        path=Path('/workspace/checkpoints/bevfusion')/item['file']
        actual=sha(path);assert actual==expected[model]
        checkpoints[model]=dict(**item,bytes=path.stat().st_size,sha256=actual,upstream_commit=spec['commit'])
    testlog=(ROOT/'acceptance-tests.log').read_text()
    assert 'Ran 10 tests' in testlog and '\nOK\n' in testlog
    source={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    complete_iteration=all(r['status']=='PASS' and len(r['checked_sample_tokens'])==6019 for r in partial.values())
    ready=acquisition['missing_files']==0 and complete_iteration
    report=dict(status='PASS' if ready else 'BLOCKED_MISSING_PAYLOAD' if acquisition['missing_files'] else 'BLOCKED_FULL_DATASET_ITERATION',
        full_gpu_admission=ready, cpu_phase_exit_code=0,
        dataset_version='v1.0-trainval',split='val',expected_samples=6019,
        manifest_sha256=sha(ROOT/'validation-file-manifest.json'), official_infos=infos,
        checkpoints=checkpoints,runtime=runtime,official_pipeline_checks=partial,
        acquisition_summary={k:v for k,v in acquisition.items() if k!='files'},
        official_evaluator_software_tests={m:dict(purpose=r['purpose'],status='PASS') for m,r in evaluator.items()},
        formatter_software_test=formatter,acceptance_logic_tests=dict(status='PASS',tests=10),
        source_sha256=source,completed_at=time.time(),
        remaining=['Acquire missing sensor payload after resize authorization',
            'Validate all 94,522 clean sensor files and iterate all 6019 samples on CPU',
            'Declare full-GPU runner flags/worker count and preserve original transformations',
            'Authorize representative GPU timing pilot only after full CPU admission',
            'Authorize full GPU inference separately; perform final evaluation on CPU'])
    (ROOT/'baseline-preflight-report.json').write_text(json.dumps(report,indent=2)+'\n')
    cfg=read('baseline-evaluation-config.json')
    cfg['caveats']=['Official sweep loader randomly selects nine of ten candidates; default test_mode=False is preserved.',
        'All four map expansion JSONs were verified; complete original transforms passed on the available real val samples.',
        'Full-val payload and CPU iteration remain incomplete; no GPU admission.',
        'Worker/rank count and cuDNN flags must be declared before full inference; previous smoke flags differ from official test.py.']
    cfg['partial_preflight_samples']=partial['fusion']['available_complete_samples']
    (ROOT/'baseline-evaluation-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    shutil.copyfile(str(Path(__file__).with_name('baseline-acceptance.py')),str(ROOT/'baseline-acceptance.py'))
    subprocess.run([sys.executable,str(Path(__file__).with_name('baseline-acceptance.py')),
        '--manifest',str(ROOT/'validation-file-manifest.json'),'--output',str(ROOT/'baseline-results.json')],check=True)
    used=int(subprocess.check_output(['du','-sx','--block-size=1','/workspace'],universal_newlines=True).split()[0])
    (ROOT/'volume-usage-after-preparation.json').write_text(json.dumps(dict(bytes=used,measured_at=time.time(),command='du -sx --block-size=1 /workspace'),indent=2)+'\n')
    subprocess.run([sys.executable,str(Path(__file__).with_name('storage_plan.py')),'--evidence',str(ROOT),'--existing-bytes',str(used)],check=True)
    capacities=read('capacity-calculation.json')
    text='''# Official clean baseline preparation report

Status: CPU preparation passed on available real inputs; full GPU admission is
BLOCKED_MISSING_PAYLOAD. Neither model has been run on the complete official val
split. Baseline status is NOT_RUN; no measured full-val mAP or NDS exists.

The manifest covers 150 official scenes / 6019 samples. All ten official archives
were streamed, gzip CRC/footer and exact HTTP lengths checked, and required tar
member sizes measured. No complete blob archive was saved. Original full metadata
and canonical raw data were preserved. Validation infos were generated using the
unchanged pinned official converter and checked against the manifest.

Both checkpoints match their recorded SHA256 and load strictly. The original
test pipeline, including BEV segmentation GT masks, was exercised on the same
%s available complete real val samples for each model. All process groups were
destroyed. Sensor-file verification passed on %s existing files; %s remain missing.
These missing files block full dataset iteration and GPU admission.

The official CPU evaluator was tested for both modality declarations with one
valid synthetic box and all 6019 sample keys. Entirely empty fixtures were rejected
by devkit 1.1.9; that failure was retained and corrected without modifying the
official evaluator. Fixture metrics are software-test outputs, never model accuracy.
The raw prediction formatter was independently checked on saved real smoke boxes
for gravity center, yaw, calibration transforms, velocity and class range filtering.
Acceptance logic passed 10 software tests.

The runtime remains Python 3.8 / torch 1.10.1+cu113 / mmcv-full 1.4.0 /
mmdet 2.20.0 / nuscenes-devkit 1.1.9. No GPU was rented in this preparation task.
No training, finetuning, corruption, model-source patch, threshold or NMS change
was performed. Main, teammate branches and both draft PRs remain untouched.

Capacity scenario A recommends %s GB; B recommends %s GB. The volume is still
50 GB. See validation-storage-plan.md for exact measured bytes and overlap-aware
accounting. Resize and all full-GPU execution await explicit user authorization.

GPU billing duration for this task: 0 seconds (no GPU Pod created). Historical
mini/one-frame smoke results are not full-val baselines. A credible full inference
budget still needs a representative pilot after complete CPU admission; a single
frame's forward time is not an adequate billing estimate. Final resource deletion
and ledger evidence are recorded separately after the CPU artifact handoff.
''' % (partial['fusion']['available_complete_samples'],acquisition['verified_files'],acquisition['missing_files'],capacities['A']['recommended_volume_gb'],capacities['B']['recommended_volume_gb'])
    (ROOT/'baseline-reproduction-report.md').write_text(text)
    print(json.dumps(dict(status=report['status'],existing_bytes=used,capacities=capacities),indent=2))


if __name__=='__main__':
    main()
