"""Package small evaluation evidence; large raw/formatted outputs stay durable."""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import argparse

ROOT = Path('/workspace/results/validation-baseline-preparation')


def main(job_exit_code=0):
    if job_exit_code:
        # Never ship a stale success after a failed merge/evaluation attempt.
        (ROOT/'baseline-results.json').write_text(json.dumps(dict(status='BASELINE_INCOMPLETE',
            cpu_job_exit_code=job_exit_code,reason='CPU merge/evaluation failed; preserve diagnostics and inspect coverage/logs.'))+'\n')
    result = json.loads((ROOT/'baseline-results.json').read_text())
    # A metrics mismatch is an explicit scientific outcome, not a reason to
    # discard evidence or leave the CPU Pod running.
    assert result['status'] in ['BASELINE_REPRODUCED', 'BASELINE_EXECUTED_BUT_NOT_REPRODUCED',
                                'BASELINE_FAILED', 'BASELINE_INCOMPLETE']
    names = ['baseline-results.json', 'baseline-preflight-report.json', 'gpu-cleanup-proof.json',
             'pilot-review.json', 'pilot-selection-report.json', 'gpu-compatibility-audit.json',
             'management-source-identity.json', 'evaluation-source-identity.json']
    for model in ['fusion', 'lidar']:
        names += ['baseline-'+model+'/execution-report.json',
                  'baseline-'+model+'/sample-tokens.json']
        names += ['baseline-'+model+'/evaluation/'+name for name in
                  ['evaluation-report.json', 'metrics_summary.json', 'metrics_details.json',
                   'formatted/raw-validation.json']]
    for pattern in ['gpu-*/job-state.json', 'gpu-*/*.log', 'gpu-*/gpu-memory.csv',
                    'gpu-*/nvidia-smi.txt', 'clean-predictions/runs/*/execution.json',
                    'clean-predictions/runs/*/resolved_config.py',
                    'clean-predictions/runs/*/input-selection.jsonl']:
        names += [str(p.relative_to(ROOT)) for p in ROOT.glob(pattern)]
    job = ROOT/('evaluation-'+os.environ['RUNPOD_POD_ID'])
    names += [str((job/'job.log').relative_to(ROOT))]
    final = ROOT/'final-evaluation-evidence.tar.gz'
    temporary = final.with_suffix('.tmp')
    with tarfile.open(temporary, 'w:gz') as archive:
        for name in sorted(set(names)):
            path = ROOT/name
            if path.is_file():archive.add(path, arcname=name, recursive=False)
    temporary.replace(final)
    receipt = dict(status='READY', baseline_status=result['status'],
        pod_id=os.environ['RUNPOD_POD_ID'], path=str(final), bytes=final.stat().st_size,
        sha256=hashlib.sha256(final.read_bytes()).hexdigest(), time=time.time(),
        large_predictions='Retained on Network Volume; excluded from small report packet',
        cleanup='No receipt wait; job exits after sync and Pod guard deletes after 30s')
    marker = ROOT/'final-evaluation-delivery.json'
    tmp = marker.with_suffix('.tmp')
    tmp.write_text(json.dumps(receipt, indent=2)+'\n')
    tmp.replace(marker)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--job-exit-code',type=int,default=0)
    main(parser.parse_args().job_exit_code)
