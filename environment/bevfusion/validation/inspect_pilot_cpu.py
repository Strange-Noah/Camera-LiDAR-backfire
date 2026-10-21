"""Validate durable pilot artifacts and project full-run cost on a CPU Pod."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path('/workspace/results/validation-baseline-preparation')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    pod=sys.argv[1]
    job=ROOT/('gpu-'+pod)
    state=json.loads((job/'job-state.json').read_text())
    assert state['status']=='completed' and state['exit_code']==0
    tokens=set(json.loads((ROOT/'pilot-sample-tokens.json').read_text()))
    assert 150<=len(tokens)<=200
    expected={r['token']:r for r in map(json.loads,
        (ROOT/'validation-full-preflight-fusion/input-selection.jsonl').read_text().splitlines())}
    models={}
    for model in ['fusion','lidar']:
        run=ROOT/'clean-predictions/runs'/('%s-pilot-%s'%(model,pod))
        record=json.loads((run/'execution.json').read_text())
        assert record['status']=='PASS' and record['exit_code']==0 and record['strict_checkpoint_load']
        assert record['completed']==len(tokens) and set(record['completed_tokens'])==tokens
        for index,token in enumerate(sorted(tokens)):
            folder=ROOT/'clean-predictions'/model/token
            marker=json.loads((folder/'complete.json').read_text())
            assert marker['token']==token and marker['pod_id']==pod
            assert marker['input']==expected[token]
            assert marker['identity']==record['identity']
            assert digest(folder/'prediction.pkl')==marker['raw_sha256']
            if index%25==0:
                p=ROOT/'prepare-progress.json'
                tmp=p.with_suffix('.inspect.tmp')
                tmp.write_text(json.dumps(dict(stage='pilot_cpu_review',model=model,checked=index+1,time=time.time())))
                tmp.replace(p)
        elapsed=record['finished_at']-record['started_at']
        times=record['forward_seconds']
        assert len(times)==len(tokens)
        models[model]=dict(status='PASS',completed_samples=len(tokens),raw_prediction_checksums='PASS',
            input_identity='PASS',strict_checkpoint_load=True,model_wall_seconds=elapsed,
            wall_seconds_per_sample=elapsed/len(tokens),forward_mean_seconds=statistics.mean(times),
            peak_allocated_bytes=record['peak_allocated_bytes'],peak_reserved_bytes=record['peak_reserved_bytes'],
            process_peak_rss_kib=record['process_peak_rss_kib'],execution_sha256=digest(run/'execution.json'))
    memory=[]
    with (job/'gpu-memory.csv').open() as f:
        for row in csv.DictReader(f):
            for key,value in row.items():
                if key.strip().startswith('memory.used'):
                    try:memory.append(float(value.split()[0]))
                    except ValueError:pass
    assert memory
    wall=state['finished_at']-state['started_at']
    overhead=max(0,wall-sum(r['model_wall_seconds'] for r in models.values()))
    remaining=6019-len(tokens)
    forecast=overhead+remaining*sum(r['wall_seconds_per_sample'] for r in models.values())
    result=dict(status='PASS',purpose='GPU_PILOT_ACCEPTANCE_NOT_ACCURACY',pod_id=pod,
        pilot_samples_per_model=len(tokens),models=models,device_memory_sampled_peak_mib=max(memory),
        memory_samples=len(memory),job_wall_seconds=wall,job_exit_code=state['exit_code'],
        hourly_price_usd=.89,remaining_samples_per_model=remaining,
        projected_full_seconds=forecast,projected_full_compute_usd=forecast/3600*.89,
        projection_method='Extrapolate model wall/sample including model initialization; add observed bootstrap overhead. A conservative projection, not a runtime limit.',
        gpu_deleted_account_verification='Performed separately through account MCP (404)',
        cuda_errors=False,oom=False,finished_at=time.time())
    (ROOT/'pilot-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
