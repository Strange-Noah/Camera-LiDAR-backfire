# 修正版:results 文件实际在 <prefix>/pred_instances_3d/results_nusc.json
import json, os, sys, glob

PRED = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/preds'
models = sys.argv[2:] or ['lidar', 'fusion']

for model in models:
    cum_path = f'{PRED}/{model}_cumulative.json'
    done_path = f'{PRED}/{model}/_done.json'
    cum = json.load(open(cum_path)) if os.path.exists(cum_path) else None
    done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
    total_new = 0
    for src in sorted(glob.glob(f'{PRED}/{model}/blob*/pred_instances_3d/results_nusc.json')) + sorted(glob.glob(f'{PRED}/{model}/pass2/pred_instances_3d/results_nusc.json')) + sorted(glob.glob(f'{PRED}/{model}/pass3/pred_instances_3d/results_nusc.json')):
        new = json.load(open(src))
        res = new['results']
        fresh = [t for t in res if t not in done]
        if cum is None:
            cum = {'meta': new.get('meta', {}), 'results': {}}
        cum['results'].update({t: res[t] for t in fresh})
        done.update(fresh)
        total_new += len(fresh)
        print('MERGE', src, 'FRESH', len(fresh))
    json.dump(cum, open(cum_path, 'w'))
    json.dump(sorted(done), open(done_path, 'w'))
    print('TOTAL', model, len(cum['results']), 'NEW_THIS_RUN', total_new)
