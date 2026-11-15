# E2 accumulation: merge {PRED}/e2_k{K}/{tag}/pred_instances_3d/results_nusc.json
# into {PRED}/e2_k{K}_cumulative.json with dedup by token (_done_e2_k{K}.json).
import json, os, sys

PRED, tag, K = sys.argv[1], sys.argv[2], sys.argv[3]
model = f'e2_k{K}'
src = f'{PRED}/{model}/{tag}/pred_instances_3d/results_nusc.json'
cum_path = f'{PRED}/{model}_cumulative.json'
done_path = f'{PRED}/_done_e2_k{K}.json'

new = json.load(open(src))
cum = json.load(open(cum_path)) if os.path.exists(cum_path) else {'meta': new.get('meta', {}), 'results': {}}
done = set(json.load(open(done_path))) if os.path.exists(done_path) else set()
fresh = [t for t in new['results'] if t not in done]
cum['results'].update({t: new['results'][t] for t in fresh})
done.update(fresh)
json.dump(cum, open(cum_path, 'w'))
json.dump(sorted(done), open(done_path, 'w'))
print('ACCUM', model, tag, 'NEW', len(fresh), 'TOTAL', len(cum['results']))
