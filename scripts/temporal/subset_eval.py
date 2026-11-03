# Same-eligibility subset eval: clean fusion vs k1 vs k2 ... on the exact same
# sample set, using the official nuScenes detection metrics.
# Usage: subset_eval.py <pred.json> [<pred.json> ...]   (one subset metric set per json)
# Common subset = intersection of the results-key sets of all input jsons
# (restricted to the 6019 val samples).
# 2026-11-03 patch: transparently reads .gz when <path>.gz exists but <path> does not;
# deletes each transient padded json right after DetectionEval loads it (disk safety).
import json, os, sys, gzip
from nuscenes import NuScenes
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

EVALROOT = '/root/autodl-tmp/evalroot'
CLEAN_FULL = '/root/autodl-tmp/preds/fusion_cumulative.json'  # only used to fetch full val tokens for padding
OUTROOT = '/root/autodl-tmp/subeval'

def resolve(p):
    if not os.path.exists(p) and os.path.exists(p + '.gz'):
        return p + '.gz'
    return p

def jload(p):
    p = resolve(p)
    if p.endswith('.gz'):
        with gzip.open(p, 'rt') as f:
            return json.load(f)
    with open(p) as f:
        return json.load(f)

paths = sys.argv[1:]
sets = []
for p in paths:
    r = jload(p)
    sets.append(set(r['results'].keys()))
common = set.intersection(*sets)
print('COMMON_SUBSET', len(common))

# Full val tokens (keys of the clean full prediction; should match GT exactly)
full_tokens = set(jload(CLEAN_FULL)['results'].keys())
cfg = config_factory('detection_cvpr_2019')
nusc = NuScenes(version='v1.0-trainval', dataroot=EVALROOT, verbose=False)

for p in paths:
    name = os.path.basename(resolve(p)).replace('.json.gz', '').replace('.json', '')
    res = jload(p)
    # Padding: fill missing tokens with empty results to satisfy the devkit's sample-set assertion
    for t in full_tokens:
        res['results'].setdefault(t, [])
    padded = f'{OUTROOT}/{name}_padded.json'
    os.makedirs(OUTROOT, exist_ok=True)
    with open(padded, 'w') as f:
        json.dump(res, f)
    ev = DetectionEval(nusc, cfg, padded, 'val', output_dir=f'{OUTROOT}/{name}', verbose=False)
    os.remove(padded)  # transient; DetectionEval has fully loaded it
    # Subset filter: restrict both GT and predictions to the common subset
    for attr in ('gt_boxes', 'pred_boxes'):
        b = getattr(ev, attr)
        b.boxes = {t: v for t, v in b.boxes.items() if t in common}
    ev.sample_tokens = [t for t in ev.sample_tokens if t in common]
    metrics, _ = ev.evaluate()
    ms = metrics.serialize()
    tp = ms['tp_errors']
    print(f'[{name}] SUBSET_N={len(common)} mAP={ms["mean_ap"]:.4f} NDS={ms["nd_score"]:.4f} '
          f'mATE={tp["trans_err"]:.4f} mASE={tp["scale_err"]:.4f} mAOE={tp["orient_err"]:.4f} '
          f'mAVE={tp["vel_err"]:.4f} mAAE={tp["attr_err"]:.4f}')
