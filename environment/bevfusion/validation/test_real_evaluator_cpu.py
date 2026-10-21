"""Full official evaluator software acceptance using a real saved smoke prediction."""
import json
from pathlib import Path
from evaluate_cpu import evaluate, digest

root = Path('/workspace/results/validation-baseline-preparation')
source = root / 'formatter-selftest/formatted/results_nusc.json'
fixture = json.loads(source.read_text())
manifest = json.loads((root / 'validation-file-manifest.json').read_text())
results = {sample['token']: [] for sample in manifest['samples']}
assert set(fixture['results']).issubset(results)
assert any(fixture['results'].values())
results.update(fixture['results'])
fixture['results'] = results
out = root / 'real-prediction-evaluator-selftest'
out.mkdir(exist_ok=True)
prediction = out / 'software-test-predictions.json'
prediction.write_text(json.dumps(fixture))
previous = out / 'report.json'
infos = Path('/workspace/datasets/nuscenes-validation/nuscenes_infos_val.pkl')
if previous.exists() and (out/'metrics_summary.json').exists():
    cache = json.loads(previous.read_text())
    if (cache.get('status') == 'PASS' and cache.get('source_sha256') == digest(source)
            and cache.get('formatted_sha256') == digest(prediction)
            and cache.get('infos_sha256') == digest(infos)
            and cache.get('metrics_sha256') == digest(out/'metrics_summary.json')):
        print('REAL_PREDICTION_EVALUATOR_SOFTWARE_TEST_CACHE_VERIFIED',flush=True)
        raise SystemExit(0)
metrics = evaluate(prediction, Path('/workspace/datasets/nuscenes-validation'), out)
for field in ['mean_ap', 'nd_score', 'tp_errors', 'label_aps', 'label_tp_errors']:
    assert field in metrics
(out / 'report.json').write_text(json.dumps(dict(status='PASS', purpose='SOFTWARE_SELF_TEST_ONLY',
    sample_tokens=len(results), nonempty_samples=sum(bool(v) for v in results.values()),
    real_prediction_source=str(source), source_sha256=digest(source),
    infos_sha256=digest(infos),
    formatted_sha256=digest(prediction), metrics_sha256=digest(out / 'metrics_summary.json'),
    baseline_accuracy_result=False), indent=2)+'\n')
print('REAL_PREDICTION_EVALUATOR_SOFTWARE_TEST_PASS', flush=True)
