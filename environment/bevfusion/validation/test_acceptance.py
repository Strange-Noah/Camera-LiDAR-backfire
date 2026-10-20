"""Exercise acceptance gates with software fixtures, never model measurements."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('acceptance', Path(__file__).with_name('baseline-acceptance.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class AcceptanceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tokens = ['software-fixture-%04d' % i for i in range(6019)]
        self.manifest = self.write('manifest.json', dict(dataset_version='v1.0-trainval', split='val',
            samples=[dict(token=t) for t in self.tokens]))
        self.records = {}
        for model, reference in gate.REFERENCES.items():
            artifacts = {}
            for name, data in [('raw_predictions', {'software_fixture': True}), ('sample_tokens', self.tokens),
                    ('resolved_config', {'software_fixture': True}), ('nuscenes_predictions', {'results': {t: [] for t in self.tokens}})]:
                path = self.write(model+'-'+name+'.json', data)
                artifacts[name] = dict(path=str(path), sha256=gate.sha(path))
            validation = self.write(model+'-raw-validation.json', dict(status='PASS', samples=6019,
                raw_predictions_sha256=artifacts['raw_predictions']['sha256'],
                sample_tokens_sha256=artifacts['sample_tokens']['sha256'],
                formatted_sha256=artifacts['nuscenes_predictions']['sha256']))
            artifacts['raw_validation'] = dict(path=str(validation), sha256=gate.sha(validation))
            execution = dict(model=model, strict_checkpoint_load=True, exit_code=0, completed_samples=6019,
                manifest_sha256=gate.sha(self.manifest), dataset_version='v1.0-trainval', split='val',
                unhandled_errors=[], unexplained_deviations=[],
                checkpoint_sha256={'fusion': 'ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122',
                    'lidar': '7f3f1f83754752ea2091543d010a87109d724dc48512a6258c31ef6545619a0e'}[model],
                upstream_commit='db75150717a9462cb60241e36ba28d65f6908607', resolved_config_sha256=artifacts['resolved_config']['sha256'],
                runtime_sha256='5d803136dbf26b68baedaf307e55ef55e9956feec590b7b8e940caf78684a3e7', started_at='test-start', finished_at='test-finish', artifacts=artifacts)
            metrics = dict(purpose='MODEL_EVALUATION', model=model, prediction_sha256=artifacts['nuscenes_predictions']['sha256'],
                dataset_version='v1.0-trainval', split='val', evaluator='nuscenes-devkit==1.1.9/detection_cvpr_2019',
                mATE=.2, mASE=.2, mAOE=.2, mAVE=.2, mAAE=.2, per_class_AP={c: .5 for c in gate.CLASSES}, **reference)
            self.records[model] = (self.write(model+'-execution.json', execution), self.write(model+'-metrics.json', metrics))

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def mutate(self, model, index, update):
        path = self.records[model][index]
        data = gate.read(path)
        update(data)
        path.write_text(json.dumps(data))

    def test_unrun_is_not_success(self):
        self.assertEqual(gate.accept(self.manifest, {})['status'], 'NOT_RUN')

    def test_complete_fixture_passes(self):
        self.assertEqual(gate.accept(self.manifest, self.records)['status'], 'BASELINE_REPRODUCED')

    def test_wrong_checkpoint_rejected(self):
        self.mutate('fusion', 0, lambda x: x.update(checkpoint_sha256='wrong'))
        self.assertEqual(gate.accept(self.manifest, self.records)['status'], 'BASELINE_FAILED')

    def test_unclean_exit_rejected(self):
        self.mutate('lidar', 0, lambda x: x.update(exit_code=1))
        self.assertEqual(gate.accept(self.manifest, self.records)['status'], 'BASELINE_FAILED')

    def test_synthetic_eval_cannot_claim_accuracy(self):
        self.mutate('fusion', 1, lambda x: x.update(purpose='SOFTWARE_SELF_TEST'))
        self.assertEqual(gate.accept(self.manifest, self.records)['status'], 'BASELINE_FAILED')

    def test_accuracy_drift_requires_investigation(self):
        self.mutate('fusion', 1, lambda x: x.update(mAP=.60))
        self.assertEqual(gate.accept(self.manifest, self.records)['status'], 'BASELINE_EXECUTED_BUT_NOT_REPRODUCED')

    def test_nan_metric_rejected(self):
        self.mutate('fusion', 1, lambda x: x.update(NDS=float('nan')))
        self.assertEqual(gate.accept(self.manifest, self.records)['status'], 'BASELINE_FAILED')

    def test_unexplained_deviation_blocks_reproduction(self):
        self.mutate('fusion', 0, lambda x: x.update(unexplained_deviations=['unknown transform']))
        self.assertEqual(gate.accept(self.manifest, self.records)['status'], 'BASELINE_EXECUTED_BUT_NOT_REPRODUCED')

    def test_persisted_artifact_corruption_rejected(self):
        path = self.root / 'fusion-raw_predictions.json'
        path.write_text('corrupted')
        self.assertEqual(gate.accept(self.manifest, self.records)['status'], 'BASELINE_FAILED')

    def test_duplicate_json_key_rejected(self):
        path = self.write('duplicate.json', {})
        path.write_text('{"sample":"a","sample":"b"}')
        with self.assertRaises(ValueError):
            gate.read(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
