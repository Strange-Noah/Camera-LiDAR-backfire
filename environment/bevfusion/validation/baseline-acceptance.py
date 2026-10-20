"""CPU-only acceptance gates. NOT_RUN is never a reproduced/failed baseline.

Input execution-report schema is documented in README.md. Checks exported global
nuScenes boxes, exact token coverage, file hashes and execution provenance.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

REFERENCES = {'fusion': {'mAP': .6852, 'NDS': .7138}, 'lidar': {'mAP': .6468, 'NDS': .6928}}
CLASSES = {'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
           'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON key: ' + key)
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_object)


def accept(manifest_path, records, tolerance=.01):
    manifest = read(manifest_path)
    expected = {row['token'] for row in manifest['samples']}
    assert len(expected) == 6019 and len(manifest['samples']) == 6019
    assert manifest['dataset_version'] == 'v1.0-trainval' and manifest['split'] == 'val'
    outcome = dict(status='NOT_RUN', official_references=REFERENCES, absolute_tolerance=tolerance, models={})
    for model in REFERENCES:
        if model not in records:
            outcome['models'][model] = dict(stage_A='NOT_RUN', stage_B='NOT_RUN')
            continue
        execution_path, evaluation_path = records[model]
        errors = []
        execution = read(execution_path)
        def check(condition, message):
            if not condition:
                errors.append(message)
        check(execution.get('model') == model, 'Model identity mismatch')
        check(execution.get('strict_checkpoint_load') is True, 'Checkpoint not strictly loaded')
        check(execution.get('exit_code') == 0, 'Process did not exit cleanly')
        check(execution.get('completed_samples') == 6019, 'Incomplete execution count')
        check(execution.get('manifest_sha256') == sha(manifest_path), 'Manifest hash mismatch')
        check(execution.get('dataset_version') == 'v1.0-trainval' and execution.get('split') == 'val', 'Wrong dataset/split')
        check(execution.get('unhandled_errors') == [], 'Unhandled execution errors')
        for key in ['checkpoint_sha256', 'upstream_commit', 'resolved_config_sha256', 'runtime_sha256', 'started_at', 'finished_at']:
            check(bool(execution.get(key)), 'Missing provenance: ' + key)
        pinned = {'fusion': 'ee0a389213922343508db40dff0ec04767b36d8f512da34b093939c37aaf2122',
                  'lidar': '7f3f1f83754752ea2091543d010a87109d724dc48512a6258c31ef6545619a0e'}
        check(execution.get('checkpoint_sha256') == pinned[model], 'Unexpected checkpoint identity')
        check(execution.get('upstream_commit') == 'db75150717a9462cb60241e36ba28d65f6908607', 'Unexpected upstream commit')
        check(execution.get('runtime_sha256') == '5d803136dbf26b68baedaf307e55ef55e9956feec590b7b8e940caf78684a3e7', 'Unexpected runtime archive identity')
        artifacts = execution.get('artifacts', {})
        check(execution.get('resolved_config_sha256') == artifacts.get('resolved_config', {}).get('sha256'), 'Resolved config identity mismatch')
        for name in ['raw_predictions', 'sample_tokens', 'resolved_config', 'nuscenes_predictions', 'raw_validation']:
            item = artifacts.get(name, {})
            path = Path(item.get('path', '/nonexistent'))
            check(path.is_file() and item.get('sha256') == sha(path) if path.is_file() else False,
                  'Missing/corrupt persisted artifact: ' + name)
        token_path = artifacts.get('sample_tokens', {}).get('path')
        if token_path and Path(token_path).is_file():
            token_data = read(token_path)
            token_list = token_data.get('sample_tokens', []) if isinstance(token_data, dict) else token_data
            check(len(token_list) == 6019 and set(token_list) == expected, 'Raw prediction token mapping incomplete/duplicated')
        validation_path = artifacts.get('raw_validation', {}).get('path')
        if validation_path and Path(validation_path).is_file():
            validation = read(validation_path)
            check(validation.get('status') == 'PASS' and validation.get('samples') == 6019, 'Raw prediction validation failed')
            for field, artifact in [('raw_predictions_sha256', 'raw_predictions'), ('sample_tokens_sha256', 'sample_tokens'), ('formatted_sha256', 'nuscenes_predictions')]:
                check(validation.get(field) == artifacts.get(artifact, {}).get('sha256'), 'Raw validation identity mismatch: ' + artifact)
        prediction_path = artifacts.get('nuscenes_predictions', {}).get('path')
        if prediction_path and Path(prediction_path).is_file():
            prediction = read(prediction_path)
            check(set(prediction.get('results', {})) == expected, 'Missing/extra sample tokens')
            for token, boxes in prediction.get('results', {}).items():
                check(len(boxes) <= 500, 'More than 500 boxes/sample')
                for box in boxes:
                    check(box.get('sample_token') == token, 'Box/sample token mismatch')
                    check(box.get('detection_name') in CLASSES, 'Unexpected class')
                    for key, size in [('translation', 3), ('size', 3), ('rotation', 4), ('velocity', 2)]:
                        values = box.get(key, [])
                        check(len(values) == size and all(isinstance(x, (int, float)) and math.isfinite(x) for x in values), 'Invalid box: ' + key)
                    check(all(x > 0 for x in box.get('size', [])), 'Nonpositive box dimensions')
                    score = box.get('detection_score')
                    check(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1, 'Invalid confidence')
                    q = box.get('rotation', [])
                    check(len(q) == 4 and abs(sum(x*x for x in q)-1) < .001, 'Invalid quaternion')
        stage_a = 'FAIL' if errors else 'PASS'
        stage_b, comparisons = 'NOT_RUN', {}
        if evaluation_path and Path(evaluation_path).is_file():
            metrics = read(evaluation_path)
            valid = metrics.get('purpose') == 'MODEL_EVALUATION' and metrics.get('model') == model
            valid = valid and metrics.get('prediction_sha256') == artifacts.get('nuscenes_predictions', {}).get('sha256')
            valid = valid and metrics.get('dataset_version') == 'v1.0-trainval' and metrics.get('split') == 'val'
            valid = valid and metrics.get('evaluator') == 'nuscenes-devkit==1.1.9/detection_cvpr_2019'
            valid = valid and set(metrics.get('per_class_AP', {})) == CLASSES
            valid = valid and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in metrics.get('per_class_AP', {}).values())
            for key in ['mAP', 'NDS', 'mATE', 'mASE', 'mAOE', 'mAVE', 'mAAE']:
                value = metrics.get(key)
                valid = valid and isinstance(value, (int, float)) and math.isfinite(value)
            for key, reference in REFERENCES[model].items():
                value = metrics.get(key)
                comparisons[key] = dict(reference=reference, measured=value,
                    absolute_difference=abs(value-reference) if isinstance(value, (int, float)) else None)
            stage_b = 'PASS' if valid else 'FAIL'
        outcome['models'][model] = dict(stage_A=stage_a, stage_B=stage_b, errors=errors,
            comparisons=comparisons, unexplained_deviations=execution.get('unexplained_deviations', ['NOT_DECLARED']))
    if all(x['stage_A'] == 'PASS' and x['stage_B'] == 'PASS' for x in outcome['models'].values()):
        reproduced = all(not x['unexplained_deviations'] and
            all(y['absolute_difference'] <= tolerance for y in x['comparisons'].values()) for x in outcome['models'].values())
        outcome['status'] = 'BASELINE_REPRODUCED' if reproduced else 'BASELINE_EXECUTED_BUT_NOT_REPRODUCED'
    elif any(x['stage_A'] == 'FAIL' or x['stage_B'] == 'FAIL' for x in outcome['models'].values()):
        outcome['status'] = 'BASELINE_FAILED'
    elif records:
        outcome['status'] = 'BASELINE_INCOMPLETE'
    return outcome


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--fusion-execution', type=Path)
    parser.add_argument('--lidar-execution', type=Path)
    parser.add_argument('--fusion-evaluation', type=Path)
    parser.add_argument('--lidar-evaluation', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    records = {model: (getattr(args, model+'_execution'), getattr(args, model+'_evaluation'))
               for model in REFERENCES if getattr(args, model+'_execution')}
    result = accept(args.manifest, records)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(result['status'])
