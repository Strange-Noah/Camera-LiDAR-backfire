"""Evaluate trusted saved predictions with the pinned upstream formatter/devkit.

Run inside the existing Python 3.8 CPU runtime. Pickle inputs must be our own
artifacts. A token sidecar is mandatory: positional alignment is never inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def evaluate(predictions, dataroot, output):
    from nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import DetectionEval
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(dataroot), verbose=False)
    evaluator = DetectionEval(nusc, config=config_factory('detection_cvpr_2019'),
        result_path=str(predictions), eval_set='val', output_dir=str(output), verbose=False)
    evaluator.main(plot_examples=0, render_curves=False)
    return json.loads((output / 'metrics_summary.json').read_text())


def format_raw(raw_path, tokens_path, infos_path, model, output):
    import mmcv
    from mmdet3d.datasets import NuScenesDataset
    from nuscenes.eval.detection.config import config_factory
    data = mmcv.load(str(infos_path))
    infos = sorted(data['infos'], key=lambda row: row['timestamp'])
    tokens = json.loads(tokens_path.read_text())
    if isinstance(tokens, dict):
        tokens = tokens['sample_tokens']
    assert tokens == [row['token'] for row in infos], 'Prediction order differs from infos'
    assert len(tokens) == len(set(tokens)), 'Duplicate prediction tokens'
    results = mmcv.load(str(raw_path))
    assert len(results) == len(tokens)
    import torch
    box_count = 0
    for prediction in results:
        assert set(['boxes_3d', 'scores_3d', 'labels_3d']).issubset(prediction)
        boxes, scores, labels = prediction['boxes_3d'].tensor, prediction['scores_3d'], prediction['labels_3d']
        assert len(boxes) == len(scores) == len(labels)
        assert torch.isfinite(boxes).all() and torch.isfinite(scores).all()
        assert (boxes[:, 3:6] > 0).all() and ((scores >= 0) & (scores <= 1)).all()
        assert ((labels >= 0) & (labels < 10)).all() and not labels.is_floating_point()
        box_count += len(boxes)
    # Bypass pipeline construction only. These are the exact fields consumed by
    # the pinned upstream _format_bbox; its coordinate/attribute code is reused.
    dataset = NuScenesDataset.__new__(NuScenesDataset)
    dataset.data_infos = infos
    dataset.CLASSES = ('car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                       'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone')
    dataset.eval_detection_configs = config_factory('detection_cvpr_2019')
    dataset.eval_version = 'detection_cvpr_2019'
    dataset.modality = dict(use_camera=model == 'fusion', use_lidar=True,
                            use_radar=False, use_map=False, use_external=False)
    formatted = Path(dataset._format_bbox(results, str(output)))
    validation = dict(status='PASS', samples=len(tokens), boxes=box_count,
        raw_predictions_sha256=digest(raw_path), sample_tokens_sha256=digest(tokens_path),
        infos_sha256=digest(infos_path), formatted_sha256=digest(formatted),
        checks=['token_order', 'raw_count', 'raw_finite_boxes_scores', 'positive_dimensions', 'class_ids_0_to_9'])
    (output / 'raw-validation.json').write_text(json.dumps(validation, indent=2))
    return formatted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--dataroot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--raw-pickle', action='store_true')
    parser.add_argument('--tokens', type=Path)
    parser.add_argument('--infos', type=Path)
    parser.add_argument('--model', choices=['fusion', 'lidar'], required=True)
    parser.add_argument('--software-self-test', action='store_true')
    args = parser.parse_args()
    import torch
    assert not torch.cuda.is_available(), 'Evaluator must run on CPU'
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.predictions
    fixture_adjustment = None
    if args.software_self_test and not args.raw_pickle:
        fixture = json.loads(source.read_text())
        if not any(fixture['results'].values()):
            # The pinned devkit infers its box type from a real entry and cannot
            # evaluate an entirely empty fixture. This branch is strictly for
            # explicitly labelled software tests, never model predictions.
            import mmcv
            infos = mmcv.load(str(args.dataroot/'nuscenes_infos_val.pkl'))['infos']
            info = min(infos, key=lambda row: row['timestamp'])
            token = info['token']
            assert token in fixture['results']
            fixture['results'][token] = [dict(sample_token=token,
                translation=info['ego2global_translation'], size=[1.,1.,1.], rotation=[1.,0.,0.,0.],
                velocity=[0.,0.], detection_name='car', detection_score=.01, attribute_name='vehicle.parked')]
            source.write_text(json.dumps(fixture))
            fixture_adjustment = 'Added one valid synthetic box for devkit 1.1.9 box-type inference; all official sample keys retained.'
    if args.raw_pickle:
        assert args.tokens and args.infos
        source = format_raw(source, args.tokens, args.infos, args.model, args.output / 'formatted')
    metrics = evaluate(source, args.dataroot, args.output)
    result = dict(model=args.model, purpose='SOFTWARE_SELF_TEST' if args.software_self_test else 'MODEL_EVALUATION',
        prediction_sha256=digest(source), selftest_fixture_adjustment=fixture_adjustment, dataset_version='v1.0-trainval', split='val',
        evaluator='nuscenes-devkit==1.1.9/detection_cvpr_2019', mAP=metrics['mean_ap'], NDS=metrics['nd_score'],
        **{name: metrics['tp_errors'][key] for name, key in
           [('mATE', 'trans_err'), ('mASE', 'scale_err'), ('mAOE', 'orient_err'), ('mAVE', 'vel_err'), ('mAAE', 'attr_err')]},
        per_class_AP={name: sum(values.values()) / len(values) for name, values in metrics['label_aps'].items()})
    (args.output / 'evaluation-report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
