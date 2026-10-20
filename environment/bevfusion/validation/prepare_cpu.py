"""Prepare real official val infos and exercise the official CPU evaluator.

Inputs are original full metadata plus an already audited manifest. Metadata
records are filtered by scene membership without editing any retained record.
The official converter and evaluator are called unchanged. No GPU required.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

OUT = Path('/workspace/results/validation-baseline-preparation')
SOURCE = Path('/workspace/datasets/nuscenes/v1.0-trainval')
VIEW = Path('/workspace/datasets/nuscenes-validation')


def tick(stage, **values):
    record = dict(stage=stage, time=time.time(), **values)
    temporary = OUT / 'prepare-progress.tmp'
    temporary.write_text(json.dumps(record))
    temporary.replace(OUT / 'prepare-progress.json')
    print(json.dumps(record), flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def subset_metadata(manifest):
    import ijson
    tokens = {r['token'] for r in manifest['samples']}
    scenes = {r['scene_token'] for r in manifest['samples']}
    table_dir = VIEW / 'v1.0-trainval'
    table_dir.mkdir(parents=True, exist_ok=True)
    previous = OUT / 'validation-metadata-view.json'
    if previous.exists():
        record = json.loads(previous.read_text())
        if record.get('original_tables') == manifest['metadata'] and all(
                (table_dir/(name+'.json')).is_file() and sha(table_dir/(name+'.json')) == item['sha256']
                for name,item in record.get('tables', {}).items()) and len(record.get('tables', {})) == 13:
            tick('metadata_view_cache_verified')
            return record['tables']
    counts = {}
    ids = {'ego_pose': set(), 'calibrated_sensor': set(), 'instance': set(), 'log': set()}
    # Keep all sensor metadata (including Radar) for selected official scenes.
    # No Radar payload is downloaded or passed to either detector.
    for table in ['sample', 'scene', 'sample_data', 'sample_annotation', 'ego_pose',
                  'calibrated_sensor', 'instance', 'log', 'sensor', 'category', 'attribute', 'visibility', 'map']:
        tick('metadata_subset', table=table)
        destination = table_dir / (table + '.json')
        count = 0
        with (SOURCE / (table + '.json')).open('rb') as stream, destination.with_suffix('.tmp').open('w') as target:
            target.write('[')
            for row in ijson.items(stream, 'item', use_float=True):
                keep = (row['token'] in tokens if table == 'sample' else
                        row['token'] in scenes if table == 'scene' else
                        row['sample_token'] in tokens if table in ['sample_data', 'sample_annotation'] else
                        row['token'] in ids[table] if table in ids else True)
                if not keep:
                    continue
                if table == 'scene':
                    ids['log'].add(row['log_token'])
                elif table == 'sample_data':
                    ids['ego_pose'].add(row['ego_pose_token'])
                    ids['calibrated_sensor'].add(row['calibrated_sensor_token'])
                elif table == 'sample_annotation':
                    ids['instance'].add(row['instance_token'])
                if count:
                    target.write(',')
                target.write(json.dumps(row, separators=(',', ':')))
                count += 1
            target.write(']\n')
        destination.with_suffix('.tmp').replace(destination)
        counts[table] = dict(records=count, sha256=sha(destination))
    (OUT / 'validation-metadata-view.json').write_text(json.dumps(dict(
        policy='Unmodified original records filtered to official val scenes; complete original metadata remains untouched',
        original_tables=manifest['metadata'], tables=counts), indent=2))
    return counts


def main():
    import torch
    assert not torch.cuda.is_available()
    manifest = json.loads((OUT / 'validation-file-manifest.json').read_text())
    assert manifest['archive_audit_status'] == 'passed'
    subset_metadata(manifest)
    # Inventory once: hundreds of thousands of per-path network-volume stat()
    # calls would be both slow and indistinguishable from a stalled job.
    present = {str((Path(folder)/name).relative_to(VIEW)) for folder, _, names in os.walk(str(VIEW)) for name in names}
    parents = set()
    for entry in manifest['files']:
        if not entry['available_locations'] or entry['relative_path'] in present:
            continue
        dest = VIEW / entry['relative_path']
        if dest.parent not in parents:
            dest.parent.mkdir(parents=True, exist_ok=True)
            parents.add(dest.parent)
        dest.symlink_to(entry['available_locations'][0])
    tick('available_payload_links_ready')
    for source in Path('/workspace/datasets/nuscenes/maps').glob('*.png'):
        dest = VIEW / 'maps' / source.name
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            dest.symlink_to(source)
    from nuscenes import NuScenes
    from nuscenes.utils.splits import val
    tick('official_metadata_load')
    nusc = NuScenes(version='v1.0-trainval', dataroot=str(VIEW), verbose=True)
    assert len(nusc.sample) == 6019 and {s['name'] for s in nusc.scene} == set(val)
    tick('official_infos_converter')
    sys.path.insert(0, '/opt/bevfusion/tools')
    from data_converter.nuscenes_converter import _fill_trainval_infos
    class ProgressSamples(list):
        def __iter__(self):
            for index,row in enumerate(super().__iter__()):
                if index % 100 == 0:
                    tick('official_infos_converter', completed_samples=index)
                yield row
    nusc.sample = ProgressSamples(nusc.sample)
    import mmcv
    info_path = VIEW / 'nuscenes_infos_val.pkl'
    previous_report = OUT / 'official-infos-report.json'
    reused = False
    if previous_report.exists() and info_path.exists():
        old = json.loads(previous_report.read_text())
        if old.get('status') == 'PASS' and old.get('samples') == 6019 and old.get('infos_sha256') == sha(info_path):
            infos = mmcv.load(str(info_path))['infos']
            reused = True
            tick('official_infos_cache_sha256_verified')
    if not reused:
        _, infos = _fill_trainval_infos(nusc, set(), {s['token'] for s in nusc.scene}, False, max_sweeps=10)
    assert len(infos) == 6019
    infos = sorted(infos, key=lambda row: row['timestamp'])
    expected = [r['token'] for r in manifest['samples']]
    assert [r['token'] for r in infos] == expected
    manifest_samples = {r['token']: r for r in manifest['samples']}
    for row in infos:
        original = manifest_samples[row['token']]
        assert len(row['cams']) == 6
        assert {str(Path(c['data_path']).relative_to(VIEW)) for c in row['cams'].values()} == set(original['camera_files'])
        assert [str(Path(s['data_path']).relative_to(VIEW)) for s in row['sweeps']] == original['sweep_candidate_files']
    mmcv.dump(dict(infos=infos, metadata=dict(version='v1.0-trainval')), str(info_path))
    (OUT / 'official-infos-report.json').write_text(json.dumps(dict(status='PASS',
        samples=6019, version='v1.0-trainval', split='val', converter_max_sweeps=10,
        converter_source_sha256=sha('/opt/bevfusion/tools/data_converter/nuscenes_converter.py'),
        infos_path=str(info_path), infos_sha256=sha(info_path), sample_tokens=expected), indent=2))
    from nuscenes.eval.detection.config import config_factory
    evaluation_config = config_factory('detection_cvpr_2019').serialize()
    from mmcv import Config
    from torchpack.utils.config import configs
    from mmdet3d.utils import recursive_eval
    spec = json.loads(Path(__file__).parent.parent.joinpath('upstream.json').read_text())
    resolved = {}
    for model, item in spec['models'].items():
        configs.clear()
        configs.load('/opt/bevfusion/' + item['config'], recursive=True)
        cfg = Config(recursive_eval(configs))
        path = OUT / ('official-resolved-' + model + '.json')
        path.write_text(json.dumps(cfg._cfg_dict.to_dict(), indent=2))
        resolved[model] = dict(config=item['config'], resolved_config_sha256=sha(path), test_pipeline=cfg.test_pipeline)
    (OUT / 'baseline-evaluation-config.json').write_text(json.dumps(dict(
        dataset_version='v1.0-trainval', split='val', expected_samples=6019,
        evaluator='nuscenes-devkit==1.1.9/detection_cvpr_2019', settings=evaluation_config,
        models=resolved, batch_size=1, seed=0,
        coordinate_formatter='Pinned upstream NuScenesDataset._format_bbox; LiDAR gravity_center; yaw=-yaw-pi/2; LiDAR->ego range filter->global',
        caveats=['Official sweep loader defaults to random nine of ten, even in test pipeline.',
            'Previous smoke omitted unused BEV segmentation GT; full official pipeline preflight remains pending map expansion JSONs and sensor payload.',
            'Worker/rank count and cuDNN flags must be declared before full inference; do not silently claim exact official RNG parity.']), indent=2))
    # Devkit 1.1.9 rejects an entirely empty set while inferring box type. Keep
    # every official sample key and add one valid synthetic box to exercise the
    # unmodified evaluator. This is never a measured model baseline.
    for model in ['fusion', 'lidar']:
        synthetic = OUT / ('synthetic-valid-' + model + '.json')
        results = {token: [] for token in expected}
        results[expected[0]] = [dict(sample_token=expected[0],
            translation=infos[0]['ego2global_translation'], size=[1.,1.,1.], rotation=[1.,0.,0.,0.],
            velocity=[0.,0.], detection_name='car', detection_score=.01, attribute_name='vehicle.parked')]
        synthetic.write_text(json.dumps(dict(meta=dict(use_camera=model=='fusion',use_lidar=True,
            use_radar=False,use_map=False,use_external=False), results=results)))
        tick('official_evaluator_software_test', model=model)
        subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate_cpu.py')),
            '--model', model, '--predictions', str(synthetic), '--dataroot', str(VIEW),
            '--output', str(OUT / ('evaluator-selftest-' + model)), '--software-self-test'], check=True)
    tick('cpu_preparation_complete')


if __name__ == '__main__':
    main()
