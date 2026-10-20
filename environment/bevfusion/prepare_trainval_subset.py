"""CPU-only, explicit one-frame trainval acquisition without altering the mini root.

Stream the official first blob archive, retaining only selected frame inputs.
Subset metadata records are copied unchanged; this is not a full-val dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request

SOURCE = Path('/workspace/datasets/nuscenes')
ROOT = Path('/workspace/datasets/nuscenes-trainval-smoke')
URL = 'https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval01_blobs.tgz'


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def select():
    import ijson
    original = SOURCE / 'v1.0-trainval'
    out = ROOT / 'v1.0-trainval'
    out.mkdir(parents=True, exist_ok=True)
    scenes = json.loads((original / 'scene.json').read_text())
    scene = next(x for x in scenes if x['name'] == 'scene-0003')
    mini = json.loads((SOURCE / 'v1.0-mini/scene.json').read_text())
    assert scene['token'] not in {x['token'] for x in mini}
    samples = sorted([x for x in json.loads((original / 'sample.json').read_text())
                      if x['scene_token'] == scene['token']], key=lambda x: x['timestamp'])
    sample = samples[2]  # Enough LiDAR history for ten real previous sweeps.
    sample_tokens = {x['token'] for x in samples}
    def rows(name, predicate):
        with (original / (name + '.json')).open('rb') as stream:
            return [x for x in ijson.items(stream, 'item', use_float=True) if predicate(x)]
    sd = rows('sample_data', lambda x: x['sample_token'] in sample_tokens)
    annotations = rows('sample_annotation', lambda x: x['sample_token'] in sample_tokens)
    by_token = {x['token']: x for x in sd}
    sensors = {x['token']: x for x in json.loads((original / 'sensor.json').read_text())}
    calibrations = json.loads((original / 'calibrated_sensor.json').read_text())
    channels = {x['token']: sensors[x['sensor_token']]['channel'] for x in calibrations}
    keys = {channels[x['calibrated_sensor_token']]: x for x in sd
            if x['sample_token'] == sample['token'] and x['is_key_frame']}
    requested = [keys[c]['filename'] for c in
                 ['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','LIDAR_TOP']]
    sweep = keys['LIDAR_TOP']
    for _ in range(10):
        assert sweep['prev'], 'Insufficient real LiDAR history'
        sweep = by_token[sweep['prev']]
        requested.append(sweep['filename'])
    pose_tokens = {x['ego_pose_token'] for x in sd}
    calib_tokens = {x['calibrated_sensor_token'] for x in sd}
    instance_tokens = {x['instance_token'] for x in annotations}
    tables = dict(scene=[scene], sample=samples, sample_data=sd, sample_annotation=annotations,
                  calibrated_sensor=[x for x in calibrations if x['token'] in calib_tokens],
                  ego_pose=rows('ego_pose', lambda x: x['token'] in pose_tokens),
                  instance=rows('instance', lambda x: x['token'] in instance_tokens),
                  log=rows('log', lambda x: x['token'] == scene['log_token']))
    for name, items in tables.items():
        save(out / (name + '.json'), items)
    for name in ['attribute','category','visibility','sensor','map']:
        shutil.copyfile(original / (name + '.json'), out / (name + '.json'))
    maps = ROOT / 'maps'
    if not maps.exists():
        maps.symlink_to(SOURCE / 'maps', target_is_directory=True)
    assert len(set(requested)) == 17
    manifest = dict(version='v1.0-trainval', scope='one official validation frame, not mini',
                    scene=scene['name'], scene_token=scene['token'], sample_token=sample['token'],
                    sample_timestamp=sample['timestamp'], mini_overlap=False,
                    official_blob_url=URL, original_metadata=str(original),
                    original_scene_count=len(scenes), subset_scene_count=1,
                    requested_files=requested, tables={k: len(v) for k,v in tables.items()})
    previous = ROOT / 'subset-manifest.json'
    if previous.exists():
        old = json.loads(previous.read_text())
        assert old['sample_token'] == manifest['sample_token']
        assert old['requested_files'] == manifest['requested_files']
        manifest = old  # Preserve acquisition provenance on an idempotent rerun.
    save(previous, manifest)
    print(json.dumps(manifest), flush=True)


class ProgressReader:
    def __init__(self, raw):
        self.raw, self.count, self.last = raw, 0, time.monotonic()
    def read(self, size=-1):
        block = self.raw.read(size)
        self.count += len(block)
        if time.monotonic() - self.last > 20:
            print('DOWNLOAD_PROGRESS compressed_bytes=' + str(self.count), flush=True)
            self.last = time.monotonic()
        return block


def download():
    manifest = json.loads((ROOT / 'subset-manifest.json').read_text())
    needed = set(manifest['requested_files'])
    prior = manifest.get('files', {})
    progress = ROOT / 'acquired-files.json'
    if not prior and progress.exists():
        prior = json.loads(progress.read_text())
    acquired = {}
    for name in list(needed):
        dest = ROOT / name
        if dest.is_file():
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()
            expected = prior.get(name)
            if expected is None:
                raise RuntimeError('Existing file has no acquisition record: ' + name)
            assert dest.stat().st_size == expected['bytes'] and digest == expected['sha256'], 'Cached input changed: ' + name
            acquired[name] = dict(bytes=dest.stat().st_size, sha256=digest, reused=True)
            needed.remove(name)
    if needed:
        # Read compressed bytes once; never store or unpack the entire 31.6GB archive.
        request = urllib.request.Request(URL, headers={'User-Agent':'Camera-LiDAR-dataset-check/1.0'})
        with urllib.request.urlopen(request, timeout=60) as response:
            manifest['archive_content_length'] = int(response.headers['Content-Length'])
            manifest['archive_etag'] = response.headers.get('ETag')
            reader = ProgressReader(response)
            with tarfile.open(fileobj=reader, mode='r|gz') as archive:
                for member in archive:
                    name = member.name[2:] if member.name.startswith('./') else member.name
                    if name not in needed:
                        continue
                    assert member.isfile() and not Path(name).is_absolute() and '..' not in Path(name).parts
                    dest = ROOT / name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    temp = dest.with_suffix(dest.suffix + '.part')
                    digest = hashlib.sha256()
                    with archive.extractfile(member) as src, temp.open('wb') as target:
                        for block in iter(lambda: src.read(1024**2), b''):
                            digest.update(block); target.write(block)
                    assert temp.stat().st_size == member.size
                    temp.replace(dest)
                    acquired[name] = dict(bytes=member.size, sha256=digest.hexdigest(), reused=False)
                    needed.remove(name)
                    print('ACQUIRED', name, 'remaining', len(needed), flush=True)
                    save(ROOT / 'acquired-files.json', acquired)
                    if not needed:
                        break
            manifest['compressed_bytes_read'] = reader.count
    assert not needed, 'Missing selected files: ' + repr(needed)
    manifest.update(acquisition_status='passed', files=acquired,
                    retained_sensor_bytes=sum(x['bytes'] for x in acquired.values()),
                    archive_full_checksum_verified=False,
                    integrity_note='Selected member sizes checked; per-file hashes recorded. Stream stopped early; no whole-archive checksum claim.')
    save(ROOT / 'subset-manifest.json', manifest)
    save(ROOT / 'acquired-files.json', acquired)
    print('TRAINVAL_INPUTS_ACQUIRED', flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('stage', choices=['select','download'])
    a=p.parse_args()
    (select if a.stage == 'select' else download)()
