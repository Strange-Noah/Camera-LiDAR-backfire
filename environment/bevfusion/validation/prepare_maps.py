"""Acquire only four official map JSONs needed by the unchanged test pipeline."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

URL = 'https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/nuScenes-map-expansion-v1.3.zip'
LOCATIONS = ['boston-seaport', 'singapore-hollandvillage', 'singapore-onenorth', 'singapore-queenstown']
OUT = Path('/workspace/results/validation-baseline-preparation')
DEST = Path('/workspace/datasets/nuscenes-validation/maps/expansion')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    existing_report = OUT / 'map-acquisition-report.json'
    if existing_report.exists():
        old = json.loads(existing_report.read_text())
        if old.get('status') == 'PASS' and len(old.get('files', [])) == 4 and all(
                Path(r['path']).is_file() and digest(Path(r['path'])) == r['sha256'] for r in old['files']):
            print('MAP_CACHE_SHA256_VERIFIED', flush=True)
            return
    # Compressed map archive is temporary on the disposable CPU container disk.
    with tempfile.TemporaryDirectory(prefix='nuscenes-maps-') as folder:
        archive_path = Path(folder) / 'maps.zip'
        with urllib.request.urlopen(URL, timeout=60) as source, archive_path.open('wb') as target:
            expected = int(source.headers['Content-Length'])
            shutil.copyfileobj(source, target, length=1024 ** 2)
        assert archive_path.stat().st_size == expected == 398535531
        members = []
        with zipfile.ZipFile(str(archive_path)) as archive:
            for location in LOCATIONS:
                matches = [n for n in archive.namelist() if n.endswith('expansion/'+location+'.json') and not n.startswith('__MACOSX')]
                assert len(matches) == 1, matches
                info = archive.getinfo(matches[0])
                path = DEST / (location+'.json')
                temporary = path.with_suffix('.part')
                with archive.open(info) as source, temporary.open('wb') as target:
                    shutil.copyfileobj(source, target, length=1024 ** 2)
                assert temporary.stat().st_size == info.file_size
                temporary.replace(path)
                data = json.loads(path.read_text())
                assert 'node' in data and 'polygon' in data and 'drivable_area' in data
                members.append(dict(archive_member=info.filename, path=str(path), uncompressed_bytes=info.file_size,
                    crc32='%08x' % info.CRC, sha256=digest(path), status='PASS'))
        report = dict(status='PASS', source=URL, compressed_bytes=expected, archive_sha256=digest(archive_path),
            retained_uncompressed_bytes=sum(x['uncompressed_bytes'] for x in members), files=members,
            rationale='Official test pipeline LoadBEVSegmentation requires these four JSONs even though the detection head does not consume the BEV GT masks.',
            archive_retained=False, temporary_storage='Disposable CPU container disk; removed after extraction')
        (OUT / 'map-acquisition-report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
