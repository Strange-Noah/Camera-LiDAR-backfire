"""Check every available clean-val file; missing payload blocks GPU admission."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    selected = [e for e in manifest['files'] if 'A' in e['scenarios']]
    def verify(entry):
        locations = entry.get('available_locations', [])
        if not locations:
            return dict(relative_path=entry['relative_path'], status='MISSING')
        path = Path(locations[0])
        try:
            assert path.stat().st_size == entry['uncompressed_size_bytes']
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 ** 2), b''):
                    h.update(block)
            if entry['sensor_modality'] == 'camera':
                from PIL import Image
                with Image.open(path) as image:
                    assert image.format == 'JPEG' and image.size == (1600, 900)
                    image.verify()
                with Image.open(path) as image:
                    image.load()
                detail = dict(image_dimensions=[1600, 900], jpeg_decode='PASS')
            else:
                import numpy as np
                assert path.stat().st_size > 0 and path.stat().st_size % 20 == 0
                points = np.fromfile(str(path), dtype=np.float32).reshape(-1, 5)
                assert np.isfinite(points).all()
                detail = dict(points=len(points), channels=5, binary_structure='PASS')
            return dict(relative_path=entry['relative_path'], path=str(path), status='PASS',
                bytes=path.stat().st_size, sha256=h.hexdigest(), **detail)
        except Exception as exc:
            return dict(relative_path=entry['relative_path'], status='FAIL', error=repr(exc))
    records = []
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, record in enumerate(pool.map(verify, selected)):
            records.append(record)
            if index % 1000 == 0:
                print(json.dumps(dict(stage='payload_validation', checked=index+1, total=len(selected), time=time.time())), flush=True)
                path = args.output.parent / 'prepare-progress.json'
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps(dict(checked=index+1, time=time.time())))
                temp.replace(path)
    missing = sum(r['status'] == 'MISSING' for r in records)
    failed = sum(r['status'] == 'FAIL' for r in records)
    report = dict(status='FAIL' if failed else 'BLOCKED_MISSING_PAYLOAD' if missing else 'PASS',
        dataset_version='v1.0-trainval', split='val', samples=6019, required_files=len(selected),
        verified_files=sum(r['status'] == 'PASS' for r in records), missing_files=missing, failed_files=failed,
        archive_crc_status=manifest['archive_audit_status'], seconds=time.time()-start,
        checksum_policy='Locally computed SHA256 after extraction; size is matched to official tar headers; whole archive gzip CRC was checked. No publisher per-member SHA256 is available.',
        calibration_ego_pose_sweep_token_check='Manifest constructor asserts references; converter report separately verifies info paths',
        files=records)
    args.output.write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'files'}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
