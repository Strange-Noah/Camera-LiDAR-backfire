#!/usr/bin/env python3
"""Explicit mini-only acquisition. Run on a Pod; never downloads trainval."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import tarfile
import time

URL = 'https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-mini.tgz'
PAGE = 'https://www.nuscenes.org/nuscenes#download'
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--persistent-root', default='/workspace')
p.add_argument('--official-sha256', help='Only supply a checksum published by the official source')
a = p.parse_args()
base = Path(a.persistent_root).resolve()
if platform.system() != 'Linux' or not base.is_mount():
    raise SystemExit('Run this on the Linux Pod with the persistent volume mounted.')
root = base / 'datasets/nuscenes'
staging = base / 'staging'
root.mkdir(parents=True, exist_ok=True)
staging.mkdir(parents=True, exist_ok=True)
manifest_path = root / 'dataset_manifest.json'
if manifest_path.exists():
    raise SystemExit('Dataset manifest already exists. Run verify_nuscenes.sh; acquisition will not overwrite it.')
if any(root.iterdir()):
    raise SystemExit('Dataset root is not empty. Inspect partial data before retrying; no files were overwritten.')
archive = staging / 'v1.0-mini.tgz'
partial = staging / 'v1.0-mini.tgz.part'
started = dt.datetime.now(dt.timezone.utc).isoformat()
start = time.monotonic()
if not archive.exists():
    subprocess.run(['curl', '--fail', '--location', '--proto', '=https', '--proto-redir', '=https',
                    '--retry', '3', '--retry-delay', '5', '--connect-timeout', '30',
                    '--continue-at', '-', '--output', str(partial), URL], check=True)
    partial.rename(archive)
subprocess.run(['gzip', '-t', str(archive)], check=True)
h = hashlib.sha256()
with archive.open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(block)
digest = h.hexdigest()
if a.official_sha256 and digest.lower() != a.official_sha256.lower():
    raise SystemExit('Official SHA-256 mismatch; archive retained, extraction refused.')
with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
            raise SystemExit(f'Unsafe or unsupported archive member: {member.name}')
    print(f'Archive integrity passed; extracting {len(members)} entries', flush=True)
    tar.extractall(root, filter='data')
    expected = [(m.name, m.size) for m in members if m.isfile()]
for name, size in expected:
    path = root / name
    if not path.is_file() or path.stat().st_size != size:
        raise SystemExit(f'Extracted size mismatch: {name}')
for rel in ('maps', 'samples/CAM_FRONT', 'samples/LIDAR_TOP', 'sweeps', 'v1.0-mini'):
    if not any(x.is_file() and x.stat().st_size > 0 for x in (root / rel).rglob('*')):
        raise SystemExit(f'No nonempty files in {rel}')
manifest = {
    'dataset': 'nuScenes', 'version': 'v1.0-mini', 'source': 'official nuScenes',
    'source_page': PAGE, 'source_url': URL,
    'source_provenance': 'URL supplied by user after login through official Mini download control',
    'dataroot': str(root), 'download_started_utc': started,
    'validation_completed_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'archive': {'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': digest,
                'official_sha256': a.official_sha256,
                'checksum_note': 'Locally calculated SHA-256 is a reproducibility fingerprint, not an official authenticity checksum.',
                'gzip_integrity': 'passed', 'tar_member_validation': 'passed', 'retained': True},
    'extracted_file_count': len(expected), 'extracted_file_bytes': sum(s for _, s in expected),
    'extracted_sizes_verified_against_archive': True,
    'disk_usage_bytes': int(subprocess.check_output(['du', '-s', '-B1', str(root)], text=True).split()[0]),
    'duration_seconds': round(time.monotonic() - start, 2)
}
manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
(staging / 'v1.0-mini.manifest.json').write_text(manifest_path.read_text())
print(json.dumps(manifest, indent=2))
