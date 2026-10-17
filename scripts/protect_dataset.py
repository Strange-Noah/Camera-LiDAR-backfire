#!/usr/bin/env python3
"""Administrator-only permission change; never edits dataset contents.
Run on the mounted project volume. Saves original metadata for review/recovery.
"""
import argparse, collections, datetime, gzip, hashlib, json, os, stat
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--apply', action='store_true', help='Required to change permissions')
a = p.parse_args()
if not a.apply or os.geteuid() != 0:
    p.error('Run as root with --apply after reviewing the target and mount')
root = Path('/workspace/datasets/nuscenes')
if not os.path.ismount('/workspace') or not (root/'v1.0-mini/scene.json').is_file() or not (root/'v1.0-trainval/scene.json').is_file():
    raise SystemExit('Expected mounted project dataset missing; refusing changes')
# A complete preflight precedes any permissions change.
paths = [Path('/workspace'), Path('/workspace/datasets')]
for parent, dirs, files in os.walk(root, followlinks=False):
    paths.append(Path(parent))
    for name in dirs:
        if (Path(parent)/name).is_symlink():
            raise SystemExit('Symlink directory found; review manually')
    paths.extend(Path(parent)/name for name in files)
records = []
for path in paths:
    s = path.lstat()
    if not (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)):
        raise SystemExit(f'Unexpected file type: {path}')
    if any('posix_acl' in x for x in os.listxattr(path)):
        raise SystemExit(f'ACL requires manual review: {path}')
    records.append(dict(path=str(path), uid=s.st_uid, gid=s.st_gid,
                        mode=stat.S_IMODE(s.st_mode), size=s.st_size,
                        mtime_ns=s.st_mtime_ns, directory=stat.S_ISDIR(s.st_mode)))
    if len(records) % 10000 == 0:
        print('Preflight entries:', len(records), flush=True)
base = Path('/workspace/results/storage_permissions')
base.mkdir(exist_ok=True)
os.chown(base, 0, 0); os.chmod(base, 0o755)
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
snapshot = base / (stamp+'-before.jsonl.gz')
with gzip.open(snapshot, 'xt') as f:
    for rec in records:
        f.write(json.dumps(rec)+'\n')
os.chmod(snapshot, 0o600)
# Hash every metadata/provenance JSON, plus one camera and LiDAR file.
checks = sorted(root.glob('*.json')) + sorted(root.glob('v1.0-*/*.json'))
for sub in ['samples/CAM_FRONT', 'samples/LIDAR_TOP']:
    checks.append(next(x for x in (root/sub).iterdir() if x.is_file()))
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()
before = {str(x): digest(x) for x in checks}
print('Snapshot and content fingerprints saved in memory; applying modes', flush=True)
for i, rec in enumerate(records):
    path = Path(rec['path'])
    os.chown(path, 0, 0, follow_symlinks=False)
    # Parent names must also be protected on this FUSE backend.
    mode = 0o555 if rec['directory'] else 0o444
    os.chmod(path, mode)
    if i and i % 10000 == 0: print('Protected entries:', i, flush=True)
for rec in records:
    path = Path(rec['path']); s = path.stat()
    mode = 0o555 if rec['directory'] else 0o444
    assert (s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode)) == (0, 0, mode), path
    assert s.st_mtime_ns == rec['mtime_ns'], path
    if not rec['directory']: assert s.st_size == rec['size'], path
assert before == {str(x): digest(x) for x in checks}, 'Content fingerprint changed'
report = dict(status='PASS', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              volume_id='p6vb29fmlu', dataset_root=str(root), entries=len(records),
              files=sum(not x['directory'] for x in records),
              permission_snapshot=str(snapshot), file_mode='0444', directory_mode='0555',
              parent_mode='0555', owner='0:0', sizes_and_mtimes_preserved=True,
              sha256_unchanged=before,
              limitations=['Non-root experiments only; root and control-plane Dev can bypass protections.',
                           'Non-root login must be provisioned on every new/restarted disposable Pod.'])
(base/'protection_report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='sha256_unchanged'}, indent=2))
