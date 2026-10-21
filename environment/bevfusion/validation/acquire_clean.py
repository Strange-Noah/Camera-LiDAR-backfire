"""Resume clean-val extraction from audited official archives; never retain tgz files."""
import concurrent.futures
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import threading
import time
import urllib.request

from audit_manifest import BASE, BASES, DEFAULT_OUT, VALROOT, Reader, dump, inventory, sha, tick

LOCK = threading.Lock()
CAPACITY = 90_000_000_000
RESERVE = 5_000_000_000


def main():
    out = DEFAULT_OUT
    os.environ['VALIDATION_PROGRESS_FILE'] = str(out / 'prepare-progress.json')
    manifest_path = out / 'validation-file-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['archive_audit_status'] == 'passed'
    assert manifest['scene_count'] == 150 and len(manifest['samples']) == 6019
    present = inventory(out)
    selected = [e for e in manifest['files'] if 'A' in e['scenarios']]
    assert len(selected) == 94522
    pending = {}
    for entry in selected:
        locations = present.get(entry['relative_path'], [])
        for path in locations:
            assert Path(path).stat().st_size == entry['uncompressed_size_bytes'], path
        if not locations:
            pending[entry['relative_path']] = entry
    used = int(subprocess.check_output(['du', '-sx', '--block-size=1', '/workspace'], text=True).split()[0])
    budget = dict(used=used, allocated=0)
    required = sum(e['uncompressed_size_bytes'] for e in pending.values())
    assert used + required + RESERVE <= CAPACITY, (used, required, 'Insufficient authorized capacity')
    dump(out / 'acquisition-start.json', dict(time=time.time(), missing_files=len(pending), missing_bytes=required,
        measured_volume_bytes=used, capacity_bytes=CAPACITY, minimum_reserve_bytes=RESERVE,
        source_manifest_sha256=sha(manifest_path)))
    groups = {}
    for name, entry in pending.items():
        groups.setdefault(entry['archive_members'][0], {})[name] = entry

    def acquire(item):
        name, wanted = item
        report_path = out / ('acquisition-' + name + '.json')
        for attempt in range(3):
            try:
                count = 0
                records = []
                tick(out, 'clean_archive_start', archive=name, needed=len(wanted), attempt=attempt+1)
                with urllib.request.urlopen(BASE + name, timeout=90) as response:
                    expected = int(response.headers['Content-Length'])
                    reader = Reader(response, out, name)
                    with gzip.GzipFile(fileobj=reader, mode='rb') as gz:
                        with tarfile.open(fileobj=gz, mode='r|', bufsize=1024**2) as archive:
                            for member in archive:
                                relative = member.name.removeprefix('./') if hasattr(str, 'removeprefix') else member.name[2:] if member.name.startswith('./') else member.name
                                if relative not in wanted:
                                    continue
                                assert member.isfile() and not Path(relative).is_absolute() and '..' not in Path(relative).parts
                                entry = wanted[relative]
                                assert member.size == entry['uncompressed_size_bytes']
                                dest = VALROOT / relative
                                if dest.exists():
                                    assert dest.stat().st_size == member.size
                                    count += 1
                                    continue
                                with LOCK:
                                    assert budget['used'] + budget['allocated'] + member.size + RESERVE <= CAPACITY
                                    # The shared filesystem's statvfs may show host capacity; enforce the volume quota independently.
                                    fs = os.statvfs('/workspace')
                                    assert fs.f_bavail * fs.f_frsize > member.size + RESERVE
                                    budget['allocated'] += member.size
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                temporary = dest.with_suffix(dest.suffix + '.part')
                                h = hashlib.sha256()
                                with archive.extractfile(member) as source, temporary.open('wb') as target:
                                    for block in iter(lambda: source.read(1024**2), b''):
                                        h.update(block)
                                        target.write(block)
                                    target.flush()
                                    os.fsync(target.fileno())
                                assert temporary.stat().st_size == member.size
                                temporary.replace(dest)
                                count += 1
                                record = dict(relative_path=relative, bytes=member.size, sha256=h.hexdigest())
                                records.append(record)
                                with LOCK:
                                    with (out / 'acquired-clean-files.jsonl').open('a') as log:
                                        log.write(json.dumps(dict(archive=name, **record)) + '\n')
                                if count % 200 == 0:
                                    tick(out, 'clean_extract', archive=name, completed=count, needed=len(wanted), allocated_bytes=budget['allocated'])
                        while gz.read(1024**2):
                            pass
                    assert reader.count == expected
                assert count == len(wanted), (name, count, len(wanted))
                report = dict(status='PASS', archive=name, source=BASE+name, compressed_bytes=expected,
                    gzip_crc='PASS', required_files=count, newly_extracted=len(records), completed_at=time.time())
                dump(report_path, report)
                tick(out, 'clean_archive_complete', **report)
                return report
            except Exception as exc:
                tick(out, 'clean_acquisition_retry', archive=name, attempt=attempt+1, error=repr(exc))
                if attempt == 2:
                    raise
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        reports = list(pool.map(acquire, sorted(groups.items())))
    present = inventory(out)
    for entry in manifest['files']:
        entry['available_locations'] = present.get(entry['relative_path'], [])
        entry['availability_status'] = 'present_size_verified' if entry['available_locations'] else 'missing'
    assert all(e['available_locations'] for e in selected)
    dump(manifest_path, manifest)
    dump(out / 'clean-extraction-report.json', dict(status='PASS', required_files=len(selected), archives=reports,
        manifest_sha256=sha(manifest_path), newly_allocated_bytes=budget['allocated'], finished_at=time.time()))
    tick(out, 'clean_extraction_complete', files=len(selected))


if __name__ == '__main__':
    main()
