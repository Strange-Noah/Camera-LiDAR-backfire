"""Root-only result mirror: atomic per-token handoff out of the readonly workload."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

LOCAL = Path(os.environ.get('VALIDATION_LOCAL_RESULTS', '/home/bfbaseline/results'))
REMOTE = Path(os.environ.get('VALIDATION_PERSISTED_RESULTS', '/workspace/results/validation-baseline-preparation/clean-predictions'))
STOP = Path(os.environ.get('VALIDATION_STOP_MARKER', '/root/gpu-inference-done'))
VERIFIED = set()


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def synchronize():
    for run in LOCAL.glob('*-*'):
        if not run.is_dir():
            continue
        model = run.name.split('-')[0]
        if model not in ['fusion', 'lidar']:
            continue
        for marker in run.glob('*/complete.json'):
            record = json.loads(marker.read_text())
            token = record['token']
            assert marker.parent.name == token and len(token) == 32 and token.isalnum()
            dest = REMOTE/model/token
            if dest.exists():
                assert (dest/'complete.json').read_text() == marker.read_text()
                if dest not in VERIFIED:
                    assert digest(dest/'prediction.pkl') == record['raw_sha256']
                    VERIFIED.add(dest)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temp = dest.with_name(token+'.pending')
            temp.mkdir(exist_ok=True)
            shutil.copyfile(marker.parent/'prediction.pkl', temp/'prediction.pkl')
            assert digest(temp/'prediction.pkl') == record['raw_sha256']
            shutil.copyfile(marker, temp/'complete.json')
            temp.replace(dest)
            VERIFIED.add(dest)
        provenance = REMOTE/'runs'/run.name
        provenance.mkdir(parents=True, exist_ok=True)
        for name in ['execution.json', 'resolved_config.py', 'input-selection.jsonl']:
            source = run/name
            if source.exists():
                temporary = provenance/(name+'.tmp')
                shutil.copyfile(source, temporary)
                temporary.replace(provenance/name)


while True:
    synchronize()
    if STOP.exists():
        # STOP may have appeared while the preceding directory iterator was
        # still scanning an older snapshot. The producer is now finished;
        # rescan once with no concurrent producer before acknowledging exit.
        synchronize()
        break
    time.sleep(3)
print('GPU_OUTPUT_PERSISTENCE_COMPLETE', flush=True)
