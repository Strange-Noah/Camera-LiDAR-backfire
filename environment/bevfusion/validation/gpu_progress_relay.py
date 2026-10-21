"""Relay inference counters, never timer-only heartbeats, to the root-only guard."""
import json
from pathlib import Path
import time

source = Path('/home/bfbaseline/results/progress.json')
target = Path('/root/gpu-progress.json')
last = None
while True:
    try:
        marker = source.stat().st_mtime_ns
        if marker != last:
            value = json.loads(source.read_text())
            temporary = target.with_suffix('.tmp')
            temporary.write_text(json.dumps(value))
            temporary.replace(target)
            last = marker
    except FileNotFoundError:
        pass
    time.sleep(3)
