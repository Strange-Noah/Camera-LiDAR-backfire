"""Root supervisor relays actual non-root pipeline progress to the CPU guard."""
import json
import sys
import time
from pathlib import Path

parent = int(sys.argv[1])
source = Path('/home/bfbaseline/results/progress.json')
target = Path('/workspace/results/validation-baseline-preparation/prepare-progress.json')
last = None
while Path('/proc/%d' % parent).exists():
    if source.exists():
        marker = source.stat().st_mtime_ns
        if marker != last:
            value = json.loads(source.read_text())
            temporary = target.with_suffix('.relay.tmp')
            temporary.write_text(json.dumps(value))
            temporary.replace(target)
            last = marker
    time.sleep(5)
