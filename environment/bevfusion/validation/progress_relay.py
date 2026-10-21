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
    try:
        marker = source.stat().st_mtime_ns
        if marker != last:
            value = json.loads(source.read_text())
            temporary = target.with_suffix('.relay.tmp')
            temporary.write_text(json.dumps(value))
            temporary.replace(target)
            last = marker
    except (OSError, json.JSONDecodeError):
        # A short/truncated read is not new progress. Retry without refreshing
        # the guard, and do not let a writer race kill the relay permanently.
        pass
    time.sleep(5)
