"""Maintenance-only acquisition into the existing persistent volume."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request


def main():
    if not os.path.ismount('/workspace'):
        raise SystemExit('/workspace must be the attached network volume')
    spec = json.loads(Path(__file__).with_name('upstream.json').read_text())
    target = Path('/workspace/checkpoints/bevfusion')
    target.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target).free < 4 * 1024**3:
        raise SystemExit('Less than 4 GiB filesystem space available')
    # Network-volume df may report pool space. Operator must also check du vs 50 GB quota.
    records = []
    for name, model in spec['models'].items():
        dest = target / model['file']
        if not dest.exists():
            partial = dest.with_suffix('.part')
            with urllib.request.urlopen(model['url'], timeout=120) as response:
                content_type = response.headers.get('Content-Type', '')
                if 'text/html' in content_type:
                    raise RuntimeError('Download returned HTML instead of checkpoint')
                with partial.open('wb') as out:
                    shutil.copyfileobj(response, out)
            if partial.stat().st_size < 1024 * 1024:
                raise RuntimeError('Unexpectedly small checkpoint')
            partial.replace(dest)
        digest = hashlib.sha256()
        with dest.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        records.append(dict(model=model, name=name, bytes=dest.stat().st_size,
                            sha256=digest.hexdigest(), code_commit=spec['commit']))
        dest.chmod(0o444)
    (target / 'manifest.json').write_text(json.dumps(records, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
