"""CPU only: preserve diagnostics and finish the automatic artifact handoff.

The existing job guard enforces a no-progress timeout while awaiting the local
controller's receipt acknowledgement. Never use this handoff wait on a GPU.
"""
import argparse
import json
import subprocess
import tarfile
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inspect-only', action='store_true')
    args = p.parse_args()
    root = Path('/workspace/results/validation-baseline-preparation')
    (root/'delivery-ready.json').unlink(missing_ok=True)
    Path('/root/validation-delivery-ack').unlink(missing_ok=True)
    if args.inspect_only:
        code = 0
    else:
        code = subprocess.call(['bash', str(Path(__file__).with_name('prepare_cpu.sh'))])
    (root/'cpu-phase-exit.json').write_text(json.dumps(dict(exit_code=code, finished_at=time.time()))+'\n')
    packet = root/'local-preparation-delivery.tgz'
    with tarfile.open(str(packet), 'w:gz') as archive:
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.name == 'validation-file-manifest.json':
                continue
            if path.suffix not in ['.json', '.log', '.md', '.txt', '.py', '.yaml']:
                continue
            archive.add(str(path), arcname=str(path.relative_to(root)))
    marker = root/'delivery-ready.json'
    marker.write_text(json.dumps(dict(packet=str(packet), bytes=packet.stat().st_size, cpu_phase_exit_code=code, time=time.time()))+'\n')
    progress = root/'prepare-progress.json'
    temporary = progress.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(stage='cpu_artifact_handoff_ready',time=time.time())))
    temporary.replace(progress)
    print('CPU_DELIVERY_READY', packet, flush=True)
    # Transfer/acknowledgement is part of the CPU task. If the controller goes
    # away, the independent Pod guard deletes this CPU Pod after its stall limit.
    while not Path('/root/validation-delivery-ack').exists():
        time.sleep(1)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
