"""Pod-external cleanup verifier; run under caffeinate on the controlling Mac.

No runtime cap. After arming, missing/stale job heartbeat for 180s is a fault.
The host must remain powered and connected; the Pod-side guard is independent.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time
from job_guard import request, write_state


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    key = Path(cfg['key_file']).read_text().strip()
    last_healthy = time.monotonic()
    armed = False
    deleting = False
    last_heartbeat = None
    terminal_seen = None
    artifacts_saved = False
    while True:
        try:
            code, pod = request(cfg['pod_id'], key)
            if code == 404:
                print(json.dumps(dict(event='deletion_verified', pod_id=cfg['pod_id'], time=time.time())), flush=True)
                return
            if code in (401, 403):
                print('Pod credential revoked/denied; account-level MCP verification required', flush=True)
                return
            if code != 200:
                print('API verification HTTP', code, flush=True)
                time.sleep(15)
                continue
            result = subprocess.run(cfg['ssh'] + ['cat ' + cfg['state_file']],
                                    capture_output=True, text=True, timeout=25)
            if result.returncode == 0:
                state = json.loads(result.stdout)
                assert state['pod_id'] == cfg['pod_id']
                write_state(a.config.parent / (cfg['pod_id'] + '.observed-state.json'), state)
                armed = True
                if state['status'] in ('completed', 'failed'):
                    if terminal_seen is None:
                        terminal_seen = time.monotonic()
                    deleting = time.monotonic() >= terminal_seen + 30
                    if not artifacts_saved:
                        paths = ['/workspace/results/deployment/trainval-cpu-' + cfg['pod_id'] + '/preflight.json',
                                 '/workspace/results/deployment/trainval-cpu-' + cfg['pod_id'] + '/input-check.json',
                                 '/workspace/results/deployment/trainval-cold-cpu-' + cfg['pod_id'] + '/preflight.json',
                                 '/workspace/cache/bevfusion-runtime/model-loads-cpu.json',
                                 '/workspace/cache/bevfusion-runtime/input-validation.json',
                                 '/workspace/cache/bevfusion-runtime/runner-preflight-fusion.json',
                                 '/workspace/cache/bevfusion-runtime/runner-preflight-lidar.json',
                                 '/workspace/cache/bevfusion-runtime/runtime.sha256',
                                 '/workspace/cache/bevfusion-runtime/installed.freeze.txt',
                                 cfg['state_file'].rsplit('/',1)[0] + '/guard.log',
                                 '/workspace/results/deployment/trainval-cpu-' + cfg['pod_id'] + '/preparation.log',
                                 '/workspace/datasets/nuscenes-trainval-smoke/subset-manifest.json',
                                 '/workspace/cache/bevfusion-runtime/os-runtime-manifest.json']
                        artifacts = {}
                        for path in paths:
                            item = subprocess.run(cfg['ssh'] + ['cat ' + path], capture_output=True,
                                                  text=True, timeout=15)
                            if item.returncode == 0:
                                artifacts[path] = item.stdout
                                write_state(a.config.parent / (cfg['pod_id'] + '.artifacts.json'), artifacts)
                        write_state(a.config.parent / (cfg['pod_id'] + '.artifacts.json'), artifacts)
                        artifacts_saved = True
                elif state['heartbeat'] != last_heartbeat:
                    last_heartbeat = state['heartbeat']
                    last_healthy = time.monotonic()
            if armed and time.monotonic() - last_healthy > 180:
                deleting = True
            if deleting:
                code, _ = request(cfg['pod_id'], key, 'DELETE')
                print(json.dumps(dict(event='delete_requested', http=code, time=time.time())), flush=True)
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            print('Monitor retry:', type(exc).__name__, flush=True)
            if armed and time.monotonic() - last_healthy > 180:
                try:
                    code, _ = request(cfg['pod_id'], key, 'DELETE')
                    print('Unreachable job cleanup HTTP', code, flush=True)
                except OSError:
                    pass
        time.sleep(10)


if __name__ == '__main__':
    main()
