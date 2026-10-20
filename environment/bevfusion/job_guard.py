"""Supervise one complete job; persist completion and request Pod deletion after 30s.

No total-runtime limit. The independent controller must verify deletion externally.
Credential is provided through a private file, never command-line text or logs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import signal
import time
import urllib.error
import urllib.request


def request(pod_id, key, method='GET'):
    req = urllib.request.Request('https://api.runpod.io/v2/pods/' + pod_id,
                                 method=method, headers={'Authorization': 'Bearer ' + key})
    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            body = response.read()
            return response.status, json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        return exc.code, None


def write_state(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod-id', required=True)
    parser.add_argument('--key-file', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--grace-seconds', type=float, default=30)
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--progress-file', type=Path)
    parser.add_argument('--stall-seconds', type=float, default=300)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    a = parser.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        parser.error('Missing job command')
    if a.grace_seconds < 0:
        parser.error('Grace period cannot be negative')
    key = a.key_file.read_text().strip()
    status, info = request(a.pod_id, key)
    if status != 200 or info['id'] != a.pod_id:
        raise SystemExit('Pod-scoped API access check failed: ' + str(status))
    a.state.parent.mkdir(parents=True, exist_ok=True)
    state = dict(pod_id=a.pod_id, status='running', started_at=time.time())
    child = subprocess.Popen(command, start_new_session=True)
    state['pid'] = child.pid
    last_progress = time.monotonic()
    last_marker = None
    while child.poll() is None:
        state['heartbeat'] = time.time()
        write_state(a.state, state)
        failure = None
        if a.results_dir and a.results_dir.exists():
            for path in a.results_dir.glob('*/acceptance.json'):
                try:
                    if json.loads(path.read_text()).get('status') == 'failed':
                        failure = 'explicit_failure_report'
                except (OSError, ValueError):
                    pass
        if a.progress_file:
            try:
                marker = a.progress_file.stat().st_mtime_ns
                if marker != last_marker:
                    last_progress = time.monotonic()
                    last_marker = marker
            except FileNotFoundError:
                pass
            if time.monotonic() - last_progress > a.stall_seconds:
                failure = 'no_progress'
        if failure:
            state['forced_cleanup_reason'] = failure
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            break
        time.sleep(5)
    if a.results_dir and a.results_dir.exists():
        shutil.copytree(str(a.results_dir), str(a.state.parent / 'predictions'),
                        dirs_exist_ok=True, symlinks=True)
    state.update(status='completed' if child.returncode == 0 else 'failed',
                 exit_code=child.returncode, finished_at=time.time())
    write_state(a.state, state)
    os.sync()
    time.sleep(a.grace_seconds)
    while True:
        try:
            state['cleanup_requested_at'] = time.time()
            write_state(a.state, state)
            status, _ = request(a.pod_id, key, 'DELETE')
            print('Pod deletion request HTTP', status, flush=True)
            if status == 404:
                return
            if status in (401, 403):
                raise SystemExit('Deletion denied; external controller must clean up')
        except (OSError, TimeoutError):
            print('Deletion transport error; retrying', flush=True)
        time.sleep(15)


if __name__ == '__main__':
    main()
