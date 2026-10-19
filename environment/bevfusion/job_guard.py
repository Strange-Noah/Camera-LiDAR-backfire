"""Supervise one complete job; persist completion and request Pod deletion after 30s.

No total-runtime limit. The independent controller must verify deletion externally.
Credential is provided through a private file, never command-line text or logs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
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
    parser.add_argument('command', nargs=argparse.REMAINDER)
    a = parser.parse_args()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    if not command:
        parser.error('Missing job command')
    key = a.key_file.read_text().strip()
    status, info = request(a.pod_id, key)
    if status != 200 or info['id'] != a.pod_id:
        raise SystemExit('Pod-scoped API access check failed: ' + str(status))
    a.state.parent.mkdir(parents=True, exist_ok=True)
    state = dict(pod_id=a.pod_id, status='running', started_at=time.time())
    child = subprocess.Popen(command, start_new_session=True)
    state['pid'] = child.pid
    while child.poll() is None:
        state['heartbeat'] = time.time()
        write_state(a.state, state)
        time.sleep(5)
    state.update(status='completed' if child.returncode == 0 else 'failed',
                 exit_code=child.returncode, finished_at=time.time())
    write_state(a.state, state)
    os.sync()
    time.sleep(30)
    while True:
        try:
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
