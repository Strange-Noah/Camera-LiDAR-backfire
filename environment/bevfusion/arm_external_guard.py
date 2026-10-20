"""Arm the local independent guard for a Pod created by this deployment."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from job_guard import request


p = argparse.ArgumentParser()
p.add_argument('--pod-id', required=True)
p.add_argument('--host', required=True)
p.add_argument('--port', required=True, type=int)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
local = root / '.local' / 'bevfusion-runtime'
local.mkdir(parents=True, exist_ok=True)
local.chmod(0o700)
key = local / (a.pod_id + '.key')
common = ['-i', str(Path.home() / '.ssh/id_ed25519'), '-o',
          'StrictHostKeyChecking=accept-new', '-o',
          'UserKnownHostsFile=' + str(root / '.local/known_hosts'),
          '-o', 'ConnectTimeout=12']
subprocess.run(['scp'] + common + ['-P', str(a.port),
               'root@' + a.host + ':/root/.bevfusion-pod-key', str(key)], check=True, timeout=45)
key.chmod(0o600)
code, pod = request(a.pod_id, key.read_text().strip())
assert code == 200 and pod['id'] == a.pod_id, 'Pod identity check failed'
cfg = dict(pod_id=a.pod_id, key_file=str(key),
           state_file='/workspace/results/deployment/bevfusion-' + a.pod_id + '/job-state.json',
           ssh=['ssh'] + common + ['-p', str(a.port), 'root@' + a.host])
config = local / (a.pod_id + '.json')
config.write_text(json.dumps(cfg)); config.chmod(0o600)
command = [sys.executable, '-u', str(Path(__file__).with_name('external_guard.py')), '--config', str(config)]
if shutil.which('caffeinate'):
    command = ['caffeinate', '-i'] + command
with (local / (a.pod_id + '.monitor.log')).open('a') as log:
    task = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
(local / (a.pod_id + '.monitor.pid')).write_text(str(task.pid))
print(json.dumps(dict(pod_id=a.pod_id, external_monitor_pid=task.pid, api_access=code)))
