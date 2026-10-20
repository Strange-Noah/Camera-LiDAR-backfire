"""Real detached process tests; the provider API is simulated, never contacted."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKER = r'''
import json,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
import job_guard as g
base=Path(sys.argv[2]);case=sys.argv[3];results=base/'results';results.mkdir()
(base/'key').write_text('test-only')
attempt=0
def request(pid,key,method='GET'):
 global attempt
 with (base/'requests.jsonl').open('a') as out:out.write(json.dumps(dict(method=method,time=time.time()))+'\n')
 if method=='GET':return 200,dict(id='test')
 attempt+=1
 if case=='retry' and attempt==1:raise OSError('simulated API transport failure')
 return 404,None
g.request=request;g.os.sync=lambda:None
child="import time; time.sleep(60)"
if case=='failure':
 child="from pathlib import Path; import time; p=Path("+repr(str(results/'fusion'))+");p.mkdir();(p/'acceptance.json').write_text('{\"status\":\"failed\"}');time.sleep(60)"
elif case=='retry':child='raise SystemExit(0)'
args=['guard','--pod-id','test','--key-file',str(base/'key'),'--state',str(base/'state.json'),'--grace-seconds','0','--results-dir',str(results)]
if case=='stall':args+=['--progress-file',str(results/'progress.json'),'--stall-seconds','1']
sys.argv=args+['--',sys.executable,'-c',child]
g.main()
(base/'finished').write_text('passed')
'''


class GuardLifecycle(unittest.TestCase):
    def run_case(self, case):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            worker = base / 'worker.py'
            worker.write_text(WORKER)
            with (base / 'worker.log').open('w') as log:
                task = subprocess.Popen([sys.executable, str(worker), str(ROOT / 'environment/bevfusion'),
                                         str(base), case], stdin=subprocess.DEVNULL,
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                self.assertEqual(task.wait(timeout=45), 0, (base / 'worker.log').read_text())
            self.assertTrue((base / 'finished').exists())
            state = json.loads((base / 'state.json').read_text())
            requests = [json.loads(line) for line in (base / 'requests.jsonl').read_text().splitlines()]
            return state, requests

    def test_failure_report_terminates_hung_child(self):
        state, requests = self.run_case('failure')
        self.assertEqual(state['forced_cleanup_reason'], 'explicit_failure_report')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(requests[-1]['method'], 'DELETE')

    def test_stalled_child_terminated_without_total_runtime_cap(self):
        state, requests = self.run_case('stall')
        self.assertEqual(state['forced_cleanup_reason'], 'no_progress')
        self.assertEqual(requests[-1]['method'], 'DELETE')

    def test_api_transport_failure_retried(self):
        state, requests = self.run_case('retry')
        self.assertEqual(state['status'], 'completed')
        self.assertEqual(sum(item['method'] == 'DELETE' for item in requests), 2)


if __name__ == '__main__':
    unittest.main()
