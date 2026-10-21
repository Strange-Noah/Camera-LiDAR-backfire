"""Exercise the durable handoff boundary without a GPU or real predictions."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.local = self.root/'local'
        self.remote = self.root/'remote'
        self.token = 'a'*32
        self.source = self.local/'fusion-pilot-fixture'/self.token
        self.source.mkdir(parents=True)
        self.stop = self.root/'done'
        self.stop.touch()

    def run_mirror(self):
        return subprocess.run([sys.executable,str(Path(__file__).with_name('persist_gpu.py'))],
            env=dict(os.environ,VALIDATION_LOCAL_RESULTS=str(self.local),VALIDATION_PERSISTED_RESULTS=str(self.remote),
                VALIDATION_STOP_MARKER=str(self.stop)),capture_output=True,text=True)

    def complete(self, expected=None):
        payload = b'software fixture, not a model prediction'
        (self.source/'prediction.pkl').write_bytes(payload)
        (self.source/'complete.json').write_text(json.dumps(dict(token=self.token,raw_sha256=
            expected or hashlib.sha256(payload).hexdigest())))

    def test_incomplete_sample_is_not_published(self):
        (self.source/'prediction.pkl').write_bytes(b'incomplete')
        self.assertEqual(self.run_mirror().returncode,0)
        self.assertFalse((self.remote/'fusion'/self.token).exists())

    def test_complete_replay_is_idempotent(self):
        self.complete()
        self.assertEqual(self.run_mirror().returncode,0)
        self.assertEqual(self.run_mirror().returncode,0)
        self.assertEqual((self.source/'prediction.pkl').read_bytes(),
            (self.remote/'fusion'/self.token/'prediction.pkl').read_bytes())

    def test_checksum_failure_is_not_published(self):
        self.complete('wrong')
        self.assertNotEqual(self.run_mirror().returncode,0)
        self.assertFalse((self.remote/'fusion'/self.token).exists())

    def test_existing_correct_output_is_not_overwritten(self):
        self.complete()
        self.assertEqual(self.run_mirror().returncode,0)
        original = (self.remote/'fusion'/self.token/'prediction.pkl').read_bytes()
        (self.source/'complete.json').write_text('{"token":"different"}')
        self.assertNotEqual(self.run_mirror().returncode,0)
        self.assertEqual((self.remote/'fusion'/self.token/'prediction.pkl').read_bytes(),original)


if __name__ == '__main__':
    unittest.main(verbosity=2)
