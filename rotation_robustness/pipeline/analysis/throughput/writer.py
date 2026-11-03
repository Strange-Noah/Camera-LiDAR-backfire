"""Bounded ordered CPU writer; completion/progress only after the original fsyncs."""
import copy
import json
import os
import tempfile
import pickle
import queue
import threading
import time
from pathlib import Path
from environment.bevfusion.experiments.experiment_manifest import (
    atomic_bytes, atomic_json, digest, validate_manifest,
)


class RunCache:
    def __init__(self, root, manifest, regime):
        # Validate the full immutable manifest ONCE, not 6019 times. Only this
        # private copy supplies condition IDs/tokens; workers receive other copies.
        self.manifest = copy.deepcopy(manifest)
        body = validate_manifest(self.manifest)
        self.tokens = tuple(body['subset']['sample_tokens'])
        self.cid = self.manifest['condition_id']
        self.regime = regime
        self.out = Path(root) / 'rotation-v1' / self.cid
        self.out.mkdir(parents=True, exist_ok=True)

    def directory(self, index, token):
        if index < 0 or index >= len(self.tokens) or self.tokens[index] != token:
            raise ValueError('Cache index/token/order mismatch')
        return self.out / token

    def load(self, index, token):
        path = self.directory(index, token)
        if not path.exists():
            return None
        # Never import a truncated last token or silently change an execution regime.
        record = json.loads((path / 'complete.json').read_text())
        if (record['status'] != 'complete' or record['token'] != token
                or record['condition_id'] != self.cid
                or record.get('execution_regime_sha256') != self.regime
                or digest(path/'prediction.pkl') != record['sha256']):
            raise ValueError('Invalid, corrupt or foreign-regime cache')
        with (path/'prediction.pkl').open('rb') as f:
            return pickle.load(f)


class OrderedWriter:
    def __init__(self, cache, depth=2):
        if depth < 0:
            raise ValueError('Negative queue depth')
        self.cache = cache
        self.queue = queue.Queue(maxsize=depth) if depth else None
        self.error = None
        self.next_index = 0
        self.times = []
        self.closed = False
        self.thread = None
        if self.queue is not None:
            self.thread = threading.Thread(target=self._loop, name='prediction-writer', daemon=True)
            self.thread.start()

    def check(self):
        if self.error is not None:
            raise RuntimeError('Prediction writer failed; no complete handoff allowed') from self.error

    def _commit(self, index, token, result, reused):
        start = time.perf_counter()
        if index != self.next_index:
            raise ValueError('Nonsequential commit')
        path = self.cache.directory(index, token)
        serialize = 0.
        if not reused:
            if path.exists():
                raise ValueError('Refusing to overwrite any existing token directory')
            then = time.perf_counter()
            payload = pickle.dumps(result, protocol=4)
            serialize = time.perf_counter()-then
            # Publish a whole committed token directory. Interrupted writes stay
            # in an isolated staging directory and never block valid-token resume.
            staging = Path(tempfile.mkdtemp(prefix='.pending-token-', dir=self.cache.out))
            atomic_bytes(staging/'prediction.pkl', payload)
            atomic_json(staging/'complete.json', dict(status='complete', condition_id=self.cache.cid,
                token=token, sha256=digest(staging/'prediction.pkl'),
                execution_regime_sha256=self.cache.regime))
            os.rename(staging, path)
            descriptor = os.open(str(self.cache.out), os.O_RDONLY)
            try: os.fsync(descriptor)
            finally: os.close(descriptor)
        atomic_json(self.cache.out/'progress.json', dict(condition_id=self.cache.cid,
            completed=index+1, total=len(self.cache.tokens), time=time.time(),
            execution_regime_sha256=self.cache.regime))
        self.next_index += 1
        self.times.append(dict(index=index, serialization_seconds=serialize,
                               commit_seconds=time.perf_counter()-start, reused=reused))

    def _loop(self):
        while True:
            task = self.queue.get()
            try:
                if task is None:
                    return
                if self.error is None:
                    self._commit(*task)
            except BaseException as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def submit(self, index, token, result, reused=False):
        if self.closed:
            raise RuntimeError('Writer already closed')
        self.check()
        start = time.perf_counter()
        # CPU-only deep snapshot before enqueue: later code cannot mutate data
        # still being serialized. This never clones/changes model-facing inputs.
        snapshot = copy.deepcopy(result) if self.queue is not None and not reused else result
        snapshot_seconds = time.perf_counter()-start
        before = time.perf_counter()
        if self.queue is None:
            self._commit(index, token, snapshot, reused)
        else:
            self.queue.put((index, token, snapshot, reused))
        self.check()
        return dict(snapshot_seconds=snapshot_seconds,
                    writer_queue_wait_seconds=time.perf_counter()-before)

    def finish(self):
        if self.closed:
            self.check()
            return
        self.closed = True
        if self.queue is not None:
            self.queue.put(None)
            self.queue.join()
            self.thread.join()
        self.check()
