"""Stream official JSON into a local SQLite index, avoiding full-devkit RAM use."""
import hashlib
import json
import sqlite3
from pathlib import Path
import ijson

TABLES = ['category', 'visibility', 'instance', 'sensor', 'calibrated_sensor', 'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation']


class Store:
    def __init__(self, dataroot, version, cache):
        self.root = Path(dataroot) / version
        self.path = Path(cache) / (version + '.sqlite')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA cache_size=-65536')
        self.db.execute('CREATE TABLE IF NOT EXISTS manifest (name TEXT PRIMARY KEY, fingerprint TEXT)')
        for table in TABLES:
            path = self.root / (table + '.json')
            fingerprint = f'{path.resolve()}:{path.stat().st_size}:{path.stat().st_mtime_ns}'
            old = self.db.execute('SELECT fingerprint FROM manifest WHERE name=?', (table,)).fetchone()
            if old and old[0] == fingerprint:
                continue
            print('Indexing', table, flush=True)
            self.db.execute(f'DROP TABLE IF EXISTS {table}')
            self.db.execute(f'CREATE TABLE {table} (token TEXT PRIMARY KEY, parent TEXT, payload TEXT)')
            parent_key = {'sample': 'scene_token', 'sample_data': 'sample_token', 'sample_annotation': 'sample_token'}.get(table)
            with path.open('rb') as f:
                batch = []
                for row in ijson.items(f, 'item', use_float=True):
                    batch.append((row['token'], row.get(parent_key, '') if parent_key else '', json.dumps(row, separators=(',', ':'))))
                    if len(batch) >= 5000:
                        self.db.executemany(f'INSERT INTO {table} VALUES (?,?,?)', batch)
                        batch.clear()
                self.db.executemany(f'INSERT INTO {table} VALUES (?,?,?)', batch)
            self.db.execute(f'CREATE INDEX {table}_parent ON {table}(parent)')
            self.db.execute('INSERT OR REPLACE INTO manifest VALUES (?,?)', (table, fingerprint))
            self.db.commit()

    def get(self, table, token):
        row = self.db.execute(f'SELECT payload FROM {table} WHERE token=?', (token,)).fetchone()
        if row is None:
            raise ValueError(f'Missing {table}/{token}')
        return json.loads(row[0])

    def all(self, table):
        return [json.loads(r[0]) for r in self.db.execute(f'SELECT payload FROM {table}')]

    def children(self, table, parent):
        return [json.loads(r[0]) for r in self.db.execute(f'SELECT payload FROM {table} WHERE parent=?', (parent,))]

    def scene_rows(self, table, scene):
        return [json.loads(r[0]) for r in self.db.execute(f'SELECT t.payload FROM {table} t JOIN sample s ON t.parent=s.token WHERE s.parent=?', (scene,))]

    def source_manifest(self):
        rows = []
        for table in TABLES:
            path = self.root / (table + '.json')
            h = hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                    h.update(block)
            rows.append({'table': table, 'bytes': path.stat().st_size, 'sha256': h.hexdigest()})
        return rows
