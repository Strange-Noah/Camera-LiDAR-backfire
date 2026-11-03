"""Byte-preserving bounded shards. No model/pickle execution or scientific policy."""
import hashlib,json,os,shutil,stat,time
from pathlib import Path,PurePosixPath

VERSION='bounded-shards-v1'
def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def sync_dir(path):
    fd=os.open(str(path),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def atomic(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.part')
    if path.is_symlink() or tmp.is_symlink():raise ValueError('Symlink atomic destination')
    with tmp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);sync_dir(path.parent)
def atomic_json(path,value):atomic(path,encoded(value)+b'\n')
def relative(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or any(x in ('','.','..') for x in p.parts) or '\\' in name or str(p)!=name:
        raise ValueError('Unsafe archive path: '+name)
    return p

def read_index(root,verify_shards=True):
    root=Path(root);d=json.loads((root/'index.json').read_text())
    body={k:v for k,v in d.items() if k!='archive_id'}
    if d.get('version')!=VERSION or hashlib.sha256(encoded(body)).hexdigest()!=d.get('archive_id'):raise ValueError('Index identity')
    names=set();shards={}
    for row in d['shards']:
        name=row['name'];relative(name)
        if '/' in name or name in shards or row['bytes']>d['shard_bytes']:raise ValueError('Shard bound/identity')
        shards[name]=row
        p=root/name
        if p.is_symlink() or not p.is_file() or p.stat().st_size!=row['bytes']:raise ValueError('Missing/truncated shard')
        if verify_shards and sha(p)!=row['sha256']:raise ValueError('Corrupt shard')
    for row in d['files']:
        name=row['path'];relative(name)
        if name in names:raise ValueError('Duplicate archive file')
        names.add(name);total=0
        for chunk in row['chunks']:
            if chunk['shard'] not in shards or chunk['offset']<0 or chunk['bytes']<=0 or chunk['offset']+chunk['bytes']>shards[chunk['shard']]['bytes']:raise ValueError('Invalid fragment')
            total+=chunk['bytes']
        if total!=row['bytes']:raise ValueError('File size/fragment mismatch')
    dirs=[]
    for row in d['directories']:
        relative(row['path'])
        if row['path'] in names or row['path'] in dirs:raise ValueError('Directory/file conflict')
        dirs.append(row['path'])
    for name in names:
        if any(str(x) in names for x in PurePosixPath(name).parents if str(x)!='.'):raise ValueError('File ancestor conflict')
    return d

def pack_tree(source,stage,condition_id,complete=True,shard_bytes=64*1024**2,chunk_bytes=4*1024**2,fail_after_shards=None):
    source,stage=Path(source).resolve(),Path(stage).absolute()
    if source==stage or source in stage.parents:raise ValueError('Archive staging must be outside payload')
    if not 0<chunk_bytes<=shard_bytes:raise ValueError('Shard/chunk bounds')
    owner=dict(version=VERSION,condition_id=condition_id,source=str(source))
    if stage.is_symlink():raise ValueError('Symlink staging')
    if stage.exists():
        if (stage/'owner.json').is_symlink() or not (stage/'owner.json').is_file() or json.loads((stage/'owner.json').read_text())!=owner:raise ValueError('Foreign staging')
        if (stage/'index.json').exists():
            d=read_index(stage)
            if d['condition_id']!=condition_id or d['completeness']!=('complete' if complete else 'partial') or d['shard_bytes']!=shard_bytes:raise ValueError('Archive recovery contract mismatch')
            # Restart can reuse bytes only when the source is still identical.
            paths=list(source.rglob('*'))
            if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in paths):raise ValueError('Nonregular source on restart')
            actual={str(p.relative_to(source)):sha(p) for p in paths if p.is_file()}
            if actual!={f['path']:f['sha256'] for f in d['files']}:raise ValueError('Source changed after packaging')
            if {str(p.relative_to(source)) for p in paths if p.is_dir()}!={x['path'] for x in d['directories']}:raise ValueError('Source directories changed')
            return d
        shutil.rmtree(stage)
    stage.mkdir(parents=True);atomic_json(stage/'owner.json',owner)
    body=dict(version=VERSION,condition_id=condition_id,completeness='complete' if complete else 'partial',shard_bytes=shard_bytes,files=[],directories=[],shards=[])
    handle=None;written=0;h=None;name=None
    def close():
        nonlocal handle
        if handle is None:return
        handle.flush();os.fsync(handle.fileno());handle.close();os.replace(stage/(name+'.part'),stage/name)
        body['shards'].append(dict(name=name,bytes=written,sha256=h.hexdigest()));handle=None
        if fail_after_shards is not None and len(body['shards'])>=fail_after_shards:raise InterruptedError('Injected archive interruption')
    start=time.monotonic()
    try:
        for p in sorted(source.rglob('*')):
            rel=str(p.relative_to(source));relative(rel);mode=p.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):raise ValueError('Only regular files/directories')
            if stat.S_ISDIR(mode):body['directories'].append(dict(path=rel,mode=stat.S_IMODE(mode)));continue
            before=p.stat();file_hash=hashlib.sha256();fragments=[];size=0
            with p.open('rb') as f:
                while True:
                    if handle is not None and written==shard_bytes:close()
                    amount=min(chunk_bytes,shard_bytes-written) if handle else min(chunk_bytes,shard_bytes)
                    b=f.read(amount)
                    if not b:break
                    if handle is None:
                        name='shard-%05d.bin'%len(body['shards']);handle=(stage/(name+'.part')).open('wb');written=0;h=hashlib.sha256()
                    fragments.append(dict(shard=name,offset=written,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
                    handle.write(b);h.update(b);file_hash.update(b);written+=len(b);size+=len(b)
            after=p.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or size!=before.st_size:raise ValueError('Payload modified during packaging')
            body['files'].append(dict(path=rel,bytes=size,sha256=file_hash.hexdigest(),chunks=fragments,mode=stat.S_IMODE(mode)))
        close();sync_dir(stage)
    finally:
        if handle is not None:handle.close()
    body['archive_id']=hashlib.sha256(encoded(body)).hexdigest();atomic_json(stage/'index.json',body)
    atomic_json(stage/'pack-timing.json',dict(pack_seconds=time.monotonic()-start,files=len(body['files']),bytes=sum(x['bytes'] for x in body['files']),shards=len(body['shards'])))
    return body

def reconstruct(root,dest,require_complete=True,resume=False):
    root,dest=Path(root),Path(dest)
    d=read_index(root)
    if require_complete and d['completeness']!='complete':raise ValueError('Partial archive is not a full condition')
    if dest.exists() or dest.is_symlink():raise ValueError('Fresh reconstruction destination required')
    pending=dest.with_name(dest.name+'.reconstructing')
    dest.parent.mkdir(parents=True,exist_ok=True)
    owner=dest.with_name(dest.name+'.reconstruction-owner.json')
    expected=dict(archive_id=d['archive_id'],destination=str(dest.absolute()))
    if pending.is_symlink() or owner.is_symlink():raise ValueError('Symlink reconstruction state')
    if pending.exists():
        if not resume or not owner.is_file() or json.loads(owner.read_text())!=expected:raise ValueError('Explicit owned reconstruction recovery required')
        shutil.rmtree(pending)
    elif owner.exists() and json.loads(owner.read_text())!=expected:raise ValueError('Foreign reconstruction state')
    atomic_json(owner,expected)
    pending.mkdir(parents=True)
    try:
        for row in sorted(d['directories'],key=lambda x:len(PurePosixPath(x['path']).parts)):(pending/row['path']).mkdir(parents=True,exist_ok=True)
        for row in d['files']:
            out=pending/row['path'];out.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
            with out.open('xb') as f:
                for c in row['chunks']:
                    with (root/c['shard']).open('rb') as src:src.seek(c['offset']);b=src.read(c['bytes'])
                    if len(b)!=c['bytes'] or hashlib.sha256(b).hexdigest()!=c['sha256']:raise ValueError('Fragment hash mismatch')
                    f.write(b);h.update(b)
                f.flush();os.fsync(f.fileno())
            if h.hexdigest()!=row['sha256']:raise ValueError('Reconstructed file hash mismatch')
            out.chmod(row['mode'])
        for row in sorted(d['directories'],key=lambda x:len(PurePosixPath(x['path']).parts),reverse=True):sync_dir(pending/row['path']);(pending/row['path']).chmod(row['mode'])
        sync_dir(pending);os.replace(pending,dest);owner.unlink();sync_dir(dest.parent)
    except BaseException:
        # Keep the explicit incomplete namespace; never make it look committed.
        raise
    return d
