"""CPU-only official val manifest and complete tar-member size audit; no payload extraction."""
import argparse,ast,concurrent.futures,gzip,hashlib,json,os,subprocess,tarfile,threading,time,urllib.request
from pathlib import Path

ROOT=Path('/workspace/datasets/nuscenes')
DEFAULT_OUT=Path('/workspace/results/validation-baseline-preparation')
VALROOT=Path('/workspace/datasets/nuscenes-validation')
ACQUIRED_BYTES=0
BASE='https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/'
CAMERAS=['CAM_FRONT','CAM_FRONT_RIGHT','CAM_FRONT_LEFT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
LOCK=threading.Lock()
BASES=[ROOT,Path('/workspace/datasets/nuscenes-trainval-smoke'),VALROOT]


def inventory(out):
    result={}
    for base in BASES:
        tick(out,'inventory',root=str(base))
        for folder,dirs,names in os.walk(base):
            dirs[:]=[d for d in dirs if d not in ['v1.0-trainval','v1.0-mini','maps']]
            for name in names:
                path=Path(folder)/name
                if not name.endswith(('.jpg','.bin')):continue
                relative=str(path.relative_to(base))
                result.setdefault(relative,[]).append(str(path))
    tick(out,'inventory_complete',files=len(result))
    return result


def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,separators=(',',':'))+'\n');temp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def tick(out,stage,**data):
    with LOCK:
        event=dict(stage=stage,time=time.time(),**data)
        dump(out/'progress.json',event)
        target=os.environ.get('VALIDATION_PROGRESS_FILE')
        if target:dump(Path(target),event)
        print(json.dumps(event),flush=True)


def stream_rows(name,predicate):
    import ijson
    with (ROOT/'v1.0-trainval'/(name+'.json')).open('rb') as stream:
        return [r for r in ijson.items(stream,'item',use_float=True) if predicate(r)]


def build(out):
    tick(out,'metadata_start')
    with open('/workspace/cache/bevfusion-runtime/runtime.tar.gz','rb') as raw, tarfile.open(fileobj=Reader(raw,out,'runtime-splits-source'),mode='r|gz') as archive:
        member=next(x for x in archive if x.name.endswith('/nuscenes/utils/splits.py'))
        split_source=archive.extractfile(member).read()
    definitions={}
    for item in ast.parse(split_source.decode()).body:
        if isinstance(item,ast.Assign) and isinstance(item.targets[0],ast.Name) and item.targets[0].id in ['val','mini_val','mini_train']:
            definitions[item.targets[0].id]=ast.literal_eval(item.value)
    assert len(definitions['val'])==150
    scenes=json.loads((ROOT/'v1.0-trainval/scene.json').read_text())
    chosen={x['token']:x for x in scenes if x['name'] in definitions['val']}
    assert len(chosen)==150
    samples=sorted(stream_rows('sample',lambda r:r['scene_token'] in chosen),key=lambda r:r['timestamp'])
    assert len(samples)==6019 and len({x['token'] for x in samples})==6019
    tokens={x['token'] for x in samples}
    sensors={x['token']:x for x in json.loads((ROOT/'v1.0-trainval/sensor.json').read_text())}
    calibs={x['token']:x for x in json.loads((ROOT/'v1.0-trainval/calibrated_sensor.json').read_text())}
    channel={t:sensors[c['sensor_token']]['channel'] for t,c in calibs.items()}
    sd=stream_rows('sample_data',lambda r:r['sample_token'] in tokens and channel[r['calibrated_sensor_token']] in CAMERAS+['LIDAR_TOP'])
    by_token={x['token']:x for x in sd}
    keyframes={}
    for row in sd:
        if row['is_key_frame']:
            key=(row['sample_token'],channel[row['calibrated_sensor_token']]);assert key not in keyframes;keyframes[key]=row
    entries={}
    def add(row,sample_token,clean):
        name=row['filename'];assert not Path(name).is_absolute() and '..' not in Path(name).parts
        entry=entries.setdefault(name,dict(relative_path=name,sensor_modality='lidar' if channel[row['calibrated_sensor_token']]=='LIDAR_TOP' else 'camera',
            channel=channel[row['calibrated_sensor_token']],sample_data_token=row['token'],timestamp=row['timestamp'],is_key_frame=row['is_key_frame'],
            referencing_sample_tokens=set(),scenarios=set(),archive_members=[],uncompressed_size_bytes=None,
            source_calibrated_sensor_token=row['calibrated_sensor_token'],source_ego_pose_token=row['ego_pose_token']))
        entry['referencing_sample_tokens'].add(sample_token);entry['scenarios'].add('B')
        if clean:entry['scenarios'].add('A')
        return name
    records=[]
    for sample in samples:
        token=sample['token'];cams=[add(keyframes[token,c],token,True) for c in CAMERAS]
        lidar=keyframes[token,'LIDAR_TOP'];current=add(lidar,token,True);sweeps=[]
        cursor=lidar
        for _ in range(10):
            if not cursor['prev']:break
            assert cursor['prev'] in by_token,'Missing previous-sweep metadata'
            cursor=by_token[cursor['prev']];sweeps.append(add(cursor,token,True))
        records.append(dict(token=token,scene=chosen[sample['scene_token']]['name'],scene_token=sample['scene_token'],timestamp=sample['timestamp'],
            camera_files=cams,lidar_keyframe=current,sweep_candidate_files=sweeps))
    # Scenario B is a declared superset: all recorded non-keyframe cameras in val scenes.
    for row in sd:
        if channel[row['calibrated_sensor_token']] in CAMERAS and not row['is_key_frame']:
            add(row,row['sample_token'],False)
    annotations=stream_rows('sample_annotation',lambda r:r['sample_token'] in tokens)
    ann_tokens={r['token'] for r in annotations};ann_counts={t:0 for t in tokens}
    for r in annotations:
        ann_counts[r['sample_token']]+=1
        assert all(not r[k] or r[k] in ann_tokens for k in ['prev','next']), 'Annotation chain leaves selected scenes'
    required_poses={x['source_ego_pose_token'] for x in entries.values()}
    present_poses={x['token'] for x in stream_rows('ego_pose',lambda r:r['token'] in required_poses)}
    assert required_poses==present_poses
    metadata={}
    for name in ['scene','sample','sample_data','sample_annotation','instance','category','attribute','visibility','sensor','calibrated_sensor','ego_pose','log','map']:
        path=ROOT/'v1.0-trainval'/(name+'.json');assert path.is_file()
        metadata[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
    for r in records:r['ground_truth_annotation_count']=ann_counts[r['token']]
    present=inventory(out)
    for e in entries.values():
        e['referencing_sample_tokens']=sorted(e['referencing_sample_tokens']);e['scenarios']=sorted(e['scenarios'])
        locations=present.get(e['relative_path'],[])
        e['availability_status']='present_unverified_size' if locations else 'missing'
        e['available_locations']=locations
    manifest=dict(schema_version=1,dataset_version='v1.0-trainval',split='val',expected_sample_count=6019,scene_count=150,
        official_splits_source_sha256=hashlib.sha256(split_source).hexdigest(),metadata=metadata,samples=records,
        sweep_policy=dict(converter_max_sweeps=10,pipeline_sweeps_num=9,pipeline_loader_test_mode=False,
            required_union='All ten possible candidates; official loader randomly selects nine when more than nine exist. No preprocessing switch is changed.'),
        scenario_B_policy='All recorded non-keyframe camera images in official val scenes; references identify original metadata sample associations, not a corruption mapping.',
        files=sorted(entries.values(),key=lambda e:e['relative_path']),archive_audit_status='not_scanned')
    dump(out/'validation-file-manifest.json',manifest)
    tick(out,'manifest_constructed',samples=6019,files=len(entries),clean_files=sum('A' in e['scenarios'] for e in entries.values()))
    return manifest


class Reader:
    def __init__(self,raw,out,name):self.raw,self.out,self.name,self.count,self.last=raw,out,name,0,time.monotonic()
    def read(self,size=-1):
        data=self.raw.read(size);self.count+=len(data)
        if time.monotonic()-self.last>20:
            tick(self.out,'archive_stream',archive=self.name,compressed_bytes=self.count);self.last=time.monotonic()
        return data


def scan_one(number,out,wanted,keyframes,present):
    global ACQUIRED_BYTES
    name='v1.0-trainval%02d_blobs.tgz'%number
    summary=out/'archives'/(name+'.summary.json');index=out/'archives'/(name+'.members.jsonl')
    if summary.exists() and index.exists():
        old=json.loads(summary.read_text())
        if old.get('status')=='passed':return old
    for attempt in range(3):
        try:
            tick(out,'archive_start',archive=name,attempt=attempt+1)
            start=time.time();count=0;total=0;matches=0;part=index.with_suffix('.jsonl.part')
            with urllib.request.urlopen(BASE+name,timeout=60) as response,part.open('w') as output, (out/'archives'/(name+'.keyframes.jsonl')).open('a') as acquired:
                expected=int(response.headers['Content-Length']);reader=Reader(response,out,name)
                with gzip.GzipFile(fileobj=reader,mode='rb') as gz:
                    with tarfile.open(fileobj=gz,mode='r|',bufsize=1024**2) as archive:
                        for member in archive:
                            if not member.isfile():continue
                            count+=1;total+=member.size
                            path=member.name[2:] if member.name.startswith('./') else member.name
                            if path in wanted:
                                output.write(json.dumps(dict(relative_path=path,uncompressed_size_bytes=member.size,archive=name))+'\n');matches+=1
                                if path in keyframes and path not in present:
                                    with LOCK:
                                        if ACQUIRED_BYTES+member.size>10_000_000_000:raise RuntimeError('CPU acquisition safety budget exceeded')
                                        ACQUIRED_BYTES+=member.size
                                    dest=VALROOT/path;dest.parent.mkdir(parents=True,exist_ok=True);temporary=dest.with_suffix(dest.suffix+'.part');digest=hashlib.sha256()
                                    with archive.extractfile(member) as source,temporary.open('wb') as target:
                                        for block in iter(lambda:source.read(1024**2),b''):digest.update(block);target.write(block)
                                    assert temporary.stat().st_size==member.size;temporary.replace(dest)
                                    present[path]=[str(dest)]
                                    acquired.write(json.dumps(dict(relative_path=path,bytes=member.size,sha256=digest.hexdigest(),archive=name))+'\n');acquired.flush()
                    # Read past tar end to validate gzip CRC/footer and HTTP byte count.
                    while gz.read(1024**2):pass
                assert reader.count==expected,(reader.count,expected)
            part.replace(index)
            result=dict(archive=name,url=BASE+name,status='passed',compressed_bytes=expected,
                uncompressed_member_bytes=total,regular_file_count=count,matched_files=matches,
                seconds=time.time()-start,gzip_crc='passed',completed_at_utc=time.time())
            dump(summary,result);tick(out,'archive_finished',**result);return result
        except Exception as exc:
            tick(out,'archive_retry',archive=name,error=str(exc),attempt=attempt+1)
            if attempt==2:raise


def finish(out,manifest,summaries):
    entries={e['relative_path']:e for e in manifest['files']}
    present=inventory(out)
    for e in entries.values():
        e['archive_members']=[];e['uncompressed_size_bytes']=None
        e['available_locations']=present.get(e['relative_path'],[])
    for summary in summaries:
        with (out/'archives'/(summary['archive']+'.members.jsonl')).open() as stream:
            for line in stream:
                row=json.loads(line);e=entries[row['relative_path']]
                if e['uncompressed_size_bytes'] is not None:assert e['uncompressed_size_bytes']==row['uncompressed_size_bytes']
                e['uncompressed_size_bytes']=row['uncompressed_size_bytes'];e['archive_members'].append(row['archive'])
    missing=[p for p,e in entries.items() if e['uncompressed_size_bytes'] is None]
    for e in entries.values():
        if e['available_locations']:
            assert e['uncompressed_size_bytes'] is not None
            for path in e['available_locations']:assert Path(path).stat().st_size==e['uncompressed_size_bytes'], 'Existing payload size mismatch: '+path
            e['availability_status']='present_size_verified'
    manifest.update(archive_audit_status='passed' if not missing else 'failed',archive_summaries=summaries,missing_archive_members=missing)
    dump(out/'validation-file-manifest.json',manifest)
    totals={}
    for scope in ['A','B']:
        selected=[e for e in entries.values() if scope in e['scenarios']]
        payload=sum(e['uncompressed_size_bytes'] or 0 for e in selected)
        present=sum(e['uncompressed_size_bytes'] or 0 for e in selected if e['available_locations'])
        totals[scope]=dict(files=len(selected),payload_bytes=payload,already_available_payload_bytes=present,
            additional_payload_bytes=payload-present,max_file_bytes=max(e['uncompressed_size_bytes'] or 0 for e in selected),
            missing_file_count=sum(not e['available_locations'] for e in selected))
    du=int(subprocess.check_output(['du','-sx','--block-size=1','/workspace'],text=True).split()[0])
    result=dict(status='passed' if not missing else 'failed',sample_count=6019,scenarios=totals,
        existing_volume_bytes=du,volume_capacity_gb=50,all_trainval_uncompressed_member_bytes=sum(x['uncompressed_member_bytes'] for x in summaries),
        manifest_sha256=sha(out/'validation-file-manifest.json'),missing_members=missing)
    dump(out/'storage-measurements.json',result);tick(out,'archive_audit_complete',**result)
    assert not missing, 'Some required files absent from all official archives'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=DEFAULT_OUT);parser.add_argument('--workers',type=int,default=2);parser.add_argument('--acquire-keyframe-lidar',action='store_true')
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'archives').mkdir(exist_ok=True)
    path=a.output/'validation-file-manifest.json'
    manifest=json.loads(path.read_text()) if path.exists() else build(a.output)
    wanted={e['relative_path'] for e in manifest['files']}
    keyframes={r['lidar_keyframe'] for r in manifest['samples']} if a.acquire_keyframe_lidar else set()
    present=inventory(a.output)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        summaries=list(pool.map(lambda number:scan_one(number,a.output,wanted,keyframes,present),range(1,11)))
    finish(a.output,manifest,summaries)
