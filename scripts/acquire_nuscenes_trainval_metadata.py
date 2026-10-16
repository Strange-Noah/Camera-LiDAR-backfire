#!/usr/bin/env python3
"""Explicit authorized metadata-only trainval acquisition; never sensor blobs."""
import datetime,hashlib,json,platform,subprocess,tarfile
from pathlib import Path
URL='https://d36yt3mvayqw5m.cloudfront.net/public/v1.0/v1.0-trainval_meta.tgz'
base=Path('/workspace');root=base/'datasets/nuscenes';stage=base/'staging'
if platform.system()!='Linux' or not base.is_mount():raise SystemExit('Requires mounted Runpod persistent volume')
if (root/'v1.0-trainval').exists():raise SystemExit('Trainval metadata exists; refusing overwrite')
archive=stage/'v1.0-trainval_meta.tgz';partial=archive.with_suffix('.tgz.part')
started=datetime.datetime.now(datetime.timezone.utc).isoformat()
if not archive.exists():
    subprocess.run(['curl','--fail','--location','--proto','=https','--proto-redir','=https','--retry','3','--continue-at','-','--max-filesize','1500000000','--output',str(partial),URL],check=True)
    partial.rename(archive)
subprocess.run(['gzip','-t',str(archive)],check=True)
h=hashlib.sha256()
with archive.open('rb') as f:
    for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
with tarfile.open(archive,'r:gz') as tar:
    members=tar.getmembers()
    for m in members:
        p=Path(m.name)
        if p.is_absolute() or '..' in p.parts or not(m.isfile() or m.isdir()):raise SystemExit('Unsafe archive member')
        if p.parts and p.parts[0] not in ['v1.0-trainval','maps','.v1.0-trainval_meta.txt','LICENSE']:raise SystemExit('Unexpected member outside metadata/maps/license; refusing extraction')
    selected=[m for m in members if Path(m.name).parts and Path(m.name).parts[0]=='v1.0-trainval']
    if not selected:raise SystemExit('No trainval metadata in archive')
    tar.extractall(root,members=selected,filter='data')
for m in selected:
    if m.isfile():assert (root/m.name).stat().st_size==m.size
assert len(json.loads((root/'v1.0-trainval/scene.json').read_text()))==850
manifest={'dataset':'nuScenes','version':'v1.0-trainval','component':'metadata only','source_page':'https://www.nuscenes.org/nuscenes#download','source_url':URL,'authorization':'Official Trainval Metadata URL supplied by user in this session','download_started_utc':started,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_name':archive.name,'archive_bytes':archive.stat().st_size,'sha256':h.hexdigest(),'official_checksum':None,'gzip_integrity':'PASS','member_sizes':'PASS','archive_retained':True,'extracted_files':sum(m.isfile() for m in selected),'extracted_bytes':sum(m.size for m in selected if m.isfile()),'maps_note':'Existing mini maps untouched; this audit needs no raster maps or sensor blobs','n_scenes':850}
(root/'trainval_metadata_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
