#!/usr/bin/env python3
"""Record persistent artifact inventory and small validation evidence."""
import argparse,datetime,hashlib,json,shutil,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--archive-was-reused',action='store_true');a=p.parse_args();out=Path(a.output);v=out/'validation';v.mkdir(exist_ok=True)
if a.archive_was_reused:
    manifest=Path('/workspace/datasets/nuscenes/trainval_metadata_manifest.json')
    m=json.loads(manifest.read_text())
    if 'archive_reused' not in m:
        m['processing_started_utc']=m.pop('download_started_utc')
        m['download_started_utc']=None
        m['archive_reused']=True
        archive=Path('/workspace/staging')/m['archive_name']
        m['archive_download_completed_utc']=datetime.datetime.fromtimestamp(archive.stat().st_mtime,datetime.timezone.utc).isoformat()
        m['provenance_correction']='The successful extraction reused the already downloaded archive; the legacy started field measured processing, not the original download start.'
        manifest.write_text(json.dumps(m,indent=2)+'\n')
for source,name in [('/workspace/results/metadata_audit_mini_final/numerical_validation.json','mini_numerical_validation.json'),('/workspace/results/metadata_audit_mini_final/acceptance_checks.json','mini_acceptance_checks.json'),('/workspace/results/audit-unit-tests.log','unit-tests.log'),('/workspace/datasets/nuscenes/trainval_metadata_manifest.json','trainval_metadata_manifest.json')]:
    shutil.copy2(source,v/name)
shutil.copy2(out/'tables/construction_validation.json',v/'construction_validation.json')
(v/'pip-freeze.txt').write_text(subprocess.check_output(['/opt/venvs/nuscenes-metadata/bin/python','-m','pip','freeze'],text=True))
(v/'python-version.txt').write_text(subprocess.check_output(['/opt/venvs/nuscenes-metadata/bin/python','--version'],text=True))
files=[]
for path in sorted(out.rglob('*')):
    if not path.is_file() or path.name=='artifact_inventory.json':continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    relative=str(path.relative_to(out))
    only_persistent=relative.startswith('tables/') or path.suffix=='.parquet' or path.name in ['motion_by_trajectory.csv','trajectory_support.csv','trajectory_distance_range.csv']
    files.append({'relative_path':relative,'bytes':path.stat().st_size,'sha256':h.hexdigest(),'storage':'persistent volume only' if only_persistent else 'persistent volume and GitHub summary copy'})
(out/'artifact_inventory.json').write_text(json.dumps({'persistent_root':str(out),'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files},indent=2)+'\n')
print('Inventoried',len(files),'files; bytes',sum(f['bytes'] for f in files))
