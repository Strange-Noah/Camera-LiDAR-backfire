"""Read-only bounded dependency/metadata preflight. No data acquisition or inference."""
import argparse,hashlib,importlib.util,json,platform,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--dataroot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise SystemExit('fresh output required')
r={'python':platform.python_version(),'platform':platform.system()+' '+platform.machine(),'free_bytes':shutil.disk_usage(a.output.parent).free,'imports':{n:bool(importlib.util.find_spec(n)) for n in ['numpy','scipy','matplotlib','cv2','torch','nuscenes','pyquaternion']},'official_files':{}}
for name in ['README.md','ocamo_drift_tracking.py','lto_drift_tracking.py','ocamo_monitoring.py','lto_monitoring.py','include/SGDSchaul.py','include/kernel_correlation.py','include/feature_extraction.py']:
 f=a.upstream/'python/inference'/name;r['official_files'][name]={'present':f.is_file(),'sha256':hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None}
r['carla_present']=(a.upstream/'example/carla_example/001.jpg').is_file() and (a.upstream/'example/carla_example/001.mat').is_file()
r['metadata']={n:(a.metadata/(n+'.json')).is_file() for n in ['sample','sample_data','sensor','calibrated_sensor','ego_pose','sample_annotation']}
if r['metadata']['sample_data']:
 rows=json.loads((a.metadata/'sample_data.json').read_text());chosen=rows[:128]
 r['sensor_coverage']={'scope':'first 128 sample_data records only; not exhaustive dataset availability','checked':len(chosen),'present':sum((a.dataroot/x['filename']).is_file() for x in chosen),'keyframe_records':sum(x['is_key_frame'] for x in chosen),'non_keyframe_records':sum(not x['is_key_frame'] for x in chosen),'timestamps_present':all(isinstance(x.get('timestamp'),int) for x in chosen)}
r['gates']={'A':'AUDITED_WITH_MISSING_DEPENDENCIES','B':'BLOCKED: official example data and runtime dependencies unavailable','C':'BLOCKED: genuine sensor observations not available at selected local root','D':'NOT_RUN: B/C gates not passed','E':'NOT_RUN: no oracle or real-estimate correction verified'}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['imports','carla_present','sensor_coverage','gates']},indent=2))
