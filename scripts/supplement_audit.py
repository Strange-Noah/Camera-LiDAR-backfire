#!/usr/bin/env python3
"""Additional ordinal visibility, camera cluster support and provenance checks."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
import ijson
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metadata_audit.analyze import support,save_json
from nuscenes.utils.splits import create_splits_scenes
p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--dataroot',default='/workspace/datasets/nuscenes');a=p.parse_args()
out=Path(a.output);cfg=json.loads((out/'audit_config.json').read_text());run=json.loads((out/'RUN_COMPLETE.json').read_text())
obj=pd.read_parquet(out/'tables/objects.parquet',columns=['annotation_token','instance_token','scene_token','category','visibility_token','distance_xy','object_speed','ego_speed','ego_yaw_rate','relative_lateral_velocity'])
spec=json.loads((out/'bin_definitions.json').read_text())['distance_xy']
obj['distance_regime']=pd.cut(obj.distance_xy,spec['edges'],labels=spec['labels'],include_lowest=True)
for groups,name in [(['category','visibility_token'],'visibility_by_class_counts'),(['distance_regime','visibility_token'],'visibility_by_distance_counts'),(['scene_token','visibility_token'],'visibility_by_scene_counts')]:
    rows=[]
    for keys,g in obj.groupby(groups,observed=True):rows.append({**dict(zip(groups,keys)),**support(g,cfg)})
    pd.DataFrame(rows).to_csv(out/f'visibility/{name}.csv',index=False)
# Distinct trajectory/scene sets are accumulated across batches; counts are never summed blindly.
sets=defaultdict(lambda: {'ann':set(),'traj':set(),'scene':set()})
cols=['annotation_token','instance_token','scene_token','camera_channel','projection_valid','partially_outside_image','near_plane_crossing']
for batch in pq.ParquetFile(out/'tables/camera_projections.parquet').iter_batches(batch_size=65536,columns=cols):
    f=batch.to_pandas()
    for channel,g in f.groupby('camera_channel'):
        for regime,h in [('all',g),('valid_projection',g[g.projection_valid]),('partial_boundary',g[g.partially_outside_image]),('near_plane_crossing',g[g.near_plane_crossing])]:
            bucket=sets[(channel,regime)]
            bucket['ann'].update(h.annotation_token);bucket['traj'].update(h.instance_token);bucket['scene'].update(h.scene_token)
rows=[{'camera_channel':c,'regime':r,'N_ann':len(s['ann']),'N_traj':len(s['traj']),'N_scene':len(s['scene'])} for (c,r),s in sets.items()]
pd.DataFrame(rows).to_csv(out/'visibility/camera_observability_independent_support.csv',index=False)
del sets
# Configured support sensitivity, not a score or an inferential significance test.
f=pd.read_csv(out/'independence/regime_concentration.csv');sensitivity=[]
for ns in [5,10,20]:
    for nt in [10,30,50]:
        for r in f.itertuples():
            sensitivity.append({'variable':r.variable,'regime':r.regime,'min_scenes':ns,'min_trajectories':nt,'max_top1_scene_fraction':.3,'supported':bool(r.N_scene>=ns and r.N_traj>=nt and r.top1_scene_fraction<=.3)})
pd.DataFrame(sensitivity).to_csv(out/'gates/support_threshold_sensitivity.csv',index=False)
source_counts={}
for table in ['scene','sample','instance','sample_annotation']:
    with (Path(a.dataroot)/run['version']/(table+'.json')).open('rb') as f:
        source_counts[table]=sum(1 for _ in ijson.items(f,'item',use_float=True))
scenes=pd.read_parquet(out/'tables/scenes.parquet');traj=pd.read_parquet(out/'tables/trajectories.parquet')
assert source_counts['scene']==len(scenes)
assert source_counts['sample']==scenes.n_samples.sum()
assert source_counts['instance']==len(traj)
assert source_counts['sample_annotation']==len(obj)
splits=create_splits_scenes()
if run['version']=='v1.0-trainval':
    assert set(scenes.scene_name)==set(splits['train'])|set(splits['val'])
    split_counts={'train':len(set(scenes.scene_name)&set(splits['train'])),'val':len(set(scenes.scene_name)&set(splits['val']))}
else:split_counts={'mini_train':len(set(scenes.scene_name)&set(splits['mini_train'])),'mini_val':len(set(scenes.scene_name)&set(splits['mini_val']))}
static=obj[obj.category.isin(['movable_object.trafficcone','movable_object.barrier'])]
checks={'status':'PASS','source_row_counts':source_counts,'official_split_membership':split_counts,
    'sanity_diagnostics_not_filters':{'static_class_object_speed_p95_m_s':float(static.object_speed.quantile(.95)),'object_speed_max_m_s':float(obj.object_speed.max()),'ego_speed_max_m_s':float(obj.ego_speed.max()),'abs_ego_yaw_rate_max_rad_s':float(obj.ego_yaw_rate.abs().max())},
    'metadata_only_feasibility':{'constructible':'All requested canonical metadata variables and geometric box projections; see missingness flags','requires_sensor_data':['Actual per-camera occlusion or pixel/object visibility','Image blur/exposure/content quality','Recomputed LiDAR return geometry and sensor noise','Any detector/fusion performance'], 'sensor_blob_download_needed_for_this_audit':False}}
save_json(out/'source_and_scope_validation.json',checks);print(json.dumps(checks,indent=2))
