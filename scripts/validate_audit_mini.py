#!/usr/bin/env python3
"""Independent reference comparisons against installed official devkit on mini."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points,BoxVisibility
p=argparse.ArgumentParser();p.add_argument('--tables',required=True);p.add_argument('--dataroot',default='/workspace/datasets/nuscenes');a=p.parse_args()
n=NuScenes(version='v1.0-mini',dataroot=a.dataroot,verbose=False)
o=pd.read_parquet(Path(a.tables)/'objects.parquet')
assert len(o)==len(n.sample_annotation) and o.annotation_token.nunique()==len(o)
assert o.scene_token.nunique()==10
errors=[]
for row in o.itertuples():
    ref=n.box_velocity(row.annotation_token)[:2]
    if row.velocity_valid:
        errors.append(float(np.max(abs(ref-[row.object_vx,row.object_vy]))))
        assert np.allclose(ref,[row.object_vx,row.object_vy],atol=1e-5,rtol=1e-5)
assert np.nanmax(errors)<1e-5
valid=o[o.relative_velocity_valid]
np.testing.assert_allclose(valid.relative_radial_velocity**2+valid.relative_lateral_velocity**2,valid.relative_speed**2,atol=1e-8)
proj=pd.read_parquet(Path(a.tables)/'camera_projections.parquet')
selected=proj[proj.projection_valid].groupby('camera_channel').head(20)
perrors=[]
for r in selected.itertuples():
    sd=n.get('sample',r.sample_token)['data'][r.camera_channel]
    _,boxes,k=n.get_sample_data(sd,box_vis_level=BoxVisibility.NONE,selected_anntokens=[r.annotation_token])
    uv=view_points(boxes[0].corners(),k,normalize=True)
    e=max(abs(np.ptp(uv[0])-r.projected_width_px),abs(np.ptp(uv[1])-r.projected_height_px))
    perrors.append(e);assert e<1e-7
# Stationary-map classes and speed outliers are diagnostics, not silently filtered.
static=o[o.category.isin(['movable_object.trafficcone','movable_object.barrier'])]
result={'status':'PASS','all_object_velocities_compared':len(errors),'max_velocity_absolute_error_m_s':max(errors),
'projection_reference_checks':len(perrors),'max_projection_error_px':max(perrors),
'relative_radial_lateral_identity_rows':len(valid),'static_class_speed_p95_m_s':float(static.object_speed.quantile(.95)),
'object_speed_max_m_s':float(o.object_speed.max()),'ego_speed_max_m_s':float(o.ego_speed.max()),
'notes':'Static-class motion and high-speed values are retained for annotation-noise scrutiny, not clamped. Independent synthetic tests cover time gaps, unequal spacing, quaternion transforms, wraparound and near-plane rejection.'}
(Path(a.tables).parent/'numerical_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
