#!/usr/bin/env python3
"""Acceptance checks on a completed metadata audit, independent of plotting."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();out=Path(a.output);t=out/'tables'
run=json.loads((out/'RUN_COMPLETE.json').read_text());g=json.loads((out/'gating_report.json').read_text())
o=pd.read_parquet(t/'objects.parquet',columns=['annotation_token','instance_token','scene_token','sample_token','timestamp','velocity_valid','object_speed','relative_velocity_valid','relative_speed','relative_radial_velocity','relative_lateral_velocity','projection_valid','projected_width_px','projected_height_px','width','distance_xy'])
tr=pd.read_parquet(t/'trajectories.parquet');s=pd.read_parquet(t/'scenes.parquet');tm=pd.read_parquet(t/'sensor_timing.parquet')
assert len(o)==o.annotation_token.nunique()==s.n_annotations.sum()==tr.n_annotations.sum()
assert len(tr)==o.instance_token.nunique()==tr.instance_token.nunique()
assert len(s)==o.scene_token.nunique()==s.scene_token.nunique()
assert (tr.n_scenes==1).all()
assert (o.groupby('instance_token').scene_token.nunique()==1).all()
assert len(tm)==6*s.n_samples.sum()
assert tm[['sample_token','camera_channel']].duplicated().sum()==0
assert (o.velocity_valid==o.object_speed.notna()).all()
assert not o.loc[~o.projection_valid,['projected_width_px','projected_height_px']].notna().any().any()
assert (o.loc[o.projection_valid,'projected_width_px']>0).all()
v=o[o.relative_velocity_valid]
np.testing.assert_allclose(v.relative_radial_velocity**2+v.relative_lateral_velocity**2,v.relative_speed**2,atol=1e-7)
assert pq.ParquetFile(t/'camera_projections.parquet').metadata.num_rows==6*len(o)
assert pq.ParquetFile(out/'mismatch_proxy/mismatch_proxy.parquet').metadata.num_rows==4*len(o)
construct=pd.read_csv(out/'gates/constructability.csv')
for r in construct.itertuples():
    expected=len(tm) if r.population!='annotations' else len(o)
    assert r.valid_count+r.missing_count==expected
for path in (out/'joint').glob('*_coverage.csv'):
    if path.name=='all_joint_coverage.csv':continue
    f=pd.read_csv(path)
    assert (f.N_scene<=f.N_traj).all() and (f.N_traj<=f.N_ann).all()
    assert (f.N_ann>=0).all()
proxy=pd.read_csv(out/'mismatch_proxy/r_regime_coverage.csv')
for dt,f in proxy.groupby('delta_t_s'):
    valid=(o.relative_lateral_velocity.notna() & (o.width>0)).sum()
    assert f.N_ann.sum()==valid
if run['version']=='v1.0-trainval':assert len(s)==850 and g['final_gating_eligible']
else:assert not g['final_gating_eligible'] and g['overall'].startswith('NOT ASSESSED')
required=['motion/object_speed.png','motion/ego_speed.png','motion/ego_yaw_rate.png','motion/relative_speed.png','motion/radial_velocity.png','motion/lateral_velocity.png','motion/motion_by_class.png','motion/motion_coverage.csv','motion/motion_summary.json','geometry/distance.png','geometry/distance_by_class.png','geometry/physical_size_by_class.png','geometry/relative_orientation.png','geometry/projected_width.png','geometry/geometry_coverage.csv','geometry/geometry_summary.json','visibility/visibility_distribution.png','visibility/visibility_by_distance.png','visibility/lidar_points.png','visibility/lidar_points_vs_distance.png','visibility/lidar_points_by_class_distance.csv','visibility/camera_observability.csv','visibility/visibility_summary.json','independence/trajectory_length.png','independence/trajectory_duration.png','independence/annotations_per_trajectory.png','independence/instances_per_scene.png','independence/observations_per_scene.png','independence/regime_concentration.csv','independence/trajectory_support.csv','independence/independence_summary.json','timing/native_camera_lidar_offset.png','timing/absolute_offset.png','timing/offset_by_camera.png','timing/offset_by_scene.csv','timing/camera_sampling_interval.png','timing/lidar_sampling_interval.png','timing/timing_summary.json','identifiability/spearman_matrix.csv','identifiability/class_conditioned_statistics.csv','identifiability/scene_concentration.csv','identifiability/matched_support_summary.csv','identifiability/identifiability_summary.json','mismatch_proxy/mismatch_summary.json','README.md']
for name in required:assert (out/name).stat().st_size>0,name
result={'status':'PASS','version':run['version'],'n_annotations':len(o),'n_trajectories':len(tr),'n_scenes':len(s),'n_sensor_pairs':len(tm),'n_camera_projection_rows':pq.ParquetFile(t/'camera_projections.parquet').metadata.num_rows,'n_required_named_outputs':len(required),'no_model_or_blob_requirement':True}
(out/'acceptance_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
