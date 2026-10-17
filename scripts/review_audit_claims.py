#!/usr/bin/env python3
"""Class-specific proxy support and numerical-quality diagnostics for research review."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metadata_audit.analyze import support,save_json
p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args();out=Path(a.output)
cfg=json.loads((out/'audit_config.json').read_text())
f=pd.read_parquet(out/'tables/objects.parquet',columns=['annotation_token','instance_token','scene_token','category','width','relative_lateral_velocity','object_speed','ego_speed','distance_xy','velocity_span_s','velocity_method','projected_width_px','projection_valid','partially_outside_image'])
rows=[]
for dt in cfg['temporal_perturbations_s']:
    ratio=f.relative_lateral_velocity*dt/f.width
    bins=pd.cut(ratio,[0,.1,.5,1,np.inf],right=False,labels=['r<0.1','0.1<=r<0.5','0.5<=r<1','r>=1'])
    for c,g in f.groupby('category',observed=True):
        for regime in bins.cat.categories:
            rows.append({'delta_t_s':dt,'category':c,'regime':regime,**support(g[bins.loc[g.index]==regime],cfg)})
pd.DataFrame(rows).to_csv(out/'mismatch_proxy/r_regime_by_class.csv',index=False)
# Descriptive high-speed flags, not automatic validity thresholds or deletions.
flags=[]
for threshold in [20,30,40]:
    for c,g in f[f.object_speed>threshold].groupby('category',observed=True):
        flags.append({'object_speed_gt_m_s':threshold,'category':c,**support(g,cfg)})
pd.DataFrame(flags).to_csv(out/'motion/high_speed_diagnostics.csv',index=False)
f.sort_values('object_speed',ascending=False).head(30).to_csv(out/'motion/high_speed_examples.csv',index=False)
projected=f[f.projection_valid]
geometry={
    'valid_front_projection_count':len(projected),
    'partial_boundary_count':int(projected.partially_outside_image.sum()),
    'projected_width_over_image_width_count':int((projected.projected_width_px>1600).sum()),
    'note':'Unclipped 3D-box envelopes can exceed the image width; not all projected widths are suitable observed object scales. 1600 is the metadata image width used by these nuScenes camera records; this diagnostic is not a rejection rule.'}
save_json(out/'geometry/projection_scale_review.json',geometry)
print(pd.DataFrame(rows).query("category == 'vehicle.car'").to_string(index=False))
print(json.dumps(geometry,indent=2))
