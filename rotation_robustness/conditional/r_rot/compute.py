"""Compute geometric features only, without predictions or matching."""
import argparse,json,gzip,hashlib
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from nuscenes import NuScenes
from geometry import CHANNELS,choose_camera
p=argparse.ArgumentParser();p.add_argument('--objects',type=Path,required=True);p.add_argument('--dataroot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise ValueError('fresh output required')
objects=json.loads(a.objects.read_text());assert len(objects)==53043
n=NuScenes(version='v1.0-trainval',dataroot=str(a.dataroot),verbose=False);cache={};result=[]
for i,o in enumerate(objects):
    tok=o['sample_token']
    if tok not in cache:
        sample=n.get('sample',tok);cams=[]
        for ch in CHANNELS:
            sd=n.get('sample_data',sample['data'][ch]);cal=n.get('calibrated_sensor',sd['calibrated_sensor_token']);pose=n.get('ego_pose',sd['ego_pose_token'])
            rg=Quaternion(pose['rotation']).rotation_matrix;rc=Quaternion(cal['rotation']).rotation_matrix
            rot=rc.T@rg.T;translation=rc.T@(-rg.T@np.array(pose['translation'])-np.array(cal['translation']))
            cams.append((ch,rot,translation,np.array(cal['camera_intrinsic']),sd['width'],sd['height']))
        cache[tok]=cams
    box=n.get_box(o['annotation_token']);assert np.allclose(box.center,o['translation'],rtol=0,atol=1e-8)
    candidates=[]
    for ch,rot,t,k,w,h in cache[tok]:
        candidates.append(dict(channel=ch,center=(rot@box.center+t).tolist(),corners=(rot@box.corners()+t[:,None]).tolist(),intrinsic=k.tolist(),image_width=w,image_height=h))
    row=choose_camera(candidates);row.update(sample_token=tok,annotation_token=o['annotation_token']);result.append(row)
    if i%5000==0:print('objects',i,flush=True)
with gzip.open(a.output,'wt') as f:json.dump(result,f,allow_nan=False)
