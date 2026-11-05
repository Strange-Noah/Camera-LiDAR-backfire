"""Read frozen GT dimensions/orientation only; never rematch or project boxes."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path


def transverse_width(size, quaternion, displacement):
    # nuScenes size=[width,length,height]; local x=length, y=width, z=height.
    w, x, y, z = quaternion
    norm = math.sqrt(w*w+x*x+y*y+z*z)
    w, x, y, z = [v/norm for v in [w,x,y,z]]
    rotation = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    dx, dy = displacement
    distance = math.hypot(dx, dy)
    if distance <= 0: raise ValueError('zero ground-plane ray')
    u = [-dy/distance, dx/distance, 0]
    lengths = [size[1], size[0], size[2]]
    return sum(lengths[j]*abs(sum(u[i]*rotation[i][j] for i in range(3))) for j in range(3))


def extract(objects, metadata):
    metadata = Path(metadata)
    needed_annotations = {o['annotation_token'] for o in objects}
    needed_samples = {o['sample_token'] for o in objects}
    proof = {}
    def load(name, needed=None, field='token'):
        p = metadata/(name+'.json'); proof[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        values = json.loads(p.read_text())
        return {v[field]:v for v in values if needed is None or v[field] in needed}
    anns = load('sample_annotation', needed_annotations)
    samples = load('sample', needed_samples)
    # Raw nuScenes sample tables do not have derived data[] maps: identify keyframe LiDAR records.
    sensors = load('sensor'); cals = load('calibrated_sensor')
    lidar_cals = {k for k,v in cals.items() if sensors[v['sensor_token']]['channel']=='LIDAR_TOP'}
    p=metadata/'sample_data.json';proof[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    rows=json.loads(p.read_text())
    lidar = {v['sample_token']:v for v in rows if v['sample_token'] in needed_samples and v['is_key_frame'] and v['calibrated_sensor_token'] in lidar_cals}
    assert len(lidar)==len(needed_samples)
    del rows
    poses=load('ego_pose',{v['ego_pose_token'] for v in lidar.values()})
    assert len(anns)==len(needed_annotations) and len(samples)==len(needed_samples)
    result=[]
    for o in objects:
        a=anns[o['annotation_token']];assert a['sample_token']==o['sample_token']
        assert max(abs(x-y) for x,y in zip(a['translation'],o['translation']))<1e-8
        assert a['instance_token']==o['instance_token']
        assert samples[o['sample_token']]['scene_token']==o['scene']
        pose=poses[lidar[o['sample_token']]['ego_pose_token']]
        ray=[a['translation'][i]-pose['translation'][i] for i in [0,1]]
        assert abs(math.hypot(*ray)-o['distance'])<1e-8
        assert all(math.isfinite(v) and v>0 for v in a['size'])
        width=transverse_width(a['size'],a['rotation'],ray)
        q=math.radians(5)*o['distance']/width
        assert math.isfinite(q) and q>=0 and width>0
        result.append(dict(sample_token=o['sample_token'],annotation_token=o['annotation_token'],physical_transverse_width=width,q_rot5=q))
    return result,proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--objects',required=True);p.add_argument('--metadata',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(exist_ok=False)
    result,proof=extract(json.loads(Path(a.objects).read_text()),a.metadata)
    with gzip.open(out/'physical.json.gz','wt') as f:json.dump(result,f,allow_nan=False)
    (out/'source_hashes.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps(dict(objects=len(result),physical_sha256=hashlib.sha256((out/'physical.json.gz').read_bytes()).hexdigest())))
