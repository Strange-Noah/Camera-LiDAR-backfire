"""Small nuScenes geometry adapter; faithful SGDSchaul and Gaussian-sum objective.

Only sensor projection/constants and ring-aware input ordering are adapted.
No GT, detection outputs, observation weights or optimizer edits are used.
"""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import cv2


def rigid(record):
    w,x,y,z=record['rotation'];m=np.eye(4);m[:3,:3]=Rotation.from_quat([x,y,z,w]).as_matrix();m[:3,3]=record['translation'];return m


def metadata(root,scene_count):
    folder=root/'v1.0-trainval'
    def load(name):return json.loads((folder/(name+'.json')).read_text())
    samples=load('sample');chosen=sorted({s['scene_token'] for s in samples})[:scene_count];tokens={s['token'] for s in samples if s['scene_token'] in chosen};scene_by_sample={s['token']:s['scene_token'] for s in samples if s['token'] in tokens}
    sensor={r['token']:r for r in load('sensor')};cal={r['token']:r for r in load('calibrated_sensor')}
    records=load('sample_data');selected=[r for r in records if r['sample_token'] in tokens and r['is_key_frame'] and sensor[cal[r['calibrated_sensor_token']]['sensor_token']]['channel'] in ['CAM_FRONT','LIDAR_TOP']];del records
    needed={r['ego_pose_token'] for r in selected};poses={r['token']:r for r in load('ego_pose') if r['token'] in needed}
    pairs={}
    for r in selected:pairs.setdefault(r['sample_token'],{})[sensor[cal[r['calibrated_sensor_token']]['sensor_token']]['channel']]=r
    result=[]
    for scene in chosen:
        rows=[]
        for token,pair in pairs.items():
            if scene_by_sample[token]!=scene or set(pair)!=set(['CAM_FRONT','LIDAR_TOP']):continue
            cam,lid=pair['CAM_FRONT'],pair['LIDAR_TOP'];wc=rigid(poses[cam['ego_pose_token']])@rigid(cal[cam['calibrated_sensor_token']]);wl=rigid(poses[lid['ego_pose_token']])@rigid(cal[lid['calibrated_sensor_token']]);cl=np.linalg.inv(wc)@wl
            rows.append(dict(sample=token,camera=cam,lidar=lid,T_C_L=cl,K=np.array(cal[cam['calibrated_sensor_token']]['camera_intrinsic']),T_W_C=wc,T_W_L=wl))
        rows.sort(key=lambda r:r['camera']['timestamp']);assert all(rows[i]['camera']['timestamp']<rows[i+1]['camera']['timestamp'] for i in range(len(rows)-1));result.append((scene,rows))
    return result


def features(points,get_corners):
    ring=points[:,4];assert np.allclose(ring,np.round(ring)) and ring.min()>=0 and ring.max()<32
    output=[];indices=[]
    for value in np.unique(ring):
        ids=np.flatnonzero(ring==value);ids=ids[np.argsort(np.arctan2(points[ids,1],points[ids,0]),kind='stable')]
        if len(ids)<11:continue
        # Preserve official extraction inside each ring; never convolve across rings/seams.
        tagged=np.column_stack([points[ids],np.arange(len(ids))]);corners=get_corners(tagged);local=corners[:,-1].astype(int);local=local[(local>=5)&(local<len(ids)-5)];indices.extend(ids[local].tolist())
    ids=np.array(sorted(set(indices)),dtype=int);return points[ids,:3],ids


def main(a):
    src=a.upstream.resolve();sys.path.insert(0,str(src/'python/inference'))
    from include.feature_extraction import get_corners
    import include.SGDSchaul as optimizer
    assert __import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=src,text=True).strip()=='1ac55de3c0eabb02f155cd44f7481117536c9526'
    out=a.output;out.mkdir(parents=True,exist_ok=False);all_rows=[];summaries=[];checks=[];scene_ids=[];start=time.monotonic()
    for scene,observations in metadata(a.dataroot,a.scenes):
        scene_ids.append(scene);states={};sequence=[];first_timestamp=None
        for i,obs in enumerate(observations):
            begin=time.monotonic();cam,lid=obs['camera'],obs['lidar'];imagepath=a.dataroot/cam['filename'];lidarpath=a.dataroot/lid['filename']
            if not imagepath.is_file() or not lidarpath.is_file():sequence.append(dict(index=i,status='MISSING_SENSOR'));continue
            image=cv2.imread(str(imagepath));assert image is not None;image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);height,width=image.shape[:2];assert (width,height)==(cam['width'],cam['height'])
            edges=cv2.Canny(image,50,100);y,x=np.where(edges);keep=y>height/3;tree=KDTree(np.column_stack([x[keep],y[keep]]));assert len(x[keep])>=10
            points=np.fromfile(lidarpath,dtype=np.float32).reshape(-1,5).astype(float);corners,ids=features(points,get_corners);assert len(corners)>0
            cl=obs['T_C_L'];lc=np.linalg.inv(cl);k=obs['K'];translation=lc[:3,3]
            homo=np.column_stack([corners,np.ones(len(corners))]);world_path=np.linalg.inv(obs['T_W_C'])@(obs['T_W_L']@homo.T);direct=cl@homo.T;error=float(np.max(np.abs(world_path-direct)));assert error<1e-9
            if first_timestamp is None:first_timestamp=cam['timestamp']
            if i==0:
                import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
                pc=direct[:3];good=pc[2]>0;uv=k@pc[:,good];uv=uv[:2]/uv[2];fig,ax=plt.subplots(figsize=(12,6));ax.imshow(image);ax.scatter(x[keep][::10],y[keep][::10],s=.3,c='lime');ax.scatter(uv[0],uv[1],s=3,c='red');ax.set(xlim=(0,width),ylim=(height,0),title='Clean projection: ring-aware LiDAR corners / original Canny edges');fig.tight_layout();fig.savefig(out/(scene+'-projection.png'),dpi=130);plt.close(fig)
            for seed in a.seeds:
                if seed not in states:
                    derived=int.from_bytes(hashlib.sha256((scene+':CAM_FRONT:'+str(seed)).encode()).digest()[:8],'little');omega=np.random.default_rng(derived).normal(0,np.deg2rad(a.sigma),3);q=Rotation.from_rotvec(omega).as_matrix();states[seed]=dict(q=q,tracker=optimizer.SGDSchaul(np.array([0,1,2])),initial=float(np.linalg.norm(omega)*180/np.pi),history=0)
                state=states[seed];q=state['q'];corrupt=lc.copy();corrupt[:3,:3]=lc[:3,:3]@q;oracle=np.eye(4);oracle[:3,:3]=q.T;assert np.max(np.abs(corrupt@oracle-lc))<1e-9
                # Original rotation parameter axes remain LiDAR axes. Center input at camera
                # origin so pure rotation correction leaves the camera->LiDAR translation fixed.
                centered=corners-translation;rc=corrupt[:3,:3]
                def objective(kdtree,pcl,params,sigma,neighbors):
                    pc=rc.T@Rotation.from_rotvec(params[:3]).as_matrix()@pcl[:,:3].T;uv=k@pc;uv=uv[:2,uv[2]>0]/uv[2,uv[2]>0];inside=(uv[0]>=0)&(uv[0]<width)&(uv[1]>=0)&(uv[1]<height);distance,_=kdtree.query(uv[:,inside].T,k=neighbors);return -np.exp(-distance**2/(2*sigma**2)).sum()
                optimizer.estimate_kernel_correlation=objective;tracker=state['tracker']
                if state['history']==0:
                    theta_oracle=Rotation.from_matrix(rc@q@rc.T).as_rotvec()
                    grid=[float(objective(tree,centered,theta_oracle+sign*np.eye(3)[axis]*np.deg2rad(.25),9,10)) for axis in range(3) for sign in [-1,1]]
                    assert np.isfinite(grid).all() and np.ptp(grid)>1e-6,'Uninformative local calibration objective'
                    checks.append(dict(scene=scene,seed=seed,local_loss_grid=grid,projection_composition_max_abs=error,oracle_matrix_max_abs=float(np.max(np.abs(corrupt@oracle-lc))),loss_corrupt=float(objective(tree,centered,np.zeros(3),9,10)),loss_clean_oracle=float(objective(tree,centered,theta_oracle,9,10))))
                tracker.update(centered,tree);state['history']+=1;assert np.isfinite(tracker.theta_).all()
                estimated=corrupt.copy();estimated[:3,:3]=Rotation.from_rotvec(tracker.theta_).as_matrix().T@rc
                correction=np.eye(4);correction[:3,:3]=rc.T@estimated[:3,:3];assert np.max(np.abs(corrupt@correction-estimated))<1e-9;assert np.array_equal(estimated[:3,3],lc[:3,3])
                residual=float(Rotation.from_matrix(lc[:3,:3].T@estimated[:3,:3]).magnitude()*180/np.pi)
                record=dict(scene=scene,sample=obs['sample'],camera='CAM_FRONT',timestamp=cam['timestamp'],lidar_timestamp=lid['timestamp'],frame_index=i,history_length=state['history'],seed=seed,initial_error_degrees=state['initial'],residual_error_degrees=residual,theta_radians=tracker.theta_.tolist(),features=len(corners),edges=len(x[keep]),elapsed_acquisition_seconds=(cam['timestamp']-first_timestamp)/1e6,frame_elapsed_seconds=time.monotonic()-begin,initial_corrupted_camera_to_lidar=corrupt.tolist(),right_correction=correction.tolist(),corrected_camera_to_lidar=estimated.tolist(),convention='column vectors; corrupted C2L rotation=clean_R@Q, translation fixed; corrected=corrupted@right_correction',upstream_commit='1ac55de3c0eabb02f155cd44f7481117536c9526')
                all_rows.append(record)
            print('NUSEQUENCE',scene,i,'corners',len(corners),flush=True)
        for seed,state in states.items():
            rows=[r for r in all_rows if r['scene']==scene and r['seed']==seed];post=[r['residual_error_degrees'] for r in rows if r['history_length']>10];summaries.append(dict(scene=scene,seed=seed,observations=len(rows),initial_error_degrees=state['initial'],final_error_degrees=rows[-1]['residual_error_degrees'],post_burn_mean_error_degrees=float(np.mean(post)) if post else None,final_improved=rows[-1]['residual_error_degrees']<state['initial'],elapsed_acquisition_seconds=rows[-1]['elapsed_acquisition_seconds'],missing_frames=sum(x['status']=='MISSING_SENSOR' for x in sequence)))
    (out/'corrections.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in all_rows));result=dict(status='EXECUTED_NOT_AUTOMATICALLY_CONVERGED',sigma_degrees_per_axis=a.sigma,seeds=a.seeds,sequences=summaries,geometry_checks=checks,elapsed_seconds=time.monotonic()-start,feature_adaptation='within-ring stable azimuth ordering; official get_corners per ring; exclude five seam/end points',objective_adaptation='only real K/image bounds/extrinsic projection with centered LiDAR coordinates; original unnormalized Gaussian kNN sum, sigma9 k10; SGDSchaul source unchanged',cadence='genuine keyframes sorted by camera timestamps; not full-frequency sweeps')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,len(summaries)//len(a.seeds),figsize=(15,4),squeeze=False)
    for ax,scene in zip(axs.flat,scene_ids):
        for seed in a.seeds:
            rows=[r for r in all_rows if r['scene']==scene and r['seed']==seed];ax.plot([r['history_length'] for r in rows],[r['residual_error_degrees'] for r in rows],label='seed%d'%seed);ax.axhline(rows[0]['initial_error_degrees'],ls=':',lw=.7)
        ax.axvspan(0,10,color='gray',alpha=.15);ax.set(title=scene[:8],xlabel='Observation count',ylabel='Residual rotation (degrees)');ax.legend()
    fig.tight_layout();fig.savefig(out/'convergence.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--dataroot',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scenes',type=int,default=3);p.add_argument('--seeds',type=int,nargs='+',default=[0,1]);p.add_argument('--sigma',type=float,default=.5);main(p.parse_args())
