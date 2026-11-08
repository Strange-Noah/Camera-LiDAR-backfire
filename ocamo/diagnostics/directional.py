"""Read-only frozen-source verification and fixed 14-direction loss paths; no SGD."""
import argparse,ast,csv,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as Rot
import cv2


def setup(a):
    src=a.baseline/'ocamo/tools/nuscenes_minimal.py';assert hashlib.sha256(src.read_bytes()).hexdigest()=='4c851022ea71b06a50857ce26f32b15ff7d7a6188b29ffb89ec896f928225904';spec=importlib.util.spec_from_file_location('frozen_adapter',src);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    sys.path.insert(0,str(a.upstream/'python/inference'));from include.feature_extraction import get_corners
    node=next(n for n in ast.walk(ast.parse(src.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='objective');code=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'frozen_objective','exec')
    scenes=mod.metadata(a.dataroot,3)[1:];return mod,get_corners,code,scenes


def frame(a,mod,corners,obs):
    cam,lid=obs['camera'],obs['lidar'];image=cv2.cvtColor(cv2.imread(str(a.dataroot/cam['filename'])),cv2.COLOR_BGR2RGB);h,w=image.shape[:2]
    edge=cv2.Canny(image,50,100);y,x=np.where(edge);mask=y>h/3;tree=KDTree(np.column_stack([x[mask],y[mask]]));points=np.fromfile(a.dataroot/lid['filename'],np.float32).reshape(-1,5).astype(float);features,ids=mod.features(points,corners);lc=np.linalg.inv(obs['T_C_L']);p=features-lc[:3,3];return p,tree,lc,obs['K'],w,h,dict(points=len(points),corners=len(features),edges=int(mask.sum()),selected_per_ring=np.bincount(points[ids,4].astype(int),minlength=32).tolist(),radial_range_quantiles_m=np.quantile(np.linalg.norm(features,axis=1),[0,.25,.5,.75,1]).tolist(),image_wh=[w,h])


def reference(p,tree,r,k,w,h):
    pc=r.T@p.T;uv=k@pc;valid=uv[2]>0;uv=uv[:2,valid]/uv[2,valid];valid=(uv[0]>=0)&(uv[0]<w)&(uv[1]>=0)&(uv[1]<h);dist,_=tree.query(uv[:,valid].T,k=10);return -np.sum(np.exp(-dist**2/162.)),int(valid.sum()),pc


def review(a):
    mod,corners,code,scenes=setup(a);rows=[];geometry=[]
    for sid,(scene,obs) in enumerate(scenes,2):
        for seed in [0,1]:
            d=int.from_bytes(hashlib.sha256((scene+':CAM_FRONT:'+str(seed)).encode()).digest()[:8],'little');omega=np.random.default_rng(d).normal(0,np.deg2rad(.5),3);q=Rot.from_rotvec(omega).as_matrix();rotations=[np.linalg.inv(o['T_C_L'])[:3,:3] for o in obs];targets=np.array([r@q@r.T for r in rotations]);vectors=Rot.from_matrix(targets).as_rotvec();variation=Rot.from_matrix(targets@targets[0].T).magnitude()*180/np.pi
            pairwise=Rot.from_matrix((targets[:,None]@np.swapaxes(targets[None],-1,-2)).reshape(-1,3,3)).magnitude()*180/np.pi
            rows.append(dict(max_pairwise_target_change_degrees=float(pairwise.max()),scene_alias='scene-'+str(sid),seed=seed,observations=len(obs),injected_angle_degrees=float(np.linalg.norm(omega)*180/np.pi),max_target_geodesic_change_from_first_degrees=float(variation.max()),max_target_vector_change_from_first_degrees=float(np.max(np.linalg.norm(vectors-vectors[0],axis=1))*180/np.pi)))
        for index in [0,19,39]:
            o=obs[index];p,tree,lc,k,w,h,counts=frame(a,mod,corners,o);q=Rot.from_rotvec(np.deg2rad(.5)*np.array([1.,2.,3.])/np.sqrt(14)).as_matrix();rc=lc[:3,:3]@q;env=dict(np=np,Rotation=Rot,rc=rc,k=k,width=w,height=h);exec(code,env);theta=Rot.from_matrix(lc[:3,:3]@q@lc[:3,:3].T).as_rotvec();direct,n,pc=reference(p,tree,lc[:3,:3],k,w,h);adapter=float(env['objective'](tree,p,theta,9,10));err=float(np.max(np.abs(rc.T@Rot.from_rotvec(theta).as_matrix()@p.T-pc)));assert abs(adapter-direct)<1e-9 and err<1e-9
            geometry.append(dict(scene_alias='scene-'+str(sid),frame_index=index,adapter_reference_loss_difference=adapter-direct,oracle_projected_camera_points_max_abs=err,projected_corners=n,**counts))
    result=dict(status='NUMERIC_REVIEW_PASS_NOT_FULL_FEATURE_EQUIVALENCE',target_variation=rows,frames=geometry,limitations=['small existing projection overlays require visual review','ring/azimuth sorting and endpoint removal are adaptations, not proof of CARLA feature equivalence'])
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def direction(a):
    mod,corners,code,scenes=setup(a);dirs=[]
    for axis in range(3):
        for sign in [-1,1]:v=np.zeros(3);v[axis]=sign;dirs.append((('+' if sign>0 else '-')+'XYZ'[axis],v))
    import itertools
    for signs in itertools.product([-1,1],repeat=3):dirs.append(('diag'+''.join('+' if x>0 else '-' for x in signs),np.array(signs)/np.sqrt(3)))
    rows=[];paths=[];phi=np.deg2rad(.5)
    for sid,(scene,observations) in enumerate(scenes,2):
        for index in [0,19,39]:
            p,tree,lc,k,w,h,counts=frame(a,mod,corners,observations[index]);r=lc[:3,:3]
            for name,u in dirs:
                q=Rot.from_rotvec(phi*u).as_matrix();assert abs(Rot.from_matrix(q).magnitude()-phi)<1e-12;rc=r@q;env=dict(np=np,Rotation=Rot,rc=rc,k=k,width=w,height=h);exec(code,env);loss=[];numbers=[]
                for alpha in np.linspace(0,1,21):
                    theta=alpha*phi*(r@u);candidate=r@Rot.from_rotvec((1-alpha)*phi*u).as_matrix();expected,n,pc=reference(p,tree,candidate,k,w,h);actual=float(env['objective'](tree,p,theta,9,10));assert abs(actual-expected)<1e-9;assert np.max(np.abs(rc.T@Rot.from_rotvec(theta).as_matrix()@p.T-pc))<1e-9
                    loss.append(actual);numbers.append(n);paths.append(dict(scene='scene-'+str(sid),frame=index,direction=name,alpha=float(alpha),loss=actual,projected_corners=n,corners=counts['corners'],edges=counts['edges']))
                best=int(np.argmin(loss));rows.append(dict(scene='scene-'+str(sid),frame=index,direction=name,angle_degrees=.5,loss_corrupt=loss[0],loss_clean=loss[-1],clean_minus_corrupt=loss[-1]-loss[0],minimum_alpha=best/20,clean_preferred=loss[-1]<loss[0],minimum_at_clean=best==20,**counts))
            print('FRAME_COMPLETE',sid,index,flush=True)
    (a.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (a.output/'paths.csv').open('w') as f:wri=csv.DictWriter(f,fieldnames=list(paths[0]),lineterminator="\n");wri.writeheader();wri.writerows(paths)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    for scene in ['scene-2','scene-3']:
        fig,axs=plt.subplots(2,3,figsize=(15,8))
        for col,index in enumerate([0,19,39]):
            for name,_ in dirs:
                v=[x for x in paths if x['scene']==scene and x['frame']==index and x['direction']==name];ys=np.array([x['loss'] for x in v]);axs[0,col].plot([x['alpha'] for x in v],ys,label=name);axs[1,col].plot([x['alpha'] for x in v],ys-ys[-1])
            axs[0,col].set(title=scene+' frame '+str(index),ylabel='Original kernel loss');axs[1,col].axhline(0,color='black',lw=.7);axs[1,col].set(xlabel='α: corrupted0 → clean1',ylabel='Loss − clean loss')
        axs[0,2].legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(a.output/(scene+'-paths.png'),dpi=150);plt.close(fig)
    print('DIRECTION_COMPLETE',len(rows),'paths',len(paths),'evaluations',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['baseline','upstream','dataroot','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--phase',choices=['review','direction'],required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);review(a) if a.phase=='review' else direction(a)
