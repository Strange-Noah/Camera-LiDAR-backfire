"""Execute pinned official scripts unchanged; capture their original globals/plots."""
import argparse,hashlib,json,os,runpy,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--script',choices=['ocamo_drift_tracking','lto_drift_tracking','ocamo_monitoring','lto_monitoring'],required=True);a=p.parse_args()
source=a.upstream.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip();assert sha=='1ac55de3c0eabb02f155cd44f7481117536c9526'
assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=source,text=True).strip()
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import scipy.io
original_loadmat=scipy.io.loadmat
loads=0
def logged_loadmat(*args,**kwargs):
 global loads
 value=original_loadmat(*args,**kwargs);loads+=1
 if loads%100==0:print('OBSERVATIONS_LOADED',loads,flush=True)
 return value
scipy.io.loadmat=logged_loadmat
plt.show=lambda:plt.savefig(out/'official_plot.png',dpi=150)
os.chdir(source/'python/inference');sys.path.insert(0,str(Path.cwd()));start=time.monotonic()
try:
 data=runpy.run_path(a.script+'.py',run_name='__main__')
 arrays={k:v for k,v in data.items() if k in ['allTh','allTh_dec','dec_all','p1','p2','v_lto1','v_lto2'] and isinstance(v,np.ndarray)}
 assert arrays and all(np.isfinite(v).all() for v in arrays.values())
 np.savez_compressed(out/'trajectory.npz',**arrays)
 result=dict(script=a.script,upstream_commit=sha,seed=42,elapsed_seconds=time.monotonic()-start,arrays={k:list(v.shape) for k,v in arrays.items()},finite=True,status='EXECUTED',official_source_modified=False)
 if 'dec_all' in arrays:
  error=(Rotation.from_rotvec(np.deg2rad(arrays['allTh'].T))*Rotation.from_rotvec(np.deg2rad(arrays['dec_all'][:3].T))).magnitude()*180/np.pi
  burn=10 if a.script=='ocamo_drift_tracking' else 0
  initial=Rotation.from_rotvec(np.deg2rad(arrays['dec_all'][:3].T)).magnitude()*180/np.pi
  result.update(observations=len(error),burn_in_observations=burn,post_burn_mean_residual_degrees=float(error[burn:].mean()),post_burn_RMS_residual_degrees=float(np.sqrt(np.mean(error[burn:]**2))),post_burn_mean_uncorrected_degrees=float(initial[burn:].mean()),final_residual_degrees=float(error[-1]),mean_seconds_per_observation=(time.monotonic()-start)/len(error),trajectory_sha256=hashlib.sha256(b''.join(v.tobytes() for v in arrays.values())).hexdigest())
 if 'monitoring' in a.script:
  clean=arrays['p1'] if 'p1' in arrays else arrays['v_lto1'];corrupted=arrays['p2'] if 'p2' in arrays else arrays['v_lto2']
  result['validity_summary']=dict(observations=len(clean),clean_mean=float(clean.mean()),decalibrated_window_mean=float(corrupted[51:110].mean()),outside_decalibrated_window_mean=float(np.concatenate([corrupted[:51],corrupted[110:]]).mean()))
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except BaseException as e:
 (out/'failure.json').write_text(json.dumps(dict(script=a.script,status='FAILED',error_type=type(e).__name__,error=str(e),elapsed_seconds=time.monotonic()-start),indent=2)+'\n');raise
