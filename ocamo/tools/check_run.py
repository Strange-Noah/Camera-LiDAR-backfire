"""Focused checks on saved official and nuScenes outputs; no inference/reruns."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);a=p.parse_args();r=a.results
result={}
for name in ['ocamo','lto']:
 first=np.load(r/(name+'-drift-1')/'trajectory.npz');second=np.load(r/(name+'-drift-2')/'trajectory.npz');assert set(first.files)==set(second.files)
 assert all(np.array_equal(first[k],second[k]) for k in first.files);result[name+'_repeated_trajectory']='byte-exact arrays'
first=[json.loads(s) for s in (r/'nuscenes-3scenes-2seeds/corrections.jsonl').read_text().splitlines()];second=[json.loads(s) for s in (r/'nuscenes-repeat/corrections.jsonl').read_text().splitlines()];assert len(first)==len(second)==240
for x,y in zip(first,second):
 assert x['sample']==y['sample'] and x['seed']==y['seed']
 for k in ['theta_radians','residual_error_degrees','right_correction','corrected_camera_to_lidar']:assert x[k]==y[k],k
 corrupted=np.array(x['initial_corrupted_camera_to_lidar']);corrected=np.array(x['corrected_camera_to_lidar']);correction=np.array(x['right_correction'])
 assert np.allclose(corrupted@correction,corrected,atol=1e-12,rtol=0)
 assert np.array_equal(corrupted[:3,3],corrected[:3,3])
 for mat in [corrupted,corrected,correction]:assert np.allclose(mat[:3,:3].T@mat[:3,:3],np.eye(3),atol=1e-12) and abs(np.linalg.det(mat[:3,:3])-1)<1e-12
 derived=int.from_bytes(hashlib.sha256((x['scene']+':CAM_FRONT:'+str(x['seed'])).encode()).digest()[:8],'little');omega=np.random.default_rng(derived).normal(0,np.deg2rad(.5),3);q=Rotation.from_rotvec(omega).as_matrix();oracle=np.eye(4);oracle[:3,:3]=q.T;clean=corrupted@oracle
 residual=Rotation.from_matrix(clean[:3,:3].T@corrected[:3,:3]).magnitude()*180/np.pi;assert abs(residual-x['residual_error_degrees'])<1e-10
 if x['history_length']<=10:assert np.array_equal(x['theta_radians'],np.zeros(3))
for scene,seed in sorted({(x['scene'],x['seed']) for x in first}):
 rows=[x for x in first if x['scene']==scene and x['seed']==seed];assert [x['history_length'] for x in rows]==list(range(1,41));assert all(rows[i]['timestamp']<rows[i+1]['timestamp'] for i in range(39))
summary=json.loads((r/'nuscenes-3scenes-2seeds/summary.json').read_text());assert all(x['projection_composition_max_abs']<1e-9 and x['oracle_matrix_max_abs']<1e-12 for x in summary['geometry_checks'])
result.update(nuscenes_repeated_estimates='exact',genuine_scenes=3,independent_seeds=2,total_updates=240,per_sequence_observations=40,scene_reset_burn_in_and_chronology='PASS',correction_composition_SO3_and_residual_recalculation='PASS',oracle_and_projection_composition='PASS',final_improved_sequences=sum(x['final_improved'] for x in summary['sequences']),calibration_reliability='NOT ESTABLISHED: mixed convergence; no threshold tuning')
(r/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
