"""CPU-only extraction from identity-verified global nuScenes predictions.

Input config has settings (original evaluation settings), and paths mapping all
condition names to formatted JSON plus expected_sha256. No official scoring.
"""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from core import match_sample, summarize, validate_population

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def save(path,value):
    tmp=path.with_suffix('.part');tmp.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(str(tmp),str(path))

def main(config,output,resume=False):
    cfg=json.loads(Path(config).read_text());settings=cfg['settings'];tokens=settings['tokens']
    if len(tokens)!=6019 or len(set(tokens))!=6019:raise ValueError('expected 6019 unique tokens')
    rows=settings['conditions'];rot=sorted([r for r in rows if r['kind']=='rotation'],key=lambda r:(r['sigma'],r['seed']))
    if [(r['sigma'],r['seed']) for r in rot]!=[(s,k) for s in range(1,6) for k in range(5)]:raise ValueError('grid')
    ordered=[next(r for r in rows if r['kind']==k) for k in ['lidar','clean']]+rot
    out=Path(output);out.mkdir(parents=True,exist_ok=resume)
    identities={}
    for row in ordered:
        item=cfg['paths'][row['name']];actual=sha(item['path'])
        if actual!=item['expected_sha256']:raise ValueError('formatted identity '+row['name'])
        identities[row['name']]=actual
    if resume and (out/'input-identities.json').exists() and json.loads((out/'input-identities.json').read_text())!=identities:raise ValueError('resume input identity mismatch')
    save(out/'input-identities.json',identities)
    from nuscenes import NuScenes
    from nuscenes.eval.common.loaders import load_gt,add_center_dist,filter_eval_boxes
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.utils import category_to_detection_name
    nusc=NuScenes(version='v1.0-trainval',dataroot=settings['dataroot'],verbose=False)
    gt=load_gt(nusc,'val',DetectionBox,verbose=False)
    gt=add_center_dist(nusc,gt)
    gt=filter_eval_boxes(nusc,gt,settings['evaluation_config']['class_range'],verbose=False)
    if set(gt.sample_tokens)!=set(tokens):raise ValueError('GT population mismatch')
    objects=[];by_sample={};seen=set()
    for tok in tokens:
        sample=nusc.get('sample',tok)
        sd=nusc.get('sample_data',sample['data']['LIDAR_TOP']);ego=nusc.get('ego_pose',sd['ego_pose_token'])['translation']
        anns=[nusc.get('sample_annotation',t) for t in sample['anns']]
        current=[]
        for box in gt[tok]:
            if box.detection_name!='car':continue
            candidates=[a for a in anns if category_to_detection_name(a['category_name'])=='car' and np.allclose(a['translation'],box.translation,rtol=0,atol=1e-8)]
            if len(candidates)!=1:raise ValueError('ambiguous annotation association')
            a=candidates[0];key=(tok,a['token'])
            if key in seen:raise ValueError('duplicate object')
            seen.add(key);xy=np.array(a['translation'][:2])-ego[:2];distance=float(np.hypot(*xy));pts=a['num_lidar_pts']
            obj=dict(sample_token=tok,annotation_token=a['token'],instance_token=a['instance_token'],scene=sample['scene_token'],name='car',translation=list(box.translation),distance=distance,num_lidar_pts=pts,db=int(np.searchsorted([10,20,30,40],distance,side='right')),pb=0 if pts<=2 else 1 if pts<=10 else 2)
            current.append(len(objects));objects.append(obj)
        by_sample[tok]=current
    if resume and (out/'objects.json').exists() and json.loads((out/'objects.json').read_text())!=objects:raise ValueError('resume denominator mismatch')
    save(out/'objects.json',objects)
    hits=np.zeros((len(objects),27),dtype=bool);ledger=json.loads((out/'ledger.json').read_text()) if resume and (out/'ledger.json').exists() else []
    complete={r['name']:r for r in ledger}
    for col,row in enumerate(ordered):
        if row['name'] in complete:
            cache=out/(row['name']+'.npz')
            if sha(cache)!=complete[row['name']]['sha256']:raise ValueError('resume cache hash')
            with np.load(cache) as z:hits[:,col]=z['hit']
            continue
        item=cfg['paths'][row['name']];raw=json.loads(Path(item['path']).read_text());preds=raw['results']
        validate_population(tokens,[list(preds)]*27)
        for tok in tokens:
            ps=preds[tok];keys=set()
            for pred in ps:
                if pred['sample_token']!=tok or len(pred['translation'])!=3 or not np.isfinite(pred['translation']).all() or not np.isfinite(pred['detection_score']):raise ValueError('malformed prediction')
                key=(pred['detection_name'],tuple(pred['translation']),pred['detection_score'],tuple(pred['size']),tuple(pred['rotation']))
                if key in keys:raise ValueError('duplicate prediction')
                keys.add(key)
            ids=by_sample[tok]
            hits[ids,col]=match_sample([objects[i] for i in ids],ps)
        np.savez_compressed(out/(row['name']+'.npz'),hit=hits[:,col])
        ledger.append(dict(name=row['name'],objects=len(objects),misses=int((~hits[:,col]).sum()),sha256=sha(out/(row['name']+'.npz'))))
        save(out/'ledger.json',ledger);del raw,preds
    scenes=sorted(set(o['scene'] for o in objects));scene_idx={s:i for i,s in enumerate(scenes)}
    def aggregate(db,pb=None):
        n=np.zeros(len(scenes),int);fn=np.zeros((len(scenes),27),int)
        for i,obj in enumerate(objects):
            if obj['db']!=db or (pb is not None and obj['pb']!=pb):continue
            si=scene_idx[obj['scene']];n[si]+=1;fn[si]+=~hits[i]
        name='distance%d%s'%(db,'' if pb is None else '_points%d'%pb)
        np.savez_compressed(out/(name+'.npz'),n=n,fn=fn,scenes=scenes)
        return summarize(n,fn)
    result=dict(status='COMPLETE_EXTRACTION_NOT_SCIENTIFIC_ACCEPTANCE',samples=len(tokens),objects=len(objects),conditions=[r['name'] for r in ordered],distance=[aggregate(i) for i in range(5)],distance_points=[[aggregate(i,j) for j in range(3)] for i in range(5)])
    save(out/'summary.json',result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');p.add_argument('output');p.add_argument('--resume',action='store_true');a=p.parse_args();main(a.config,a.output,a.resume)
