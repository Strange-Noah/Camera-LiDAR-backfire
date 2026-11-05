"""Focused reconciliation of comparison aggregates with frozen source labels."""
import argparse,gzip,hashlib,json,math,sys
from pathlib import Path
import numpy as np
from compare import Data,PROTOCOL,bootstrap,onset
from extract_physical import transverse_width


def validate(a):
    data=Data(a.labels,a.geometry,a.scenes)
    with gzip.open(a.physical,'rt') as f:p=json.load(f)
    assert data.keys==[v['sample_token']+':'+v['annotation_token'] for v in p]
    q=np.array([v['q_rot5'] for v in p]);w=np.array([v['physical_transverse_width'] for v in p])
    assert len(q)==53043 and (w>0).all() and np.isfinite(w).all()
    assert np.allclose(q,np.radians(5)*np.array(data.values['distance'])/w,rtol=1e-12,atol=1e-12)
    for size,quat,ray,expected in [([2,4,1],[1,0,0,0],[1,0],2),([2,4,1],[1,0,0,0],[0,1],4),([2,4,1],[math.sqrt(.5),0,0,math.sqrt(.5)],[1,0],4),([2,4,1],[1,0,0,0],[1,1],6/math.sqrt(2)),([2,4,1],[math.sqrt(.5),math.sqrt(.5),0,0],[1,0],1)]:
        assert abs(transverse_width(size,quat,ray)-expected)<1e-12
    valid=np.isfinite(data.values['projected_width'])&np.isfinite(data.values['r_rot5']);assert valid.sum()==52664
    r=json.loads(Path(a.summary).read_text());di=data.groups['distance'][1]
    oldw=json.loads((Path(__file__).parents[1]/'distance_width/results/summary.json').read_text())['grid']
    oldr=json.loads((Path(__file__).parents[1]/'r_rot/results/summary.json').read_text())['fixed_sigma5_groups']
    features=dict(width=data.groups['projected_width'][1],r=data.groups['r_rot5'][1],q=np.searchsorted(PROTOCOL['q_edges'],q,side='right'))
    weights=np.random.default_rng(PROTOCOL['scene_seed']).multinomial(150,np.full(150,1/150),size=500)
    draws=np.random.default_rng(PROTOCOL['seed_sensitivity_seed']).integers(0,5,size=(500,5,5))
    for name,ii in features.items():
        groups=r['variables'][name]['groups'];assert sum(g['N'] for g in groups)==52664
        _,_,boots=bootstrap(data,valid,di*4+ii,20,weights,draws)
        for k,g in enumerate(groups):
            mask=valid&(di*4+ii==k);assert int(mask.sum())==g['N']
            assert hashlib.sha256('\n'.join(v for v,m in zip(data.keys,mask) if m).encode()).hexdigest()==g['denominator_sha256']
            if name in ['width','r']:
                old=(oldw if name=='width' else oldr)[k//4][k%4]
                assert old['N']==g['N']
                if g['N']:
                    for field in ['B','D','G','D_mean','G_mean','D_sd','G_sd','rates_percent']:assert np.allclose(old[field],g[field],atol=1e-12,rtol=0)
            if not g['N']:continue
            rate=100*(~data.hits[mask]).sum(0)/mask.sum();assert np.allclose(rate,g['rates_percent'],atol=1e-12,rtol=0)
            assert len(np.unique(data.si[mask]))==g['scenes']
            assert abs(rate[0]-rate[1]-g['B'])<1e-12
            assert np.allclose((rate[2:]-rate[1]).reshape(5,5),g['D'],atol=1e-12,rtol=0)
            assert np.allclose(g['G'],np.array(g['D'])-g['B'],atol=1e-10,rtol=0)
            for key in ['D','G']:
                v=np.array(g[key]);assert np.allclose(v.mean(1),g[key+'_mean'],atol=1e-12,rtol=0);assert np.allclose(v.std(1,ddof=1),g[key+'_sd'],atol=1e-12,rtol=0)
            # Independent object-level weighted rate for one paired bootstrap replicate.
            ow=weights[0,data.si[mask]]
            if not ow.sum():continue
            br=100*(ow[:,None]*(~data.hits[mask])).sum(0)/ow.sum()
            assert np.allclose(br[0]-br[1],boots['scene']['B'][0,k],atol=1e-12,rtol=0)
            bd=(br[2:]-br[1]).reshape(5,5)
            assert np.allclose(bd.mean(1),boots['scene']['D'][0,k],atol=1e-12,rtol=0)
            for s in range(5):assert np.allclose(bd[s,draws[0,s]].mean(),boots['scene_seed']['D'][0,k,s],atol=1e-12,rtol=0)
            for mode,b in boots.items():
                for key in ['D','G']:
                    expected=np.nanpercentile(b[key][:,k,:],[2.5,97.5],axis=0).T
                    assert np.allclose(expected,g['uncertainty'][mode][key+'_CI95'],atol=1e-12,rtol=0)
    assert r['unassigned']['N']==379 and r['full_population']['N']==53043
    assert np.allclose(r['unassigned']['D'],0) and np.allclose(r['unassigned']['G'],0)
    assert onset(np.array([1,1,0,-1]),np.array([[-1,1,1,1,1],[-1,-1,-1,-1,-1],[1,1,1,1,1],[1,1,1,1,1]])).tolist()[:2]==[2.,6.]
    assert all(f['rank']==f['columns'] for fs in r['heldout']['coefficients'].values() for f in fs)
    idx=np.flatnonzero(valid);si=data.si[idx];base=np.eye(5)[di[idx]]
    designs={'distance':base}
    for name,ii in features.items():designs['distance+'+name]=np.column_stack([base,np.eye(4)[ii[idx]][:,1:]])
    designs['distance+q+r']=np.column_stack([designs['distance+q'],np.eye(4)[features['r'][idx]][:,1:]])
    perm=np.random.default_rng(PROTOCOL['cv_seed']).permutation(150);fold=np.empty(150,int);fold[perm]=np.arange(150)%5
    assert hashlib.sha256(fold.tobytes()).hexdigest()==r['heldout']['fold_sha256']
    y=100*(data.hits[idx,1,None,None].astype(float)-data.hits[idx,2:].reshape(-1,5,5).astype(float)).mean(-1)
    losses={}
    for name,x in designs.items():
        pred=np.zeros_like(y)
        for item in r['heldout']['coefficients'][name]:
            f=item['fold'];train=fold[si]!=f;test=~train;coef=np.array(item['coefficients'])
            assert np.max(np.abs(x[train].T@(y[train]-x[train]@coef)))<1e-6
            pred[test]=x[test]@coef
            reported=next(v for v in r['heldout']['folds'] if v['model']==name and v['fold']==f)
            assert abs(np.mean((pred[test]-y[test])**2)-reported['MSE_pp2'])<1e-10
        losses[name]=np.mean((pred-y)**2,axis=1)
    assert abs(losses['distance'].mean()-r['heldout']['baseline_MSE_pp2'])<1e-10
    for item in r['heldout']['comparisons']:
        delta=losses[item['reference']]-losses[item['model']]
        assert abs(delta.mean()-item['MSE_improvement_pp2'])<1e-10
        sums=np.bincount(si,weights=delta,minlength=150);counts=np.bincount(si,minlength=150)
        expected=np.percentile((weights@sums)/(weights@counts),[2.5,97.5])
        assert np.allclose(expected,item['improvement_CI95_pp2'],atol=1e-10,rtol=0)
    print(json.dumps(dict(status='PASS',accepted=53043,common=52664,missing=379,physical_width_fixtures='PASS',q_definition='PASS',prior_width_r_reconciliation='PASS',fixed_masks_rates_BDG_seeds='PASS',independent_paired_bootstrap_and_CIs='PASS',CV_train_equations_OOF_scores_and_paired_CI='PASS'),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for v in ['labels','geometry','scenes','physical','summary']:p.add_argument('--'+v,required=True)
    validate(p.parse_args())
