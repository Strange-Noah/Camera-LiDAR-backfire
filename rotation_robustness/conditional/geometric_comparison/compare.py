"""Paired fixed-population scene/seed statistics on frozen labels; no scoring."""
import argparse,csv,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'screening'))
from screen import Data,csvwrite,row
from core import summarize
PROTOCOL=json.loads(Path(__file__).with_name('protocol.json').read_text())


def estimates(rates):
    b=rates[...,0]-rates[...,1]
    d=(rates[...,2:]-rates[...,1,None]).reshape(*rates.shape[:-1],5,5)
    g=(rates[...,2:]-rates[...,0,None]).reshape(*rates.shape[:-1],5,5)
    return b,d,g


def ci(v):
    v=np.asarray(v);v=v[np.isfinite(v)]
    return np.percentile(v,[2.5,97.5]).tolist() if len(v) else [None,None]


def stat(v,boot):
    v=float(v);boot=np.asarray(boot);boot=boot[np.isfinite(boot)]
    return dict(estimate_pp=v,ci95_pp=ci(boot),positive_share=float(np.mean(boot>0)) if len(boot) else None,replicates=len(boot))


def onset(b,g):
    # Censor code 6 is used ONLY for ordering, not as an observed rotation angle.
    g=np.asarray(g);res=np.full(g.shape[:-1],6.)
    for s in range(4,-1,-1):res=np.where(np.all(g[...,s:]>0,axis=-1),s+1,res)
    return np.where(np.asarray(b)>0,res,np.nan)


def bootstrap(data,valid,indices,k,weights,seed_draws):
    n=np.zeros((150,k),int);fn=np.zeros((150,k,27),int)
    np.add.at(n,(data.si[valid],indices[valid]),1)
    np.add.at(fn,(data.si[valid],indices[valid]),~data.hits[valid])
    den=weights@n
    with np.errstate(divide='ignore',invalid='ignore'):
        rates=100*np.einsum('rs,sck->rck',weights,fn)/den[...,None]
    bb,dd,gg=estimates(rates)
    bm=dd.mean(-1);gm=gg.mean(-1)
    ds=np.empty_like(bm);gs=np.empty_like(gm)
    for s in range(5):
        ix=np.broadcast_to(seed_draws[:,s,None,:],(len(weights),k,5))
        ds[...,s]=np.take_along_axis(dd[...,s,:],ix,axis=-1).mean(-1)
        gs[...,s]=np.take_along_axis(gg[...,s,:],ix,axis=-1).mean(-1)
    return n,fn,dict(scene=dict(B=bb,D=bm,G=gm),scene_seed=dict(B=bb,D=ds,G=gs))


def tests(name,groups,boots,dl,il):
    trends=[];contrasts=[];cross=[]
    order=list(range(3,-1,-1)) if name=='width' else list(range(4))
    for axis in ['indicator_within_distance','distance_within_indicator']:
        for stratum in range(5 if axis=='indicator_within_distance' else 4):
            ids=[stratum*4+j for j in order] if axis=='indicator_within_distance' else [i*4+stratum for i in range(5)]
            ids=[i for i in ids if groups[i]['supported']]
            label=dl[stratum] if axis=='indicator_within_distance' else il[stratum]
            if len(ids)<2:
                trends.append(dict(variable=name,axis=axis,stratum=label,status='NOT_TESTABLE',supported_cells=len(ids)))
                continue
            for mode,b in boots.items():
                for s in range(6):
                    v=b['D'][:,ids,:].mean(-1) if s==5 else b['D'][:,ids,s]
                    good=np.all(np.isfinite(v),axis=1)
                    trends.append(dict(variable=name,axis=axis,stratum=label,mode=mode,metric='D_curve_average' if s==5 else 'D_sigma%d'%(s+1),supported_cells=len(ids),status='TESTED',replicates=int(good.sum()),strict_order_share=float(np.mean(np.all(np.diff(v[good],axis=1)>0,axis=1)))))
                ons=onset(b['B'][:,ids],b['G'][:,ids,:]);eligible=np.all(np.isfinite(ons),axis=1)
                trends.append(dict(variable=name,axis=axis,stratum=label,mode=mode,metric='first_sustained_crossover',supported_cells=len(ids),status='TESTED',replicates=int(eligible.sum()),eligible_share=float(eligible.mean()),strict_order_share=float(np.mean(np.all(np.diff(ons[eligible],axis=1)<0,axis=1))) if eligible.any() else None,non_strict_order_share=float(np.mean(np.all(np.diff(ons[eligible],axis=1)<=0,axis=1))) if eligible.any() else None))
                pairs=[('endpoint',ids[0],ids[-1])]+[('adjacent',a,z) for a,z in zip(ids[:-1],ids[1:])]
                if len(ids)==2:pairs=pairs[:1]
                for kind,low,high in pairs:
                    for key in ['B','D','G']:
                        for s in ([-1] if key=='B' else range(6)):
                            def point(g):return g['B'] if key=='B' else np.mean(g[key+'_mean']) if s==5 else g[key+'_mean'][s]
                            raw=b[key][:,high]-b[key][:,low]
                            if key!='B':raw=raw.mean(-1) if s==5 else raw[:,s]
                            result=stat(point(groups[high])-point(groups[low]),raw)
                            if key!='B':
                                per_seed=np.array(groups[high][key])-np.array(groups[low][key])
                                result['paired_seed_SD_pp']=float(per_seed.std(axis=1,ddof=1)[s]) if s!=5 else None
                            contrasts.append(dict(variable=name,axis=axis,stratum=label,mode=mode,kind=kind,low=groups[low]['label'],high=groups[high]['label'],metric=key if key=='B' else key+'_curve_average' if s==5 else key+'_sigma%d'%(s+1),**result))
                    lo=onset(b['B'][:,low],b['G'][:,low]);hi=onset(b['B'][:,high],b['G'][:,high]);eligible=np.isfinite(lo)&np.isfinite(hi)
                    cross.append(dict(variable=name,axis=axis,stratum=label,mode=mode,kind=kind,low=groups[low]['label'],high=groups[high]['label'],both_clean_benefits_positive_share=float(eligible.mean()),eligible_replicates=int(eligible.sum()),high_earlier_share=float(np.mean(hi[eligible]<lo[eligible])) if eligible.any() else None,tie_share=float(np.mean(hi[eligible]==lo[eligible])) if eligible.any() else None,high_later_share=float(np.mean(hi[eligible]>lo[eligible])) if eligible.any() else None))
    return trends,contrasts,cross


def heldout(data,valid,features,weights):
    idx=np.flatnonzero(valid);si=data.si[idx];di=data.groups['distance'][1][idx]
    y=(data.hits[idx,1,None,None].astype(float)-data.hits[idx,2:].reshape(-1,5,5).astype(float)).mean(-1)*100
    base=np.eye(5)[di];design={'distance':base}
    for name,ids in features.items():design['distance+'+name]=np.column_stack([base,np.eye(4)[ids[idx]][:,1:]])
    if 'q' in features:design['distance+q+r']=np.column_stack([design['distance+q'],np.eye(4)[features['r'][idx]][:,1:]])
    perm=np.random.default_rng(PROTOCOL['cv_seed']).permutation(150);fold=np.empty(150,int);fold[perm]=np.arange(150)%5
    losses={};fold_rows=[];coeff={}
    for name,x in design.items():
        pred=np.zeros_like(y);coeff[name]=[]
        for f in range(5):
            train=fold[si]!=f;test=~train
            coef,_,rank,_=np.linalg.lstsq(x[train],y[train],rcond=None);pred[test]=x[test]@coef
            coeff[name].append(dict(fold=f,rank=int(rank),columns=x.shape[1],coefficients=coef.tolist()))
            fold_rows.append(dict(model=name,fold=f,scenes=int((fold==f).sum()),objects=int(test.sum()),MSE_pp2=float(np.mean((y[test]-pred[test])**2))))
        err=((y-pred)**2).mean(1);sn=np.bincount(si,minlength=150);se=np.bincount(si,weights=err,minlength=150)
        losses[name]=(err,se,sn)
    comparisons=[]
    for name,(err,se,sn) in losses.items():
        if name=='distance':continue
        reference='distance+q' if name=='distance+q+r' else 'distance'
        delta=losses[reference][1]-se;boot=(weights@delta)/(weights@sn)
        comparisons.append(dict(model=name,reference=reference,objects=len(idx),MSE_pp2=float(err.mean()),reference_MSE_pp2=float(losses[reference][0].mean()),MSE_improvement_pp2=float(delta.sum()/sn.sum()),improvement_CI95_pp2=ci(boot),positive_improvement_share=float(np.mean(boot>0))))
    pairwise=[]
    for candidate,reference in [('distance+q','distance+width'),('distance+r','distance+q'),('distance+r','distance+width')]:
        delta=losses[reference][1]-losses[candidate][1];sn=losses[candidate][2];boot=(weights@delta)/(weights@sn)
        pairwise.append(dict(candidate=candidate,reference=reference,MSE_improvement_pp2=float(delta.sum()/sn.sum()),CI95_pp2=ci(boot),positive_share=float(np.mean(boot>0))))
    return dict(pairwise_indicators=pairwise,baseline_MSE_pp2=float(losses['distance'][0].mean()),comparisons=comparisons,folds=fold_rows,coefficients=coeff,fold_sha256=hashlib.sha256(fold.tobytes()).hexdigest())


def run(a):
    out=Path(a.output);out.mkdir(exist_ok=False)
    data=Data(a.labels,a.geometry,a.scenes)
    valid=np.isfinite(data.values['projected_width']) & np.isfinite(data.values['r_rot5'])
    features=dict(width=data.groups['projected_width'][1],r=data.groups['r_rot5'][1]);labels=dict(width=data.groups['projected_width'][0][:4],r=data.groups['r_rot5'][0][:4])
    physical=None
    if a.physical:
        with gzip.open(a.physical,'rt') as f:physical=json.load(f)
        assert data.keys==[v['sample_token']+':'+v['annotation_token'] for v in physical]
        q=np.array([v['q_rot5'] for v in physical]);assert np.isfinite(q).all()
        features['q']=np.searchsorted(PROTOCOL['q_edges'],q,side='right');labels['q']=PROTOCOL['q_labels']
    else:raise ValueError('Physical GT cache required for the three-variable comparison')
    assert valid.sum()==52664
    weights=np.random.default_rng(PROTOCOL['scene_seed']).multinomial(150,np.full(150,1/150),size=PROTOCOL['replicates'])
    seed_draws=np.random.default_rng(PROTOCOL['seed_sensitivity_seed']).integers(0,5,size=(len(weights),5,5))
    dl=data.groups['distance'][0][:5];di=data.groups['distance'][1]
    summary={};trends=[];contrasts=[];cross=[];csvrows=[];cellcross=[]
    for name,ii in features.items():
        n,fn,boots=bootstrap(data,valid,di*4+ii,20,weights,seed_draws);groups=[]
        for k in range(20):
            mask=valid&(di*4+ii==k);g=summarize(n[:,k],fn[:,k]);g['supported']=g['N']>=200 and g.get('scenes',0)>=10
            g['denominator_sha256']=hashlib.sha256('\n'.join(v for v,m in zip(data.keys,mask) if m).encode()).hexdigest()
            g['label']=dl[k//4]+' / '+labels[name][k%4]
            if g['N']:
                g['observed_upward_brackets']=([[0,1]] if g['B']>0 and g['G_mean'][0]>0 else [])+g['upward_brackets']
                g['uncertainty']={}
                for mode,b in boots.items():
                    g['uncertainty'][mode]=dict(B_CI95=ci(b['B'][:,k]),D_CI95=[ci(b['D'][:,k,s]) for s in range(5)],G_CI95=[ci(b['G'][:,k,s]) for s in range(5)])
                    cx=onset(b['B'][:,k],b['G'][:,k]);eligible=np.isfinite(cx)
                    cellcross.append(dict(variable=name,cell=g['label'],N=g['N'],supported=g['supported'],mode=mode,observed_brackets=json.dumps(g['observed_upward_brackets']),baseline_no_benefit=g['baseline_no_benefit'],positive_clean_benefit_share=float(eligible.mean()),**{'first_sustained_sigma%d_share'%s:float(np.mean(cx[eligible]==s)) if eligible.any() else None for s in range(1,6)},no_cross_through5_share=float(np.mean(cx[eligible]==6)) if eligible.any() else None))
            groups.append(g);csvrows.append(row(g,variable=name,distance=dl[k//4],indicator=labels[name][k%4]))
        summary[name]=dict(distance_labels=dl,indicator_labels=labels[name],groups=groups)
        t,c,x=tests(name,groups,boots,dl,labels[name]);trends+=t;contrasts+=c;cross+=x
    cv=heldout(data,valid,features,weights)
    # Preserve the entire original population separately, including geometrically unassigned objects.
    full=data.group(np.ones(53043,bool));missing=data.group(~valid)
    result=dict(protocol=PROTOCOL,common_objects=int(valid.sum()),accepted_objects=53043,missing_objects=int((~valid).sum()),full_q_bin_counts=np.bincount(features['q'],minlength=4).tolist(),unassigned_q_bin_counts=np.bincount(features['q'][~valid],minlength=4).tolist(),common_identity_sha256=hashlib.sha256('\n'.join(v for v,m in zip(data.keys,valid) if m).encode()).hexdigest(),full_population=full,unassigned=missing,variables=summary,heldout=cv)
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,rows in [('cells',csvrows),('trends',trends),('contrasts',contrasts),('crossover_comparisons',cross),('cell_crossovers',cellcross)]:csvwrite(out/(name+'.csv'),rows)
    csvwrite(out/'heldout_pairwise.csv',cv['pairwise_indicators']);csvwrite(out/'heldout.csv',cv['comparisons']);csvwrite(out/'heldout_folds.csv',cv['folds'])
    print(json.dumps(dict(common_objects=int(valid.sum()),heldout=cv['comparisons']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--labels',required=True);p.add_argument('--geometry',required=True);p.add_argument('--scenes',required=True);p.add_argument('--physical',required=True);p.add_argument('--output',required=True);run(p.parse_args())
