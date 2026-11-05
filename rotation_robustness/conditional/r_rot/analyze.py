"""Reuse accepted object hits; stratify by one prespecified geometric coefficient."""
import argparse,json,gzip,hashlib,csv,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import summarize

def run(labels,features,output):
 labels=Path(labels);output=Path(output);output.mkdir(exist_ok=False)
 objects=json.loads((labels/'objects.json').read_text());base=json.loads((labels/'summary.json').read_text())
 with gzip.open(features,'rt') as f:geo=json.load(f)
 assert len(objects)==len(geo)==53043
 for a,b in zip(objects,geo):assert (a['sample_token'],a['annotation_token'])==(b['sample_token'],b['annotation_token'])
 hits=np.stack([np.load(labels/(name+'.npz'))['hit'] for name in base['conditions']],axis=1)
 assert hits.shape==(53043,27)
 coef=np.array([r.get('coefficient_per_degree',np.nan) for r in geo]);valid=np.isfinite(coef);db=np.array([r['db'] for r in objects]);scenes=sorted(set(r['scene'] for r in objects));si=np.array([scenes.index(r['scene']) for r in objects])
 bins=lambda sigma:np.searchsorted([.1,.5,1.],coef*sigma,side='right')
 def group(mask):
  counts=np.bincount(si[mask],minlength=len(scenes));misses=np.zeros((len(scenes),27),int)
  np.add.at(misses,si[mask],~hits[mask]);r=summarize(counts,misses)
  r['denominator_sha256']=hashlib.sha256('\n'.join(objects[i]['sample_token']+':'+objects[i]['annotation_token'] for i in np.flatnonzero(mask)).encode()).hexdigest()
  return r
 fixed=[[group(valid&(db==d)&(bins(5)==r)) for r in range(4)] for d in range(5)]
 dynamic=[]
 for sigma in range(1,6):
  for d in range(5):
   for r in range(4):
    g=group(valid&(db==d)&(bins(sigma)==r));row=dict(sigma=sigma,distance_bin=d,r_bin=r,N=g['N'],scenes=g.get('scenes',0),denominator_sha256=g['denominator_sha256'])
    if g['N']:
     row.update(scenes=g['scenes'],B_pp=g['B'],lidar_miss_percent=g['rates_percent'][0],clean_miss_percent=g['rates_percent'][1],D_mean_pp=g['D_mean'][sigma-1],D_sd_pp=g['D_sd'][sigma-1],G_mean_pp=g['G_mean'][sigma-1],G_sd_pp=g['G_sd'][sigma-1],backfire_seeds=g['backfire_seeds'][sigma-1])
     for seed in range(5):
      row['noisy_miss_seed%d_percent'%seed]=g['rates_percent'][2+(sigma-1)*5+seed];row['D_seed%d_pp'%seed]=g['D'][sigma-1][seed];row['G_seed%d_pp'%seed]=g['G'][sigma-1][seed]
    dynamic.append(row)
 for sigma in range(1,6):assert sum(r['N'] for r in dynamic if r['sigma']==sigma)+int((~valid).sum())==53043
 result=dict(objects=53043,samples=6019,geometrically_assigned=int(valid.sum()),unassigned=int((~valid).sum()),coefficient_quantiles=np.percentile(coef[valid],[0,5,25,50,75,95,100]).tolist(),fixed_sigma5_groups=fixed,dynamic=dynamic,unassigned_summary=group(~valid),label_hashes={name:hashlib.sha256((labels/(name+'.npz')).read_bytes()).hexdigest() for name in base['conditions']})
 for row in dynamic:
  if row['N']:
   for seed in range(5):assert abs(row['G_seed%d_pp'%seed]-(row['D_seed%d_pp'%seed]-row['B_pp']))<1e-10
 (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 fields=list(dict.fromkeys(k for r in dynamic for k in r))
 with (output/'distance_r_sigma.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(dynamic)
 render(result,output/'figures')
 return result

def render(data,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(out);out.mkdir(exist_ok=False)
 grid=data['fixed_sigma5_groups'];rl=['<0.1','0.1–0.5','0.5–1','≥1'];dl=['0–10','10–20','20–30','30–40','≥40'];colors=plt.get_cmap('tab10').colors
 def save(fig,name):
  fig.tight_layout()
  for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=180)
  plt.close(fig)
 matrices=[]
 for key in ['B','D_mean','G_mean']:
  matrices.append(np.array([[np.nan if not g['N'] else g[key] if key=='B' else g[key][-1] for g in row] for row in grid]))
 lim=max(np.nanmax(np.abs(a)) for a in matrices);fig,axs=plt.subplots(1,3,figsize=(14,5))
 for ax,a,title in zip(axs,matrices,['Initial benefit B','Degradation D at σ=5°','Backfire gap G at σ=5°']):
  im=ax.imshow(a,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(4),rl);ax.set_yticks(range(5),dl);ax.set(xlabel='r_rot at σ=5°',ylabel='Car distance (m)',title=title);fig.colorbar(im,ax=ax,label='percentage points')
  for i in range(5):
   for j in range(4):
    g=grid[i][j];label='empty' if not g['N'] else '%.1f\nN=%d\nS=%d%s'%(a[i,j],g['N'],g['scenes'],'*' if g['N']<200 or g['scenes']<10 else '')
    ax.text(j,i,label,ha='center',va='center',fontsize=6.5)
 fig.suptitle('Car: fixed σ=5° geometry groups; * limited support');save(fig,'A_distance_r_heatmaps')
 fig,axs=plt.subplots(1,4,figsize=(16,4.5),sharey=True)
 for j,ax in enumerate(axs):
  for i in range(5):
   g=grid[i][j]
   if not g['N'] or g['N']<200 or g['scenes']<10:continue
   ax.errorbar(range(1,6),g['D_mean'],yerr=g['D_sd'],color=colors[i],marker='o',ms=3,label=dl[i]+'m')
   ax.axhline(g['B'],color=colors[i],ls='--',lw=1)
  ax.set(xlabel='Rotation σ (degrees)',title='Fixed r_rot(5°) '+rl[j]);ax.legend(fontsize=7)
 axs[0].set_ylabel('Degradation D (solid) / initial benefit B (dashed), pp');fig.suptitle('Same GT objects across σ within each curve; bars: seed SD');save(fig,'B_D_versus_B_fixed_groups')
 fig,ax=plt.subplots(figsize=(8,5))
 for j in range(4):
  yy=[grid[i][j]['D_mean'][-1] if grid[i][j]['N']>=200 and grid[i][j]['scenes']>=10 else np.nan for i in range(5)]
  ax.plot(range(5),yy,marker='o',label=rl[j])
 ax.set_xticks(range(5),dl);ax.set(xlabel='Car distance (m)',ylabel='Mean degradation at σ=5° (pp)',title='Residual distance association within r_rot(5°) groups');ax.legend(title='Fixed r_rot bins');save(fig,'C_residual_distance')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--labels');p.add_argument('--features');p.add_argument('--output',required=True);p.add_argument('--summary');a=p.parse_args()
 if a.summary:render(json.loads(Path(a.summary).read_text()),a.output)
 else:run(a.labels,a.features,a.output)
