"""Local grouping of accepted Car hit caches; no inference, matching or rescoring."""
import argparse,csv,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core import summarize

class Data:
 def __init__(self,labels,geometry,scene_table):
  labels=Path(labels);self.objects=json.loads((labels/'objects.json').read_text());reference=json.loads((labels/'summary.json').read_text())
  with gzip.open(geometry,'rt') as f:geo=json.load(f)
  assert len(self.objects)==len(geo)==53043
  self.keys=[o['sample_token']+':'+o['annotation_token'] for o in self.objects];assert len(set(self.keys))==53043
  assert self.keys==[g['sample_token']+':'+g['annotation_token'] for g in geo]
  self.hits=np.stack([np.load(labels/(n+'.npz'))['hit'] for n in reference['conditions']],1);assert self.hits.shape==(53043,27)
  self.scenes=sorted(set(o['scene'] for o in self.objects));self.si=np.array([self.scenes.index(o['scene']) for o in self.objects]);assert len(self.scenes)==150
  rows=list(csv.DictReader(open(scene_table)));scene={r['scene_token']:r for r in rows};assert len(scene)==len(rows) and all(o['scene'] in scene for o in self.objects)
  angle=lambda g:float(np.degrees(np.arctan2(np.linalg.norm(g['center_camera'][:2]),g['center_camera'][2]))) if g['valid'] else np.nan
  self.values=dict(distance=[o['distance'] for o in self.objects],lidar_points=[o['num_lidar_pts'] for o in self.objects],r_rot5=[g.get('coefficient_per_degree',np.nan)*5 for g in geo],projected_width=[g.get('width_pixels',np.nan) for g in geo],off_axis_angle=[angle(g) for g in geo],camera=[g.get('camera','missing') for g in geo],scene_mean_ego_speed=[float(scene[o['scene']]['ego_mean_speed']) for o in self.objects],scene_max_ego_yaw_rate=[float(scene[o['scene']]['ego_max_yaw_rate']) for o in self.objects])
  self.bins=json.loads(Path(__file__).with_name('bins.json').read_text());self.groups={}
  for key,cfg in self.bins.items():
   if 'categories' in cfg:
    labs=cfg['categories'];idx=np.array([labs.index(v) if v in labs else len(labs) for v in self.values[key]])
   else:
    labs=cfg['labels'];v=np.array(self.values[key]);idx=np.searchsorted(cfg['edges'],v,side='right');idx[~np.isfinite(v)]=len(labs)
   self.groups[key]=(labs+['missing'],idx)
 def group(self,mask):
  n=np.bincount(self.si[mask],minlength=150);fn=np.zeros((150,27),int);np.add.at(fn,self.si[mask],~self.hits[mask]);g=summarize(n,fn)
  g['denominator_sha256']=hashlib.sha256('\n'.join(k for k,m in zip(self.keys,mask) if m).encode()).hexdigest();g['supported']=g['N']>=200 and g.get('scenes',0)>=10
  if g['N']:
   g['observed_upward_brackets']=([[0,1]] if g['B']>0 and g['G_mean'][0]>0 else [])+g['upward_brackets']
  if g['N']:assert np.allclose(np.array(g['G']),np.array(g['D'])-g['B'],atol=1e-10,rtol=0)
  return g

def csvwrite(path,rows):
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with open(path,'w') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def row(g,**extra):
 r=dict(**extra,N=g['N'],scenes=g.get('scenes',0),supported=g['supported'],denominator_sha256=g['denominator_sha256'])
 if g['N']:
  r.update(B_pp=g['B'],D5_pp=g['D_mean'][-1],G5_pp=g['G_mean'][-1],G5_SD_pp=g['G_sd'][-1],positive_seeds5=g['backfire_seeds'][-1],first_positive_sigma=g['first_positive_sigma'],upward_brackets=json.dumps(g['observed_upward_brackets']),small_initial_benefit=abs(g['B'])<1,baseline_no_benefit=g['baseline_no_benefit'],lidar_miss_percent=g['rates_percent'][0],clean_miss_percent=g['rates_percent'][1])
  for i in range(5):
   for s in range(5):r['D_%d_seed%d_pp'%(i+1,s)]=g['D'][i][s];r['G_%d_seed%d_pp'%(i+1,s)]=g['G'][i][s];r['noisy_miss_%d_seed%d_percent'%(i+1,s)]=g['rates_percent'][2+i*5+s]
 return r

def phase_a(data,out):
 out=Path(out);out.mkdir(exist_ok=False);result={};rows=[]
 for name,(labs,idx) in data.groups.items():
  result[name]=[dict(label=lab,**data.group(idx==i)) for i,lab in enumerate(labs)];assert sum(g['N'] for g in result[name])==53043
  rows += [row(g,variable=name,group=g['label']) for g in result[name]]
 (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');csvwrite(out/'screening.csv',rows)
 compact=['variable','group','N','scenes','B_pp','D5_pp','G5_pp','G5_SD_pp','positive_seeds5','first_positive_sigma','upward_brackets','small_initial_benefit','baseline_no_benefit','supported'];csvwrite(out/'screening_compact.csv',[{k:r.get(k,'') for k in compact} for r in rows])
 ranks=[]
 for name,groups in result.items():
  gs=[g for g in groups if g['supported'] and g['label']!='missing']
  ranks.append(dict(variable=name,supported_groups=len(gs),B_span_pp=max(g['B'] for g in gs)-min(g['B'] for g in gs),D5_span_pp=max(g['D_mean'][-1] for g in gs)-min(g['D_mean'][-1] for g in gs),G5_span_pp=max(g['G_mean'][-1] for g in gs)-min(g['G_mean'][-1] for g in gs)))
 csvwrite(out/'screening_ranges.csv',ranks);plots(result,out/'figures');(out/'COMPLETE.json').write_text(json.dumps(dict(phase='A',objects=53043,samples=6019,conditions=27,variables=list(result))))

def phase_b(data,out,pair,phase_a_path):
 assert (Path(phase_a_path)/'COMPLETE.json').exists() and (Path(phase_a_path)/'ONE_DIMENSIONAL_FINDINGS.md').exists(),'Phase A findings required before 2D selection'
 out=Path(out);out.mkdir(exist_ok=False);left,right=pair.split(',');ll,li=data.groups[left];rl,ri=data.groups[right];grid=[];rows=[]
 for i,l in enumerate(ll):
  groups=[]
  for j,r in enumerate(rl):
   g=data.group((li==i)&(ri==j));groups.append(g);rows.append(row(g,left=l,right=r))
  grid.append(groups)
 assert sum(g['N'] for gs in grid for g in gs)==53043
 result=dict(pair=[left,right],left_labels=ll,right_labels=rl,grid=grid);(out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');csvwrite(out/'table.csv',rows);heatmaps(result,out/'figures')

def plots(result,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(out);out.mkdir(exist_ok=False)
 for name,groups in result.items():
  for mode in ['G','BD']:
   fig,ax=plt.subplots(figsize=(8,5))
   for g in groups:
    if not g['N']:continue
    label=g['label']+' (N=%d%s)'%(g['N'],'' if g['supported'] else ', sparse')
    if mode=='G':ax.errorbar(range(1,6),g['G_mean'],yerr=g['G_sd'],marker='o',label=label)
    else:
     line=ax.errorbar(range(1,6),g['D_mean'],yerr=g['D_sd'],marker='o',label=label);ax.axhline(g['B'],ls='--',color=line.lines[0].get_color(),lw=1)
   ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Rotation σ (degrees per axis)',ylabel='G (percentage points)' if mode=='G' else 'D (solid) and B (dashed), pp',title=name.replace('_',' ')+'; mean ± seed SD');ax.legend(fontsize=8);fig.tight_layout()
   for ext in ['png','pdf','svg']:fig.savefig(out/(name+'_'+mode+'.'+ext),dpi=180)
   plt.close(fig)

 # Overview panels complement the full-size single-variable figures.
 names=list(result)
 for start in [0,6]:
  chosen=names[start:start+6]
  fig,axs=plt.subplots(2 if len(chosen)>3 else 1,3 if len(chosen)>3 else len(chosen),figsize=(16,9 if len(chosen)>3 else 4.5),squeeze=False)
  for ax,name in zip(axs.flat,chosen):
   for g in result[name]:
    if g['N']:ax.errorbar(range(1,6),g['G_mean'],yerr=g['G_sd'],marker='o',ms=3,label=g['label'])
   ax.axhline(0,color='black',lw=.8);ax.set(title=name.replace('_',' '),xlabel='Rotation σ (degrees)',ylabel='G (pp)');ax.legend(fontsize=7)
  fig.suptitle('Car conditional backfire; mean ± seed SD');fig.tight_layout()
  for ext in ['png','pdf','svg']:fig.savefig(out/('overview_%d.'%start+ext),dpi=180)
  plt.close(fig)

def heatmaps(result,out):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 out=Path(out);out.mkdir(exist_ok=False);grid=result['grid'];arrays=[]
 for key in ['B','D_mean','G_mean']:arrays.append(np.array([[np.nan if not g['N'] else g[key] if key=='B' else g[key][-1] for g in row] for row in grid]))
 lim=max(np.nanmax(np.abs(v)) for v in arrays);fig,axs=plt.subplots(1,3,figsize=(15,6))
 for ax,v,title in zip(axs,arrays,['Initial benefit B','Degradation D at σ=5°','Backfire G at σ=5°']):
  im=ax.imshow(v,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto');ax.set_xticks(range(len(result['right_labels'])),result['right_labels'],rotation=30,ha='right');ax.set_yticks(range(len(result['left_labels'])),result['left_labels']);ax.set(title=title,xlabel=result['pair'][1],ylabel=result['pair'][0]);fig.colorbar(im,ax=ax,label='percentage points')
  for i in range(len(grid)):
   for j,g in enumerate(grid[i]):ax.text(j,i,'empty' if not g['N'] else '%.1f\nN=%d%s'%(v[i,j],g['N'],'' if g['supported'] else '*'),ha='center',va='center',fontsize=7)
 fig.suptitle('Fixed GT groups; * N<200 or fewer than 10 scenes');fig.tight_layout()
 for ext in ['png','pdf','svg']:fig.savefig(out/('benefit_degradation.'+ext),dpi=180)
 plt.close(fig)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--labels');p.add_argument('--geometry');p.add_argument('--scenes');p.add_argument('--output',required=True);p.add_argument('--pair');p.add_argument('--phase-a');p.add_argument('--regenerate');a=p.parse_args()
 if a.regenerate:
  r=json.loads(Path(a.regenerate).read_text());heatmaps(r,a.output) if 'pair' in r else plots(r,a.output)
 else:
  d=Data(a.labels,a.geometry,a.scenes);phase_b(d,a.output,a.pair,a.phase_a) if a.pair else phase_a(d,a.output)
