"""Local aggregation/figures only. Official metrics are read, never recomputed."""
import argparse,csv,hashlib,json,math,pathlib,shutil,statistics,sys,os
os.environ.setdefault("MPLCONFIGDIR","/tmp/rotation-mpl-cache")
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
P=pathlib.Path
METRICS=('mAP','NDS')
def writecsv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def crossings(points):
 result=[]
 for (a,x),(b,y) in zip(points[:-1],points[1:]):
  if x*y<0:result.append(dict(interval_degrees=[a,b],interpolated_sigma_degrees=a-x*(b-a)/(y-x),direction='enter_backfire' if x>y else 'exit_backfire'))
 return result

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--recovered',type=P,required=True);parser.add_argument('--output',type=P,required=True);parser.add_argument('--lidar-reference',type=P);a=parser.parse_args()
 src=a.recovered/'analysis';out=a.output;out.mkdir(parents=True,exist_ok=True)
 complete=json.loads((src/'all-complete.json').read_text())
 if complete['status']!='COMPLETE' or complete['rotation_conditions']!=25:raise ValueError('Incomplete official scoring')
 index=json.loads((src/'experiment_index.json').read_text());values=[];perclass=[]
 reference_path=a.lidar_reference or (P(__file__).resolve().parents[2]/'reports/2026-10-21_validation-baseline/lidar-execution-report.json')
 lidar_reference=json.loads(reference_path.read_text());normalizations=[]
 for entry in index['conditions']:
  if entry['kind']=='lidar':
   original_identity=entry['scientific_identity']
   entry['scientific_identity']=dict(checkpoint_sha256=lidar_reference['checkpoint_sha256'],config_sha256=lidar_reference['resolved_config_sha256'],dataset_sha256=lidar_reference['manifest_sha256'],runtime_sha256=lidar_reference['runtime_sha256'],upstream_commit=lidar_reference['upstream_commit'],input_protocol=lidar_reference['input_protocol'])
   normalizations.append(dict(name=entry['name'],old_generic_identity=original_identity,correct_baseline_identity=entry['scientific_identity'],reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),reason='Reporting metadata only: use the accepted LiDAR checkpoint/config rather than the generic Fusion identity; predictions/metrics unchanged.'))
 (out/'index_normalization.json').write_text(json.dumps(normalizations,indent=2)+'\n')
 index['evaluation_status']='ALL27_OFFICIAL_COMPLETE'
 for entry in index['conditions']:entry['evaluation_status']='COMPLETE'
 (out/'experiment_index.json').write_text(json.dumps(index,indent=2)+'\n')
 normalized=[]
 for entry in index['conditions']:
  ident=entry['scientific_identity'];normalized.append(dict(name=entry['name'],kind=entry['kind'],sigma_degrees=entry.get('sigma'),seed=entry.get('seed'),condition_id=entry.get('condition_id'),source=entry['source'],manifest_sha256=entry.get('manifest_sha256'),raw_prediction_sha256=entry['raw_prediction_sha256'],checkpoint_sha256=ident.get('checkpoint_sha256'),model_config_sha256=ident.get('config_sha256'),dataset_sha256=ident.get('dataset_sha256'),runtime_sha256=ident.get('runtime_sha256'),samples=entry['samples'],completeness=entry['completeness'],evaluation_config_sha256=entry['evaluation_config_sha256'],metrics_reused=bool(entry.get('reuse')),evaluation_status='COMPLETE'))
 writecsv(out/'experiment_index.csv',normalized)
 for row in index['conditions']:
  name=row['name'];d=json.loads((src/name/'parsed_metrics.json').read_text());official=json.loads((src/name/'official/metrics_summary.json').read_text())
  if d['status']!='COMPLETE':raise ValueError('Incomplete metric')
  if not all(math.isfinite(official[k]) and 0<=official[k]<=1 for k in ('mean_ap','nd_score')):raise ValueError('Official metric range/finite check')
  record=dict(name=name,kind=row['kind'],sigma_degrees=row.get('sigma'),seed=row.get('seed'),condition_id=row.get('condition_id'),samples=row['samples'],evaluation_status=d['status'],mAP_percent=100*official['mean_ap'],NDS_percent=100*official['nd_score'],metrics_reused=d['metrics_reused'],official_summary_sha256=hashlib.sha256((src/name/'official/metrics_summary.json').read_bytes()).hexdigest())
  values.append(record)
  dest=out/'official'/name;dest.parent.mkdir(exist_ok=True);shutil.copytree(src/name/'official',dest,dirs_exist_ok=True)
  for cls,ap in official['mean_dist_aps'].items():perclass.append(dict(condition=name,sigma_degrees=row.get('sigma'),seed=row.get('seed'),class_name=cls,AP_percent=100*ap,**{k:(None if v is None or not math.isfinite(v) else v) for k,v in official['label_tp_errors'][cls].items()}))
 clean=next(x for x in values if x['kind']=='clean');lidar=next(x for x in values if x['kind']=='lidar');rot=[x for x in values if x['kind']=='rotation']
 if len(rot)!=25 or {(x['sigma_degrees'],x['seed']) for x in rot}!={(s,k) for s in range(1,6) for k in range(5)}:raise ValueError('Noncanonical grid')
 for row in values:
  for metric in METRICS:
   row['delta_clean_'+metric+'_pp']=row[metric+'_percent']-clean[metric+'_percent'];row['delta_lidar_'+metric+'_pp']=row[metric+'_percent']-lidar[metric+'_percent']
 writecsv(out/'metrics_all_conditions.csv',values);writecsv(out/'per_class_metrics.csv',perclass)
 aggregated=[];analysis={};critical=float(t.ppf(.975,4))
 for sigma in range(1,6):
  seeds=sorted([x for x in rot if x['sigma_degrees']==sigma],key=lambda r:r['seed']);r=dict(sigma_degrees=sigma,n=5)
  for metric in METRICS:
   xs=np.array([x[metric+'_percent'] for x in seeds]);mean=float(xs.mean());sd=float(xs.std(ddof=1));margin=mean-lidar[metric+'_percent'];half=critical*sd/math.sqrt(5)
   r.update({metric+'_mean_percent':mean,metric+'_sample_sd_pp':sd,metric+'_min_percent':float(xs.min()),metric+'_max_percent':float(xs.max()),metric+'_delta_clean_pp':mean-clean[metric+'_percent'],metric+'_margin_lidar_pp':margin,metric+'_margin_ci95_low_pp':margin-half,metric+'_margin_ci95_high_pp':margin+half,metric+'_margin_min_pp':float(xs.min())-lidar[metric+'_percent'],metric+'_margin_max_pp':float(xs.max())-lidar[metric+'_percent'],metric+'_backfire_seeds':int(sum(xs<lidar[metric+'_percent'])),metric+'_backfire_frequency':float(sum(xs<lidar[metric+'_percent']))/5})
   for k,x in enumerate(xs):
    r[metric+'_seed'+str(k)+'_percent']=float(x);r[metric+'_margin_seed'+str(k)+'_pp']=float(x)-lidar[metric+'_percent']
  aggregated.append(r)
 writecsv(out/'metrics_by_sigma.csv',aggregated)
 for metric in METRICS:
  below=[r['sigma_degrees'] for r in aggregated if r[metric+'_margin_lidar_pp']<0]
  points=[(0,clean[metric+'_percent']-lidar[metric+'_percent'])]+[(r['sigma_degrees'],r[metric+'_margin_lidar_pp']) for r in aggregated]
  changes=[dict(from_sigma=points[i][0],to_sigma=points[i+1][0],change_pp=points[i+1][1]-points[i][1]) for i in range(len(points)-1)]
  analysis[metric]=dict(observed_grid_threshold_degrees=min(below) if below else None,observed_statement=('Smallest tested five-seed mean below LiDAR-only: '+str(min(below))+' degrees') if below else 'No mean-level backfire observed within σ ≤ 5°.',crossings=crossings(points),nonmonotonic_increases=[x for x in changes if x['change_pp']>0],changes=changes,clean_percent=clean[metric+'_percent'],lidar_percent=lidar[metric+'_percent'],per_sigma=[{k:v for k,v in r.items() if k.startswith(metric+'_') or k in ('sigma_degrees','n')} for r in aggregated])
 methods=dict(uncertainty='Pointwise two-sided95% Student-t interval on mean(Fusion−fixed LiDAR), df4; sample SD across5 seeds, t*SD/sqrt5.',assumptions='Approximate independent, identically distributed, roughly normal run-level margins across5 calibration seeds on this fixed validation population; each run also contains residual execution variability that is not separately identified. LiDAR and Clean references treated as fixed; model execution variability and data-population uncertainty not estimated.',limitations='n=5; no6019-frame pseudo-replication; intervals not simultaneous/multiplicity-corrected. Seed frequencies descriptive, not calibrated probabilities. Same seed is paired across sigma by the frozen perturbation design. Clean sigma0 is one run, not five.',interpolation='Piecewise linear only for adjacent observed margins bracketing zero. Sigma0 anchor is one accepted Clean observation; all nonzero points are five-seed means. No smoothing, monotonic constraint or extrapolation.',definition='Backfire iff Fusion metric is strictly below measured unchanged LiDAR-only. mAP and NDS treated separately.',sigma_definition='Standard deviation of the frozen camera-local rotation-vector error per axis; persistent six-camera calibration draw per condition, not per-frame random corruption.')
 (out/'threshold_analysis.json').write_text(json.dumps(dict(metrics=analysis,methods=methods,t_critical_df4=critical,units='metric levels in percent; differences/SD/intervals in percentage points'),indent=2)+'\n')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':12,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 colors=dict(mAP='#1f5f99',NDS='#654b9c')
 def draw(ax,metric,title):
  color=colors[metric]
  for seed in range(5):
   obs=sorted([r for r in rot if r['seed']==seed],key=lambda r:r['sigma_degrees'])
   ax.scatter([r['sigma_degrees'] for r in obs],[r[metric+'_percent'] for r in obs],color=color,alpha=.5,s=30,zorder=3,edgecolors='white',linewidth=.4)
  xs=[r['sigma_degrees'] for r in aggregated];ys=[r[metric+'_mean_percent'] for r in aggregated];sd=[r[metric+'_sample_sd_pp'] for r in aggregated]
  ax.plot([0]+xs,[clean[metric+'_percent']]+ys,color=color,lw=1.5,zorder=2)
  ax.errorbar(xs,ys,yerr=sd,fmt='o',color=color,ecolor=color,elinewidth=1.6,capsize=4,markersize=5,zorder=4)
  ax.scatter([0],[clean[metric+'_percent']],marker='D',color='#2b7a49',s=40,zorder=5)
  ax.axhline(clean[metric+'_percent'],color='#2b7a49',ls='--',lw=1.3)
  ax.axhline(lidar[metric+'_percent'],color='#b43b3b',ls='-.',lw=1.5)
  ax.set(xlim=(-.15,5.15),xticks=list(range(6)),xlabel='Rotation error σ (° per axis)',ylabel=metric+' (%)',title=title)
  ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
  ax.text(.02,.04,'Five seeds per nonzero σ; bars: ±1 sample SD',transform=ax.transAxes,fontsize=9,color='#444')
 def legend(fig):
  handles=[Line2D([],[],marker='o',linestyle='none',color='#777',alpha=.6,label='Individual seeds (n=5)'),Line2D([],[],marker='o',color='#333',label='Mean ±1 SD; linear connectors'),Line2D([],[],color='#2b7a49',ls='--',label='Clean Fusion (single reference)'),Line2D([],[],color='#b43b3b',ls='-.',label='LiDAR-only (fixed reference)')]
  fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.01),ncol=2,frameon=False,fontsize=9)
 def export(fig,name):
  for extension in ('pdf','svg','png'):fig.savefig(out/(name+'.'+extension),dpi=600 if extension=='png' else None,bbox_inches='tight',facecolor='white')
  plt.close(fig)
 for metric,name in [('mAP','map_vs_sigma'),('NDS','nds_vs_sigma')]:
  fig,ax=plt.subplots(figsize=(7,5.1));draw(ax,metric,metric+' under rotational calibration error');legend(fig);fig.subplots_adjust(bottom=.23);export(fig,name)
 fig,axes=plt.subplots(1,2,figsize=(12.5,5));draw(axes[0],'mAP','(A) Detection accuracy');draw(axes[1],'NDS','(B) nuScenes detection score');legend(fig);fig.subplots_adjust(bottom=.23,wspace=.28);export(fig,'rotation_two_panel')
 text=['# Overall rotational backfire thresholds','',f"Measured references: Clean Fusion mAP {clean['mAP_percent']:.6f}%, NDS {clean['NDS_percent']:.6f}%; LiDAR-only mAP {lidar['mAP_percent']:.6f}%, NDS {lidar['NDS_percent']:.6f}%.",'','σ=0 is accepted Runner B Clean #1 only. Clean #2 is not duplicated as another seed.','']
 for metric in METRICS:
  info=analysis[metric];text+=['## '+metric,'',info['observed_statement']+'.','','|σ (°)|Mean (%)|Sample SD (pp)|Margin vs LiDAR (pp)|Pointwise95% t interval (pp)|Seeds below LiDAR|','|---|---:|---:|---:|---|---|']
  for r in aggregated:text.append('|%d|%.6f|%.6f|%+.6f|[%+.6f, %+.6f]|%d/5|'%(r['sigma_degrees'],r[metric+'_mean_percent'],r[metric+'_sample_sd_pp'],r[metric+'_margin_lidar_pp'],r[metric+'_margin_ci95_low_pp'],r[metric+'_margin_ci95_high_pp'],r[metric+'_backfire_seeds']))
  text+=['']
  if info['crossings']:
   for c in info['crossings']:text.append('Linear interpolated crossing: **%.2f°**, bracket **%g–%g°**, direction %s. This is not a measured threshold.'%(c['interpolated_sigma_degrees'],*c['interval_degrees'],c['direction']))
  else:text.append('No adjacent observed margins bracket zero; no continuous crossing estimate or extrapolation is reported.')
  if info['nonmonotonic_increases']:text.append('Non-monotonic mean increases are retained: '+json.dumps(info['nonmonotonic_increases'])+'. Multiple crossing intervals, if present, remain distinct.')
  text+=['']
 text+=['## Interpretation and limitations','',methods['uncertainty'],methods['assumptions'],methods['limitations'],methods['interpolation'],methods['sigma_definition'],'','One seed crossing alone is not a statistically established overall threshold. The observed-grid rule is descriptive; confidence intervals qualify evidence rather than redefining the threshold. mAP/NDS disagreement is preserved. No conditional backfire analysis is performed.']
 (out/'threshold_analysis.md').write_text('\n'.join(text)+'\n')
 # Self-contained small official/provenance outputs; large original predictions stay on NV.
 prov=out/'provenance';prov.mkdir(exist_ok=True)
 for n in ['experiment_index.csv','experiment_index.json','evaluation-settings.json','dataset-integrity.json','all-complete.json']:
  if (src/n).exists():shutil.copyfile(src/n,prov/n)
 for f in src.glob('*.log'):shutil.copyfile(f,prov/f.name)
 (out/'analysis_methods.json').write_text(json.dumps(methods,indent=2)+'\n')
 print(json.dumps(analysis,indent=2))
if __name__=='__main__':main()
