"""Official CPU scoring of frozen artifacts. No inference/injection implementation."""
import argparse,base64,collections,gc,hashlib,importlib.metadata,json,os,pathlib,pickle,shutil,subprocess,sys,time,traceback
P=pathlib.Path
CLASSES=('car','truck','construction_vehicle','bus','trailer','barrier','motorcycle','bicycle','pedestrian','traffic_cone')
def sha(p):
 h=hashlib.sha256()
 with P(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 p=P(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.part')
 with tmp.open('w') as f:json.dump(d,f,sort_keys=True,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def safe(v):
 import math
 if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
 if isinstance(v,list):return [safe(x) for x in v]
 if isinstance(v,float) and not math.isfinite(v):return None
 return v

def archive_extract(source,dest,cid,names=None,complete=True):
 from analysis.batch_execution.archive import read_index
 source=P(source);dest=P(dest);dest.mkdir(parents=True,exist_ok=False)
 d=read_index(source);r=json.loads((source/'receipt.json').read_text())
 expected=dict(archive_id=d['archive_id'],condition_id=d['condition_id'],index_sha256=sha(source/'index.json'),shards={x['name']:x['sha256'] for x in d['shards']},completeness=d['completeness'])
 if any(r.get(k)!=v for k,v in expected.items()):raise ValueError('Archive receipt binding')
 if complete and (r['status']!='DURABLE_COMPLETE' or d['condition_id']!=cid):raise ValueError('Incomplete/foreign predictions')
 candidates=[x for x in d['files'] if P(x['path']).name=='completion.json'] if names is None else []
 if names is None:
  if len(candidates)!=1:raise ValueError('Expected unique aggregate handoff')
  parent=P(candidates[0]['path']).parent
  names=[str(parent/n) for n in ('manifest.json','sample_tokens.json','raw_predictions.pkl','input_trace.jsonl','geometry_trace.jsonl','execution_report.json','completion.json')]
 selected={x['path']:x for x in d['files'] if x['path'] in names}
 if set(selected)!=set(names):raise ValueError('Missing aggregate archive members')
 for name,row in selected.items():
  rel=P(name)
  if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive member')
  out=dest/rel;out.parent.mkdir(parents=True,exist_ok=True)
  h=hashlib.sha256()
  with out.open('xb') as f:
   for c in row['chunks']:
    with (source/c['shard']).open('rb') as src:src.seek(c['offset']);b=src.read(c['bytes'])
    if len(b)!=c['bytes'] or hashlib.sha256(b).hexdigest()!=c['sha256']:raise ValueError('Archive chunk hash')
    f.write(b);h.update(b)
  if h.hexdigest()!=row['sha256']:raise ValueError('Reconstructed hash')
 return (dest/parent if names is not None and candidates else dest),dict(archive_id=d['archive_id'],receipt_sha256=sha(source/'receipt.json'),index_sha256=sha(source/'index.json'),shard_hashes_verified=True)

def handoff(path,expected,tokens,identity):
 path=P(path);m=json.loads((path/'manifest.json').read_text());body=m['condition'];cid=m['condition_id']
 if expected and (cid!=expected['condition_id'] or sha(path/'manifest.json')!=expected['manifest_sha256']):raise ValueError('Exact condition manifest mismatch')
 if expected and (body['sigma_R_degrees_per_axis']!=expected['sigma'] or body['run_seed']!=expected['seed']):raise ValueError('Sigma/seed mismatch')
 if body['subset']['sample_tokens']!=tokens or json.loads((path/'sample_tokens.json').read_text())!=tokens:raise ValueError('Frozen token mismatch')
 for k,v in identity.items():
  if body[k]!=v:raise ValueError('Scientific identity mismatch: '+k)
 complete=json.loads((path/'completion.json').read_text())
 for name,h in complete['files'].items():
  if sha(path/name)!=h:raise ValueError('Handoff completion hash: '+name)
 with (path/'raw_predictions.pkl').open('rb') as f:raw=pickle.load(f)
 if raw['condition_id']!=cid or raw['sample_tokens']!=tokens or len(raw['predictions'])!=6019 or raw['model']!='fusion':raise ValueError('Prediction envelope population/identity')
 return m,raw['predictions']

def cache_metrics(row,out,cfg):
 out=P(out);out.mkdir(parents=True,exist_ok=True)
 source=P(row['cached_metrics_root'])
 if row.get('cached_archive'):
  archive=P(row['cached_archive']);tmp=out/'accepted-evaluator-cache'
  archive_extract(archive,tmp,'accepted-cache',names=['official/metrics_summary.json','official/metrics_details.json','formatted/raw-validation.json'],complete=False)
  source=tmp/'official'
 for name in ('metrics_summary.json','metrics_details.json'):
  src=source/name
  if sha(src)!=row['accepted_metric_hashes'][name]:raise ValueError('Accepted metric bytes changed: '+row['name'])
  shutil.copyfile(src,out/name)
 m=json.loads((out/'metrics_summary.json').read_text())
 if m['cfg']!=cfg:raise ValueError('Accepted evaluation configuration changed')
 return m

def verify_row(row,settings,work,out):
 import torch
 if torch.cuda.is_available():raise RuntimeError('CPU-only execution required')
 row=dict(row);tokens=settings['tokens'];identity=settings['identity'];root=P(work)/row['name']
 if row['kind']=='rotation':
  hand,receipt=archive_extract(row['source'],root,row['condition_id']);row['publication']=receipt
  if row.get('expected_receipt_sha256') and receipt['receipt_sha256']!=row['expected_receipt_sha256']:raise ValueError('Frozen receipt SHA')
  if row.get('accepted_archive_id') and receipt['archive_id']!=row['accepted_archive_id']:raise ValueError('Accepted retained archive identity')
  # archive_extract returns the aggregate directory even for selective reads.
  found=list(root.rglob('completion.json'))
  if len(found)!=1:raise ValueError('Handoff reconstruction')
  hand=found[0].parent;row['handoff']=str(hand)
  m,raw=handoff(hand,row,tokens,identity)
  if row.get('accepted_prediction_sha256') and sha(hand/'raw_predictions.pkl')!=row['accepted_prediction_sha256']:raise ValueError('Accepted retained prediction identity')
 elif row['kind']=='clean':
  found=list((P(row['source'])/'payload').rglob('completion.json'))
  if len(found)!=1:raise ValueError('Unique original Clean handoff')
  hand=found[0].parent;row['handoff']=str(hand)
  m,raw=handoff(hand,None,tokens,identity)
  if m['condition']['sigma_R_degrees_per_axis']!=0:raise ValueError('Clean reference condition')
  proof=json.loads((P(row['source'])/'collection-proof.json').read_text())
  relative=str((hand/'raw_predictions.pkl').relative_to(row['source']))
  if proof['run']!='clean1' or proof['mode']!='clean' or proof['status']!='COMPLETE' or proof['files'][relative]!=sha(hand/'raw_predictions.pkl'):raise ValueError('Original Clean publication identity')
  row.update(condition_id=m['condition_id'],manifest_sha256=sha(hand/'manifest.json'))
 else:
  if sha(row['raw_predictions'])!=row['accepted_prediction_sha256'] or sha(row['sample_tokens'])!=row['tokens_sha256']:raise ValueError('LiDAR artifact identity')
  if json.loads(P(row['sample_tokens']).read_text())!=tokens:raise ValueError('LiDAR token alignment')
  import mmcv
  raw=mmcv.load(row['raw_predictions'])
  if len(raw)!=6019:raise ValueError('LiDAR population')
  row['handoff']=None
 if len(tokens)!=6019 or len(set(tokens))!=6019:raise ValueError('Expected exact6019 unique tokens')
 if row['kind']!='lidar':row['raw_prediction_sha256']=sha(P(row['handoff'])/'raw_predictions.pkl')
 else:row['raw_prediction_sha256']=sha(row['raw_predictions'])
 if row.get('reuse'):
  # Reuse only the exact original accepted official summaries; no metric rerun.
  m=cache_metrics(row,P(out)/row['name']/'official',settings['evaluation_config'])
  row.update(metrics_reused=True,metrics=dict(mAP=m['mean_ap'],NDS=m['nd_score']))
 row.update(samples=6019,unique_tokens=6019,completeness='VERIFIED',evaluation_config_sha256=settings['evaluation_config_sha256'],scientific_identity=settings['identity'])
 save(P(out)/'verified'/ (row['name']+'.json'),row)
 del raw;gc.collect();return row

def score(row,settings,out):
 import torch,mmcv
 if torch.cuda.is_available():raise RuntimeError('GPU not allowed')
 from environment.bevfusion.validation.evaluate_cpu import format_raw,evaluate
 out=P(out)/row['name'];out.mkdir(parents=True,exist_ok=True)
 m,raw=handoff(row['handoff'],row,settings['tokens'],settings['identity'])
 raw_path=out/'raw-for-official.pkl';mmcv.dump(raw,str(raw_path));del raw;gc.collect()
 formatted=format_raw(raw_path,P(row['handoff'])/'sample_tokens.json',P(settings['infos']),'fusion',out/'formatted')
 # The raw pickle is a reproducible temporary reserialization; original envelope
 # remains byte-identical in the verified prediction archive.
 raw_path.unlink()
 rawcheck=json.loads((out/'formatted/raw-validation.json').read_text())
 if rawcheck['samples']!=6019 or rawcheck['infos_sha256']!=settings['infos_sha256']:raise ValueError('Formatter population/configuration')
 start=time.time();metrics=evaluate(formatted,P(settings['dataroot']),out/'official')
 if metrics['cfg']!=settings['evaluation_config']:raise ValueError('Official evaluator config mismatch')
 parsed=dict(name=row['name'],sigma=row['sigma'],seed=row['seed'],condition_id=row['condition_id'],mAP=metrics['mean_ap'],NDS=metrics['nd_score'],per_class_AP=metrics['mean_dist_aps'],per_class_TP=metrics['label_tp_errors'],TP_errors=metrics['tp_errors'],official_evaluation_seconds=time.time()-start,input_archive_identity=row.get('publication'),formatted_sha256=sha(formatted),evaluator=settings['software'],status='COMPLETE',metrics_reused=False)
 save(out/'parsed_metrics.json',safe(parsed));return parsed

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=P,required=True);p.add_argument('--work',type=P,required=True);p.add_argument('--output',type=P,required=True);p.add_argument('--verify-row');p.add_argument('--score-row');a=p.parse_args()
 cfg=json.loads(a.input.read_text());a.output.mkdir(parents=True,exist_ok=True);a.work.mkdir(parents=True,exist_ok=True)
 if a.verify_row:
  row=next(r for r in cfg['conditions'] if r['name']==a.verify_row);verify_row(row,cfg,a.work,a.output);return
 if a.score_row:
  row=json.loads((a.output/'verified'/(a.score_row+'.json')).read_text());score(row,cfg,a.output);return
 import torch
 if torch.cuda.is_available():raise RuntimeError('CPU only')
 from environment.bevfusion.experiments.evaluation_integrity import verify_evaluation_inputs
 from nuscenes.eval.detection.config import config_factory
 if importlib.metadata.version('nuscenes-devkit')!='1.1.9':raise ValueError('Pinned official evaluator version')
 cfg['evaluation_config']=config_factory('detection_cvpr_2019').serialize()
 cfg['evaluation_config_sha256']=hashlib.sha256(json.dumps(cfg['evaluation_config'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
 base=P('/workspace/results/validation-baseline-preparation');ready=json.loads((base/'baseline-preflight-report.json').read_text());cfg['infos_sha256']=ready['official_infos_sha256']
 proof=verify_evaluation_inputs(base/'validation-file-manifest.json',cfg['identity']['dataset_sha256'],cfg['infos'],cfg['infos_sha256'],cfg['dataroot'],base/'validation-metadata-view.json')
 cfg['software']=dict(python=sys.version,nuscenes_devkit=importlib.metadata.version('nuscenes-devkit'),numpy=importlib.metadata.version('numpy'),torch=torch.__version__,formatter_sha256=sha('/opt/management/environment/bevfusion/validation/evaluate_cpu.py'),config=cfg['evaluation_config'])
 # Tokens come from authenticated original production manifest, never timestamps guessed.
 if sha(cfg['reference_manifest'])!=cfg['reference_manifest_sha256']:raise ValueError('Original reference manifest SHA')
 original=json.loads(P(cfg['reference_manifest']).read_text());cfg['tokens']=original['condition']['subset']['sample_tokens']
 if any(original['condition'][k]!=v for k,v in cfg['identity'].items()):raise ValueError('Frozen reference scientific identity')
 import mmcv
 infos=mmcv.load(cfg['infos'])['infos']
 if cfg['tokens']!=[r['token'] for r in sorted(infos,key=lambda r:r['timestamp'])]:raise ValueError('Official infos token order mismatch')
 del infos;gc.collect()
 save(a.output/'evaluation-settings.json',safe(cfg));save(a.output/'dataset-integrity.json',proof)
 settings=a.output/'evaluation-settings.json';verified=[]
 for row in cfg['conditions']:
  print('CONSOLIDATE '+row['name'],flush=True)
  command=[sys.executable,__file__,'--input',str(settings),'--work',str(a.work),'--output',str(a.output),'--verify-row',row['name']]
  with (a.output/(row['name']+'-verification.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  verified.append(json.loads((a.output/'verified'/(row['name']+'.json')).read_text()))
 save(a.output/'experiment_index.json',dict(status='ALL_INPUTS_VERIFIED',conditions=verified,evaluator=cfg['software']))
 import csv
 with (a.output/'experiment_index.csv').open('w') as f:
  keys=['name','kind','sigma','seed','condition_id','source','manifest_sha256','raw_prediction_sha256','samples','unique_tokens','completeness','evaluation_config_sha256','metrics_reused']
  w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(verified)
 for row in verified:
  if time.time()>float(os.environ['OFFICIAL25_EXECUTION_DEADLINE'])-180:raise TimeoutError('Finite scoring admission deadline')
  print('OFFICIAL_SCORE '+row['name']+' reused='+str(bool(row.get('reuse'))),flush=True)
  if not row.get('reuse'):
   command=[sys.executable,__file__,'--input',str(settings),'--work',str(a.work),'--output',str(a.output),'--score-row',row['name']]
   with (a.output/(row['name']+'-evaluation.log')).open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
  else:
   metrics=json.loads((a.output/row['name']/'official/metrics_summary.json').read_text());save(a.output/row['name']/'parsed_metrics.json',safe(dict(name=row['name'],sigma=row.get('sigma'),seed=row.get('seed'),condition_id=row.get('condition_id'),mAP=metrics['mean_ap'],NDS=metrics['nd_score'],per_class_AP=metrics['mean_dist_aps'],per_class_TP=metrics['label_tp_errors'],TP_errors=metrics['tp_errors'],metrics_reused=True,status='COMPLETE',evaluator=cfg['software'])))
  save(a.output/'evaluation-progress.json',dict(last_completed=row['name'],epoch=time.time()))
 save(a.output/'all-complete.json',dict(status='COMPLETE',conditions=27,rotation_conditions=25,newly_scored=24,reused=3,epoch=time.time(),index_sha256=sha(a.output/'experiment_index.json')))
 print('OFFICIAL25_ALL_COMPLETE',flush=True)

if __name__=='__main__':
 try:main()
 except BaseException:
  traceback.print_exc();raise
