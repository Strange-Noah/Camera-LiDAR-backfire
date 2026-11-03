"""Root-side publication of isolated CPU outputs; original archives untouched."""
import argparse,hashlib,io,json,os,pathlib,shutil,tarfile,time
P=pathlib.Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 tmp=p.with_name(p.name+'.part')
 with tmp.open('w') as f:json.dump(d,f,sort_keys=True,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=P,required=True);p.add_argument('--job',type=P,required=True);p.add_argument('--exit-code',type=int,default=None);a=p.parse_args()
 dest=a.job/'analysis';dest.mkdir(parents=True,exist_ok=True);records={}
 journal=a.job/'publication-records.json'
 if journal.exists():records=json.loads(journal.read_text())
 def durable_copy(source,destination):
  def inventory(root):return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
  expected=inventory(source)
  if destination.exists():raise ValueError('Fresh publication namespace required')
  shutil.copytree(source,destination)
  for f in destination.rglob('*'):
   if f.is_file():
    with f.open('rb') as handle:os.fsync(handle.fileno())
  if inventory(destination)!=expected:raise ValueError('Official output readback hash')
  fd=os.open(str(destination),os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
  return expected
 for folder in sorted(a.source.iterdir()):
  if folder.is_dir() and (folder/'parsed_metrics.json').exists() and folder.name not in records:
   records[folder.name]=durable_copy(folder,dest/folder.name);save(journal,records)
 for name in ('experiment_index.json','experiment_index.csv','evaluation-settings.json','dataset-integrity.json','evaluation-progress.json','all-complete.json'):
  f=a.source/name
  if f.is_file():
   b=f.read_bytes();target=dest/name;tmp=target.with_name(target.name+'.part');tmp.write_bytes(b);os.replace(tmp,target)
 if a.exit_code is None:return
 # Copy all small provenance and evaluator logs; never recopy large formatted JSON.
 for f in a.source.glob('*.log'):shutil.copyfile(f,dest/f.name)
 verified=a.source/'verified'
 if verified.exists():shutil.copytree(verified,dest/'verified',dirs_exist_ok=True)
 result=dict(worker_exit=a.exit_code,epoch=time.time(),published_conditions=sorted(records),all_complete=(a.source/'all-complete.json').exists(),publication_record_sha256=sha(journal) if journal.exists() else None)
 save(a.job/'execution-result.json',result)
 files=[]
 for f in dest.rglob('*'):
  if f.is_file() and f.name!='results_nusc.json' and f.stat().st_size<12*1024**2:files.append(f)
 files += [a.job/'execution-result.json',journal,a.job/'task.log']
 previous=a.job/'previous-attempt-evidence'
 if previous.is_dir():files+=list(previous.glob('*.log'))
 contents={}
 for f in files:
  if f.exists() and f.stat().st_size<12*1024**2:contents[str(f.relative_to(a.job))]=f.read_bytes()
 contents['small-evidence-proof.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in contents.items()},sort_keys=True).encode()
 out=io.BytesIO()
 with tarfile.open(fileobj=out,mode='w:gz') as arc:
  for n,b in contents.items():
   info=tarfile.TarInfo(n);info.size=len(b);info.mode=0o600;arc.addfile(info,io.BytesIO(b))
 data=out.getvalue()
 target=a.job/'analysis-evidence.tar.gz'
 with target.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 save(a.job/'analysis-evidence-proof.json',dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),execution=result))
 print('OFFICIAL_EVALUATION_DURABLE '+json.dumps(result),flush=True)
if __name__=='__main__':main()
