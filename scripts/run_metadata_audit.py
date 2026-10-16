#!/usr/bin/env python3
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from metadata_audit.construct import construct
from metadata_audit.analyze import analyze,save_json
p=argparse.ArgumentParser(description='Metadata-only audit. No data download or inference.')
p.add_argument('--dataroot',default=os.environ.get('NUSCENES_ROOT','/workspace/datasets/nuscenes'))
p.add_argument('--version',choices=['v1.0-mini','v1.0-trainval'],required=True)
p.add_argument('--output',required=True)
p.add_argument('--cache',default='/tmp/nuscenes-audit-index')
p.add_argument('--config',default=str(Path(__file__).resolve().parents[1]/'configs/audit.json'))
p.add_argument('--analyze-only',action='store_true')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if (out/'RUN_COMPLETE.json').exists():raise SystemExit('Completed output exists; use a new output directory for reproducible runs.')
cfg=json.loads(Path(a.config).read_text())
if not a.analyze_only:construct(a.dataroot,a.version,out/'tables',a.cache,cfg)
validation=json.loads((out/'tables/construction_validation.json').read_text())
if a.version=='v1.0-trainval' and validation['scenes']!=850:raise SystemExit('Final trainval audit requires exactly 850 scenes')
gates=analyze(out/'tables',out,a.version,cfg)
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
dirty=subprocess.check_output(['git','status','--porcelain'],cwd=Path(__file__).resolve().parents[1],text=True).strip()
save_json(out/'RUN_COMPLETE.json',{'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'version':a.version,'git_commit':commit,'git_dirty':bool(dirty),'overall':gates['overall'],'config':cfg})
