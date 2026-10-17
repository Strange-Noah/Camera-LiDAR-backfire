#!/usr/bin/env python3
"""Root-driven checks execute as the experiment user; destructive probes use fixtures only."""
import argparse, datetime, json, os, pwd, shutil, subprocess, sys, tempfile
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--user', required=True); a=p.parse_args()
assert os.geteuid()==0
identity=pwd.getpwnam(a.user); assert identity.pw_uid!=0
base=Path('/workspace/results/storage_permissions')
# Root maintenance briefly enables fixture creation, then restores read-only parents.
os.chmod('/workspace/datasets',0o755)
try:
    fixture=Path(tempfile.mkdtemp(prefix='.permission-fixture-',dir='/workspace/datasets'))
finally:
    os.chmod('/workspace/datasets',0o555)
(fixture/'original.txt').write_text('fixture-only\n')
os.chmod(fixture/'original.txt',0o444); os.chmod(fixture,0o555)
root=Path('/workspace/datasets/nuscenes')
reads=[root/'v1.0-mini/scene.json',root/'v1.0-trainval/scene.json']
for name in ['samples/CAM_FRONT','samples/LIDAR_TOP']:
    reads.append(next(x for x in (root/name).iterdir() if x.is_file()))
code=r'''import errno,json,os,sys
from pathlib import Path
fixture=Path(sys.argv[1]); login=sys.argv[2]; reads=[Path(x) for x in sys.argv[3:]]
assert os.geteuid()!=0
checks={}
for path in reads:
    with path.open('rb') as f: assert f.read(1)
checks['real_metadata_camera_lidar_readable']=True
for path in [Path('/workspace'),Path('/workspace/datasets'),Path('/workspace/datasets/nuscenes')]+reads:
    assert not os.access(path,os.W_OK), str(path)
checks['real_dataset_and_parents_not_writable']=True
ops={
 'create':lambda:(fixture/'new.txt').write_text('x'),
 'overwrite':lambda:(fixture/'original.txt').write_text('x'),
 'delete':lambda:(fixture/'original.txt').unlink(),
 'rename_file':lambda:(fixture/'original.txt').rename(fixture/'renamed.txt'),
 'rename_directory_via_parent':lambda:fixture.rename(fixture.with_name(fixture.name+'-renamed')),

}
for name,op in ops.items():
    try: op()
    except OSError as exc:
        assert exc.errno in (errno.EACCES,errno.EPERM,errno.EROFS),repr(exc)
        checks[name+'_denied']=True
    else: raise AssertionError('Unexpected permission: '+name)
# FUSE may permit chmod despite ownership. Content writes must still be denied.
try: os.chmod(fixture/'original.txt',0o666)
except PermissionError: pass
try: (fixture/'original.txt').write_text('bypass')
except PermissionError: checks['write_denied_after_chmod_attempt']=True
else: raise AssertionError('Permission-change bypass')
for top in ['results','cache','staging','checkpoints']:
    import tempfile
    fd,name=tempfile.mkstemp(prefix='.permission-check-',dir=f'/workspace/{top}/{login}')
    with os.fdopen(fd,'w') as f:f.write('test')
    assert Path(name).read_text()=='test'
    Path(name).unlink()
checks['personal_outputs_writable']=True
print(json.dumps({'status':'PASS','uid':os.geteuid(),'gid':os.getegid(),'checks':checks}))
'''
try:
    result=subprocess.run(['runuser','-u',a.user,'--',sys.executable,'/usr/local/lib/camera-lidar/run_readonly_experiment.py','--',sys.executable,'-c',code,str(fixture),a.user,*map(str,reads)],capture_output=True,text=True,check=True)
    report=json.loads(result.stdout)
    report.update(user=a.user,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  destructive_probes='temporary fixture only; never real dataset')
    (base/'access_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
except subprocess.CalledProcessError as e:
    print(e.stdout,e.stderr,file=sys.stderr);raise
finally:
    os.chmod('/workspace/datasets',0o755)
    try:
        for candidate in [fixture,fixture.with_name(fixture.name+'-renamed')]:
            if candidate.exists():
                os.chmod(candidate,0o755)
                shutil.rmtree(candidate)
    finally:
        os.chmod('/workspace/datasets',0o555)
