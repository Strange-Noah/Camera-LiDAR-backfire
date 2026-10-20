"""Cache installed runtime package payloads on CPU; no host keys or credentials."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

root = Path('/workspace/cache/bevfusion-runtime')
targets = ['openssh-server','ca-certificates','curl','python3','git','libgl1',
           'libglib2.0-0','libopenmpi3','python3-lib2to3','python3-distutils']
raw = subprocess.check_output(['apt-cache','depends','--recurse','--installed',
    '--no-recommends','--no-suggests','--no-conflicts','--no-breaks','--no-replaces',
    '--no-enhances',*targets], text=True)
packages = sorted({line.strip() for line in raw.splitlines()
                   if re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9+.:_-]*',line)})
paths = set()
for package in packages:
    for line in subprocess.check_output(['dpkg-query','-L',package],text=True).splitlines():
        path=Path(line)
        if not line.startswith(('/usr/','/lib/','/lib64/','/bin/','/sbin/','/etc/')):
            continue
        if path.is_file() or path.is_symlink():
            if 'ssh_host_' not in line and not line.startswith('/etc/ssl/private/'):
                paths.add(path)
paths.add(Path('/etc/ssl/certs/ca-certificates.crt'))
# Generated public configuration is not in dpkg's package payload list.
for pattern in ['/etc/pam.d/common-*','/etc/ssh/sshd_config','/etc/ssh/ssh_config','/etc/nsswitch.conf']:
    import glob
    paths.update(Path(x) for x in glob.glob(pattern) if Path(x).is_file())
for path in Path('/etc/alternatives').iterdir():
    if path.is_symlink(): paths.add(path)
archive=root/'os-runtime.tar.gz';temp=archive.with_suffix('.gz.part')
with tarfile.open(temp,'w:gz',compresslevel=3,dereference=False) as target:
    for path in sorted(paths):
        target.add(str(path),arcname=str(path).lstrip('/'),recursive=False)
temp.replace(archive)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(root/'os-runtime.sha256').write_text(digest+'  '+str(archive)+'\n')
(root/'os-runtime-manifest.json').write_text(json.dumps(dict(packages=packages,
    files=len(paths),bytes=archive.stat().st_size,sha256=digest,
    base_image='nvidia/cuda:11.3.1-devel-ubuntu20.04@sha256:e26c89e76b66307f4b7a2bd3e3009c2ec9067be354b12187c44a895657f498f1',
    note='Runtime payload cache from pinned Ubuntu20.04 CPU. Package DB is not restored; no host SSH private keys included.'),indent=2)+'\n')
print('OS_RUNTIME_CACHED',archive.stat().st_size,digest,flush=True)
