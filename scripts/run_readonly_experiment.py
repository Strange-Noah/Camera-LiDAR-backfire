#!/usr/bin/env python3
"""Fail-closed Linux Landlock write allowlist, inherited by experiment subprocesses.
Usage: python3 scripts/run_readonly_experiment.py -- COMMAND [ARG...]
"""
import ctypes, os, platform, pwd, stat, sys
from pathlib import Path

def main():
    if os.geteuid()==0:
        raise SystemExit('Refusing root experiment execution; use your member login')
    if platform.system()!='Linux' or platform.machine() not in ('x86_64','aarch64'):
        raise SystemExit('Supported Linux Landlock host required; no unprotected fallback')
    command=sys.argv[1:]
    if command[:1]==['--']: command=command[1:]
    if not command: raise SystemExit('Usage: run_readonly_experiment.py -- COMMAND [ARG...]')
    libc=ctypes.CDLL(None,use_errno=True)
    def syscall(number,*args):
        result=libc.syscall(ctypes.c_long(number),*args)
        if result<0: raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
        return result
    abi=syscall(444,0,0,1)
    if abi<3: raise SystemExit('Landlock ABI >= 3 required, including truncate protection')
    # Write, remove, create, cross-directory reference and truncate; reads remain allowed.
    handled=(1<<1)|sum(1<<b for b in range(4,15))
    class Ruleset(ctypes.Structure): _fields_=[('handled_access_fs',ctypes.c_uint64)]
    class PathRule(ctypes.Structure):
        _pack_=1
        _fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int32)]
    attrs=Ruleset(handled); fd=syscall(444,ctypes.byref(attrs),ctypes.sizeof(attrs),0)
    account=pwd.getpwuid(os.getuid()); home=Path(account.pw_dir).resolve()
    if home==Path('/') or home==Path('/workspace') or Path('/workspace') in home.parents:
        raise SystemExit('Experiment home must be Pod-local, outside /workspace')
    paths=[home,Path('/tmp'),Path('/dev/shm')]
    paths += [Path('/workspace')/top/account.pw_name for top in ['results','cache','staging','checkpoints']]
    try:
        for path in paths:
            if not path.is_dir() or path.is_symlink(): raise SystemExit(f'Missing/unsafe writable directory: {path}')
            if str(path).startswith('/workspace/') and path.resolve()!=path:
                raise SystemExit(f'Symlink writable path refused: {path}')
            pathfd=os.open(path,os.O_PATH|os.O_CLOEXEC)
            try:
                rule=PathRule(handled,pathfd); syscall(445,fd,1,ctypes.byref(rule),0)
            finally: os.close(pathfd)
        # Writable device sinks, not arbitrary /dev or /proc paths.
        for name in ['/dev/null','/dev/tty']:
            if not os.path.exists(name): continue
            pathfd=os.open(name,os.O_PATH|os.O_CLOEXEC)
            try:
                rule=PathRule(1<<1,pathfd); syscall(445,fd,1,ctypes.byref(rule),0)
            finally: os.close(pathfd)
        if libc.prctl(38,1,0,0,0)!=0: raise OSError(ctypes.get_errno(),'PR_SET_NO_NEW_PRIVS')
        syscall(446,fd,0)
    finally: os.close(fd)
    os.environ['PROJECT_DATASET_WRITE_GUARD']='landlock'
    os.execvp(command[0],command)

if __name__=='__main__': main()
