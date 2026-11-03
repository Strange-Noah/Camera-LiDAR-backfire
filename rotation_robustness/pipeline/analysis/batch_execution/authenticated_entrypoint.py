"""Fresh-process admission bridge. No model import until downstream invocation."""
import argparse
import json
import runpy
import sys
from .frozen_admission import install_before_consumers, REGISTRY_PATH, read_authenticated
from .archive import sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--module',required=True,
        choices=('analysis.throughput.runner','analysis.batch_execution.evaluate'))
    args,rest=p.parse_known_args()
    binding=None
    if args.module=='analysis.throughput.runner':
        if rest.count('--manifest')!=1:
            raise ValueError('One explicit authenticated manifest path required')
        path=rest[rest.index('--manifest')+1]
        manifest,_=read_authenticated(path)
        binding=dict(condition_id=manifest['condition_id'],manifest_sha256=sha(path))
    install_before_consumers()
    print('AUTHENTICATED_MANIFEST_ADMISSION:'+json.dumps(dict(
        policy='authenticated-frozen-rotation-v1',registry_sha256=sha(REGISTRY_PATH),
        admission_source_sha256=sha(__import__('analysis.batch_execution.frozen_admission',fromlist=['x']).__file__),binding=binding,
        matrix_source='authenticated saved Q; no resampling or replacement'),sort_keys=True),flush=True)
    sys.argv=[args.module]+rest
    runpy.run_module(args.module,run_name='__main__')


if __name__=='__main__':main()
