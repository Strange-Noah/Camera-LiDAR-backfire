"""Bind CPU evaluation inputs to the frozen dataset/reference identities.

This is an integrity guard, not a metric implementation or data repair. Large
metadata hashes belong on CPU, never inside an inference forward loop.
"""
import json
from pathlib import Path
from .experiment_manifest import digest

TABLES={'category','attribute','visibility','instance','sensor','calibrated_sensor','ego_pose','log','scene','sample','sample_data','sample_annotation','map'}

def verify_evaluation_inputs(dataset_manifest,expected_dataset_sha256,infos,expected_infos_sha256,dataroot,metadata_view=None):
    dataset_manifest=Path(dataset_manifest)
    if digest(dataset_manifest)!=expected_dataset_sha256:raise ValueError('Evaluation dataset-manifest hash mismatch')
    manifest=json.loads(dataset_manifest.read_text())
    if manifest.get('dataset_version')!='v1.0-trainval' or manifest.get('split')!='val':
        raise ValueError('Unexpected evaluation population')
    if digest(infos)!=expected_infos_sha256:raise ValueError('Evaluation infos hash mismatch')
    metadata=manifest['metadata']
    if set(metadata)!=TABLES:raise ValueError('Incomplete canonical metadata reference')
    reference_tables=metadata
    if metadata_view is not None:
        view=json.loads(Path(metadata_view).read_text())
        if view.get('original_tables')!=metadata or set(view.get('tables',{}))!=TABLES:
            raise ValueError('Unbound/incomplete validation metadata view')
        reference_tables=view['tables']
    for name,reference in sorted(reference_tables.items()):
        # Validate the actual requested root, not just reference absolute paths.
        target=Path(dataroot)/manifest['dataset_version']/(name+'.json')
        if digest(target)!=reference['sha256']:raise ValueError('Evaluation metadata hash mismatch: '+name)
    return dict(status='PASS',scope='infos_and_all_13_requested_root_metadata_tables',metadata_reference='trusted_cpu_prepared_validation_view' if metadata_view is not None else 'canonical_tables',dataset_sha256=expected_dataset_sha256,infos_sha256=expected_infos_sha256)
