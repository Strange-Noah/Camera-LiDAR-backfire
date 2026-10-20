"""Use official converter on a writable symlink view; never write raw dataset."""
import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if os.geteuid() == 0 or os.environ.get('PROJECT_DATASET_WRITE_GUARD') != 'landlock':
        raise SystemExit('Use the protected non-root experiment shell')
    source = Path('/workspace/datasets/nuscenes').resolve()
    output = args.output.resolve()
    if output == source or source in output.parents:
        raise SystemExit('Output must be outside the protected dataset')
    output.mkdir(parents=True, exist_ok=True)
    for name in ('samples', 'sweeps', 'maps', 'v1.0-mini'):
        path = source / name
        if not path.is_dir():
            raise SystemExit('Missing input directory: ' + str(path))
        link = output / name
        if link.is_symlink() and link.resolve() == path:
            continue
        if link.exists():
            raise SystemExit('Unexpected output entry: ' + str(link))
        link.symlink_to(path, target_is_directory=True)
    sys.path.insert(0, '/opt/bevfusion/tools')
    from data_converter.nuscenes_converter import create_nuscenes_infos
    # Invoking the converter directly avoids unnecessary training GT database writes.
    create_nuscenes_infos(str(output), 'nuscenes', version='v1.0-mini', max_sweeps=10)
    # Make infos portable across CPU/GPU mount user namespaces. Raw paths in
    # the generated records must point at the canonical dataset, not this view.
    import mmcv
    def canonicalize(value):
        if isinstance(value, str) and value.startswith(str(output) + '/'):
            return str(source) + value[len(str(output)):]
        if isinstance(value, dict):
            return {key: canonicalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [canonicalize(item) for item in value]
        if isinstance(value, tuple):
            return tuple(canonicalize(item) for item in value)
        return value
    for path in output.glob('nuscenes_infos_*.pkl'):
        mmcv.dump(canonicalize(mmcv.load(str(path))), str(path))


if __name__ == '__main__':
    main()
