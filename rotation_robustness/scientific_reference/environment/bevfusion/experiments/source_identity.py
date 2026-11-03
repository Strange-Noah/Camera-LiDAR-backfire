"""Immutable source fingerprint excludes generated manifests and bytecode."""
from pathlib import Path
from .experiment_manifest import identity,digest


def source_identity():
    root=Path(__file__).parent
    files={str(p.relative_to(root)):digest(p) for p in sorted((p for p in root.rglob('*') if p.suffix in ('.py','.sh') or p.name=='specification-lock.json'))}
    for name in ('input_protocol.py','evaluate_cpu.py'):
        files['../validation/'+name]=digest(root.parent/'validation'/name)
    return identity(files)
