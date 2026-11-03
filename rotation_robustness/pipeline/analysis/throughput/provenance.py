"""Operational regime identity separate from frozen scientific condition IDs."""
import json
from pathlib import Path
from environment.bevfusion.experiments.experiment_manifest import identity, digest, atomic_json


def execution_regime(workers, prefetch, writer_depth, capture):
    source={p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    body=dict(version='throughput-v1', source=source, workers=workers, prefetch=prefetch,
              writer_depth=writer_depth, pin_memory=False, multiprocessing_context='fork',
              capture_inputs=capture, batch_size=1, model='unchanged MMDataParallel/FP16/eval/no_grad')
    return dict(body=body,sha256=identity(body))


def bind_regime(root, regime):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    path=root/'execution-regime.json'
    if path.exists():
        if json.loads(path.read_text())!=regime:raise ValueError('Conflicting execution regime')
    else:atomic_json(path,regime)


def capture_inputs(path, token, batch):
    # Existing CPU-only descriptor, no change to the executed MMDataParallel batch.
    from analysis.diagnosis.contracts import collated_cpu_payload, describe, metadata
    payload=collated_cpu_payload(batch)
    with Path(path).open('a') as f:
        f.write(json.dumps(dict(token=token,tensors=describe(payload),metadata=metadata(payload)),sort_keys=True)+'\n')
