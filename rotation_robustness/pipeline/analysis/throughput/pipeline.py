"""CPU-only ordered prefetch with frozen geometry/transforms and explicit RNG handoff."""
import copy
import hashlib
import os
import pickle
import random
import time
from pathlib import Path
import numpy as np
import torch


def tensor_array(value):
    if hasattr(value, 'data') and not isinstance(value, torch.Tensor):
        value = value.data
    if hasattr(value, 'tensor'):
        value = value.tensor
    if value.device.type != 'cpu':
        raise ValueError('Loader/IPC payload must remain CPU-only')
    return value.detach().contiguous().numpy()


def array_hash(value):
    return hashlib.sha256(tensor_array(value).tobytes()).hexdigest()


def restore_forward_rng(packet):
    # The serial path seeds in __getitem__, then consumes CPU randomness in the
    # pipeline. Restore its POST-pipeline states, not simply seed(0) in the parent.
    random.setstate(packet['rng']['python'])
    np.random.set_state(packet['rng']['numpy'])
    torch.set_rng_state(packet['rng']['torch_cpu'])
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(packet['input']['seed'])


def single_packet(items):
    if len(items) != 1:
        raise ValueError('Only batch size one is supported')
    return items[0]


def ordered_loader(dataset, workers=4, prefetch=2, context='fork'):
    from torch.utils.data import DataLoader
    # Do not let DataLoader base-seed generation consume model/default CPU RNG.
    options = dict(batch_size=1, shuffle=False, drop_last=False,
                   collate_fn=single_packet, num_workers=workers, pin_memory=False,
                   generator=torch.Generator().manual_seed(0))
    if workers:
        options.update(prefetch_factor=prefetch, multiprocessing_context=context,
                       persistent_workers=False, timeout=120)
    return DataLoader(dataset, **options)


class PreparedDataset:
    """Return trace evidence alongside each sample; no shared worker callbacks.

    Apply the unchanged dynamic wrappers INSIDE each worker. Parent metadata is
    built once; fork shares it copy-on-write. Per-worker trace files avoid
    concurrent append corruption; the parent orders the canonical final traces.
    """
    def __init__(self, dataset, body, expected, evidence_root):
        self.base = dataset
        self.cameras = copy.deepcopy(body['cameras'])
        self.clean_repeat = body['experiment'] == 'clean_repeat'
        self.cid = None
        self.expected = expected
        self.evidence_root = Path(evidence_root)
        self.owner = None
        self.geometry = None

    def __len__(self):
        return len(self.base)

    def _prepare(self):
        from environment.bevfusion.experiments.geometry_dataset import wrap_dataset
        from environment.bevfusion.validation.input_protocol import traced_dataset
        self.owner = os.getpid()
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        path = self.evidence_root / ('worker-%d.jsonl' % self.owner)
        def trace(token, cameras):
            self.geometry = dict(token=token, cameras=cameras)
        wrap_dataset(self.base, self.cameras, trace, clean_repeat=self.clean_repeat)
        traced_dataset(self.base, path, self.expected)

    def __getitem__(self, index):
        if self.owner != os.getpid():
            self._prepare()
        start = time.perf_counter()
        data = self.base[index]
        token = self.base.data_infos[index]['token']
        geometry = copy.deepcopy(self.geometry)
        if geometry is None or geometry['token'] != token:
            raise ValueError('Missing/misaligned worker geometry')
        info = self.base.data_infos[index]
        gt = {k:v for k,v in info.items() if k.startswith('gt_') or k in
              ('num_lidar_pts', 'num_radar_pts', 'valid_flag')}
        geometry['gt_info_sha256'] = hashlib.sha256(pickle.dumps(gt, protocol=4)).hexdigest()
        return dict(index=index, token=token, data=data, geometry=geometry,
                    input=copy.deepcopy(self.base.last_input_record),
                    rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                             torch_cpu=torch.get_rng_state()),
                    prepare_seconds=time.perf_counter()-start)
