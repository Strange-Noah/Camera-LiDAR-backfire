"""Transparent input tracing; upstream transforms and sweep sampling stay unchanged.

The explicit reproducibility convention is a token-derived NumPy/Python/Torch
seed before each sample (base seed 0), four workers, one GPU/rank. This differs from
the README's eight-rank worker RNG streams, but preserves random-nine-of-ten
sampling rather than switching to nearest sweeps. It makes retry/batch boundaries
irrelevant and gives both detectors identical LiDAR inputs.
"""
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch

POLICY = dict(base_seed=0, workers_per_gpu=4, ranks=1,
    seed_algorithm='uint32 little-endian first four SHA256 bytes of clean-val-v1:0:<sample_token>',
    sweep_selection='Unmodified upstream loader: all available if <=9, otherwise np.random.choice(n,9,replace=False)',
    upstream_source_modified=False,
    difference_from_readme='Token-local RNG streams replace eight-rank/worker-dependent RNG streams; sampling distribution and transforms unchanged')


def sample_seed(token):
    return int.from_bytes(hashlib.sha256(('clean-val-v1:0:'+token).encode()).digest()[:4], 'little')


def traced_dataset(dataset, record_path, expected=None):
    record_path = Path(record_path)
    record_path.parent.mkdir(parents=True, exist_ok=True)
    assert not record_path.exists(), 'Do not overwrite existing input evidence'
    stages = dataset.pipeline.transforms
    matches = [i for i, s in enumerate(stages) if type(s).__name__ == 'LoadPointsFromMultiSweeps']
    assert len(matches) == 1
    loader = stages[matches[0]]
    assert loader.sweeps_num == 9 and not loader.test_mode and not loader.load_augmented
    active = {}

    class SweepTrace:
        def __call__(self, results):
            original = loader._load_points
            paths = []
            def tracked(path):
                paths.append(str(path))
                return original(path)
            loader._load_points = tracked
            try:
                result = loader(results)
            finally:
                loader._load_points = original
            active.update(sweep_paths=paths, available_sweeps=len(results['sweeps']),
                padded_keyframe_copies=9 if not results['sweeps'] and loader.pad_empty_sweeps else 0)
            return result
    stages[matches[0]] = SweepTrace()
    original_class = dataset.__class__

    class InputTracedDataset(original_class):
        def __getitem__(self, index):
            token = self.data_infos[index]['token']
            seed = sample_seed(token)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            active.clear()
            item = super().__getitem__(index)
            points = item['points'].data
            if hasattr(points, 'tensor'):
                points = points.tensor
            array = points.detach().cpu().contiguous().numpy()
            record = dict(token=token, seed=seed, lidar_keyframe=str(self.data_infos[index]['lidar_path']),
                points_shape=list(array.shape), points_sha256=hashlib.sha256(array.tobytes()).hexdigest(), **active)
            if expected is not None:
                assert record == expected[token], ('Input identity mismatch', token)
            self.last_input_record = record
            with record_path.open('a') as output:
                output.write(json.dumps(record, separators=(',', ':'))+'\n')
            return item
    dataset.__class__ = InputTracedDataset
    return dataset
