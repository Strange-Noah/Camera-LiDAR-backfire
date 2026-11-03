from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mmengine.registry import Registry
from mmdet.registry import DATASETS

from .data_container import DataContainer


def legacy_collate(batch):
    """Collate BEVFusion samples according to their former DataContainer flags."""
    if not batch:
        return batch
    first = batch[0]
    if isinstance(first, DataContainer):
        values = [item.data for item in batch]
        if first.cpu_only:
            return values
        if first.stack:
            return torch.stack(values, dim=0)
        return values
    if isinstance(first, Mapping):
        return {key: legacy_collate([item[key] for item in batch]) for key in first}
    if isinstance(first, torch.Tensor):
        return torch.stack(batch, dim=0)
    if isinstance(first, np.ndarray):
        return torch.as_tensor(np.stack(batch, axis=0))
    if isinstance(first, tuple):
        return tuple(legacy_collate(list(items)) for items in zip(*batch))
    if isinstance(first, list):
        return [legacy_collate(list(items)) for items in zip(*batch)]
    return batch


def build_dataloader(
    dataset,
    samples_per_gpu,
    workers_per_gpu,
    num_gpus=1,
    dist=True,
    shuffle=True,
    seed=None,
    **kwargs,
):
    """Build a PyTorch loader while preserving this project’s old batch shape."""
    distributed = dist and torch.distributed.is_available() and torch.distributed.is_initialized()
    sampler = (
        DistributedSampler(dataset, shuffle=shuffle, seed=seed or 0)
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=samples_per_gpu,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        num_workers=workers_per_gpu,
        collate_fn=legacy_collate,
        pin_memory=kwargs.pop("pin_memory", True),
        drop_last=kwargs.pop("drop_last", False),
        persistent_workers=kwargs.pop("persistent_workers", workers_per_gpu > 0),
        **kwargs,
    )


from .builder import *
from .custom_3d import *
from .nuscenes_dataset import *
from .pipelines import *
from .utils import *
