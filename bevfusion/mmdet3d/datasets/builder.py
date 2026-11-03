import platform

from mmengine.dataset import ClassBalancedDataset, ConcatDataset, RepeatDataset
from mmengine.registry import Registry, build_from_cfg
from mmdet.registry import DATASETS

if platform.system() != "Windows":
    # Avoid exhausting file descriptors when DataLoader workers are started.
    import resource

    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(
        resource.RLIMIT_NOFILE,
        (min(max(4096, soft_limit), hard_limit), hard_limit),
    )

OBJECTSAMPLERS = Registry("object sampler")


def _build_dataset(cfg, default_args=None):
    """Build datasets and legacy-style dataset wrappers using MMDet 3 APIs."""
    if isinstance(cfg, (list, tuple)):
        return ConcatDataset([_build_dataset(item, default_args) for item in cfg])

    cfg = cfg.copy()
    dataset_type = cfg.get("type")
    if dataset_type == "ConcatDataset":
        datasets = [_build_dataset(item, default_args) for item in cfg.pop("datasets")]
        cfg.pop("type")
        # MMEngine's ConcatDataset does not expose MMDet 2's separate_eval flag.
        cfg.pop("separate_eval", None)
        if cfg:
            raise TypeError(f"Unsupported ConcatDataset options: {sorted(cfg)}")
        return ConcatDataset(datasets)
    if dataset_type == "RepeatDataset":
        return RepeatDataset(
            _build_dataset(cfg["dataset"], default_args), cfg["times"]
        )
    if dataset_type == "ClassBalancedDataset":
        return ClassBalancedDataset(
            _build_dataset(cfg["dataset"], default_args), cfg["oversample_thr"]
        )
    if dataset_type == "CBGSDataset":
        from mmdet3d.datasets.dataset_wrappers import CBGSDataset

        return CBGSDataset(_build_dataset(cfg["dataset"], default_args))

    ann_files = cfg.get("ann_file")
    if isinstance(ann_files, (list, tuple)):
        datasets = []
        for ann_file in ann_files:
            item_cfg = cfg.copy()
            item_cfg["ann_file"] = ann_file
            datasets.append(build_from_cfg(item_cfg, DATASETS, default_args))
        return ConcatDataset(datasets)

    return build_from_cfg(cfg, DATASETS, default_args)


def build_dataset(cfg, default_args=None):
    return _build_dataset(cfg, default_args)
