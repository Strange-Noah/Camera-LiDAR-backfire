import copy

from . import voxel_generator


def build_voxel_generator(cfg, **kwargs):
    """Builder of voxel generator."""
    if isinstance(cfg, voxel_generator.VoxelGenerator):
        return cfg
    elif isinstance(cfg, dict):
        cfg = copy.deepcopy(cfg)
        obj_type = cfg.pop("type")
        cls = getattr(voxel_generator, obj_type) if isinstance(obj_type, str) else obj_type
        return cls(**cfg, **kwargs)
    else:
        raise TypeError("Invalid type {} for building a sampler".format(type(cfg)))
