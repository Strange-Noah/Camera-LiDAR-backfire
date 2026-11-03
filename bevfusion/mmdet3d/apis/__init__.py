from .test import single_gpu_test


def train_model(*args, **kwargs):
    """Import the legacy training stack only when training is requested."""
    from .train import train_model as _train_model

    return _train_model(*args, **kwargs)
