"""Compatibility for the small subset of MMCV 1.x DataContainer used here."""


class DataContainer:
    """Carry per-sample data and its batching policy through the pipeline."""

    def __init__(
        self,
        data,
        stack=False,
        padding_value=0,
        cpu_only=False,
        pad_dims=2,
    ):
        self.data = data
        self._data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only
        self.pad_dims = pad_dims
