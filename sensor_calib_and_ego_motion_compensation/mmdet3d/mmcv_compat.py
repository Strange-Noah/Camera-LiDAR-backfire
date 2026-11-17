"""Small compatibility helpers for APIs removed from MMCV 2.x."""
from functools import wraps
import inspect

import torch
from mmengine.model import BaseModule


def _convert(value, dtype):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point():
            return value.to(dtype=dtype)
        return value
    if isinstance(value, list):
        return [_convert(item, dtype) for item in value]
    if isinstance(value, tuple):
        return tuple(_convert(item, dtype) for item in value)
    if isinstance(value, dict):
        return {key: _convert(item, dtype) for key, item in value.items()}
    return value


def _cast_decorator(dtype, apply_to=None, output_dtype=None, requires_fp16_enabled=True):
    def decorate(func):
        signature = inspect.signature(func)

        @wraps(func)
        def wrapped(*args, **kwargs):
            module = args[0] if args else None
            if requires_fp16_enabled and not getattr(module, "fp16_enabled", False):
                return func(*args, **kwargs)

            bound = signature.bind_partial(*args, **kwargs)
            names = apply_to or tuple(
                name for name in signature.parameters if name not in ("self", "cls")
            )
            for name in names:
                if name in bound.arguments:
                    bound.arguments[name] = _convert(bound.arguments[name], dtype)
            output = func(*bound.args, **bound.kwargs)
            if output_dtype is not None:
                output = _convert(output, output_dtype)
            return output

        return wrapped

    return decorate


def force_fp32(apply_to=None, out_fp16=False):
    """Cast selected half precision tensor inputs to float32."""
    return _cast_decorator(
        torch.float32,
        apply_to=apply_to,
        output_dtype=torch.float16 if out_fp16 else None,
    )


def auto_fp16(apply_to=None, out_fp32=False):
    """Cast selected float32 tensor inputs to float16 when enabled on a module."""
    return _cast_decorator(
        torch.float16,
        apply_to=apply_to,
        output_dtype=torch.float32 if out_fp32 else None,
    )