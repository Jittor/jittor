"""Small neural-network construction helpers."""

from .weight_norm import WeightNorm, remove_weight_norm, weight_norm


def skip_init(module_cls, *args, **kw):
    return module_cls(*args, **kw)


__all__ = ["WeightNorm", "remove_weight_norm", "skip_init", "weight_norm"]
