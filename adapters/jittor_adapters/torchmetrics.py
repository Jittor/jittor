"""TorchMetrics' bounded helper policies, outside the generic Torch facade."""
from ._common import require_version, required_patch, replace, replace_bound_aliases, UnsupportedAdapterVersion

SUPPORTED_VERSIONS = frozenset(("1.7.4",))


def _backend():
    import jittor
    return jittor


def _publish(module, name, factory, marker):
    original = vars(module).get(name)
    if not callable(original):
        raise UnsupportedAdapterVersion("TorchMetrics helper is missing: " + name)
    if getattr(original, marker, None) is not None:
        return False
    fast = factory(original)
    setattr(fast, marker, original)
    replace(module, name, fast, original)
    replace_bound_aliases("torchmetrics", name, original, fast)
    return True


def _bincount(original):
    def bounded(x, minlength=None):
        if minlength is None:
            return original(x, minlength=minlength)
        import numpy as np
        if not isinstance(minlength, (int, np.integer)):
            return original(x, minlength=minlength)
        jt = _backend()
        size = max(int(minlength), 0)
        flat = x.reshape(-1).int64()
        out = jt.zeros((size,), dtype=jt.int64)
        if flat.numel() == 0:
            return out
        return out.scatter_add(0, flat, jt.ones((flat.shape[0],), dtype=jt.int64))
    return bounded


def _dim_zero_cat(original):
    def concatenate(x):
        jt = _backend()
        if isinstance(x, jt.Var):
            return x
        try:
            count = len(x)
        except TypeError:
            return original(x)
        if count == 0:
            raise ValueError("No samples to concatenate")
        if count == 1:
            value = x[0]
            if not isinstance(value, jt.Var):
                return original(x)
            return value.unsqueeze(0) if value.numel() == 1 and getattr(value, "ndim", 0) == 0 else value.clone()
        return original(x)
    return concatenate


def _safe_divide(original):
    def divide(num, denom, zero_division=0.0):
        if not isinstance(zero_division, (float, int)):
            return original(num, denom, zero_division=zero_division)
        if not hasattr(num, "is_floating_point") or not hasattr(denom, "is_floating_point"):
            return original(num, denom, zero_division=zero_division)
        jt = _backend()
        import torch
        num = num if num.is_floating_point() else num.float()
        denom = denom if denom.is_floating_point() else denom.float()
        result = num / denom
        fill = jt.zeros_like(result)
        if zero_division != 0:
            fill = fill + zero_division
        return torch.where(denom != 0, result, fill)
    return divide


@required_patch
def patch_data(module):
    require_version("torchmetrics", SUPPORTED_VERSIONS)
    # Check both contracts before changing either helper.
    for name in ("_bincount", "dim_zero_cat"):
        if not callable(vars(module).get(name)):
            raise UnsupportedAdapterVersion("TorchMetrics helper is missing: " + name)
    changed = _publish(module, "_bincount", _bincount, "_jittor_orig_bincount")
    return _publish(module, "dim_zero_cat", _dim_zero_cat, "_jittor_orig_dim_zero_cat") or changed


@required_patch
def patch_compute(module):
    require_version("torchmetrics", SUPPORTED_VERSIONS)
    return _publish(module, "_safe_divide", _safe_divide, "_jittor_orig_safe_divide")


def register(register_module_patch):
    register_module_patch("torchmetrics.utilities.data", patch_data)
    register_module_patch("torchmetrics.utilities.compute", patch_compute)
