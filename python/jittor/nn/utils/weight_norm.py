"""Weight-normalization reparameterization for Jittor modules."""

import jittor as jt
from jittor import nn


def _normalize_dim(value, dim):
    if dim is None or dim == -1:
        return None
    ndim = value.ndim
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(
            "Dimension out of range (expected to be in range of [{}, {}], "
            "but got {})".format(-ndim, ndim - 1, dim)
        )
    return dim


def _norm_except_dim(value, dim):
    dim = _normalize_dim(value, dim)
    if dim is None:
        return jt.sqrt((value * value).sum())
    axes = tuple(axis for axis in range(value.ndim) if axis != dim)
    if not axes:
        return value.abs()
    return jt.sqrt((value * value).sum(axes, keepdims=True))


def _ensure_reparam_hook(module):
    functions = getattr(module, "_reparam_fns", None)
    if functions is not None:
        return functions

    functions = []
    module._reparam_fns = functions
    previous = getattr(module, "__fhook2__", None)
    previous_with_kwargs = getattr(module, "__fhook2_with_kwargs__", False)
    module._reparam_previous_hook = previous
    module._reparam_previous_hook_with_kwargs = previous_with_kwargs

    def dispatch(owner, *args):
        if previous is not None:
            previous(owner, *args)
        for function in tuple(owner._reparam_fns):
            function(owner)

    dispatch._jittor_reparam_dispatch = True
    module.register_forward_pre_hook(
        dispatch,
        with_kwargs=previous_with_kwargs,
    )
    return functions


def _restore_previous_hook(module):
    previous = getattr(module, "_reparam_previous_hook", None)
    previous_with_kwargs = getattr(
        module, "_reparam_previous_hook_with_kwargs", False
    )
    module.remove_pre_forward_hook()
    if previous is not None:
        module.register_forward_pre_hook(
            previous,
            with_kwargs=previous_with_kwargs,
        )
    for name in (
        "_reparam_fns",
        "_reparam_previous_hook",
        "_reparam_previous_hook_with_kwargs",
    ):
        if hasattr(module, name):
            delattr(module, name)


class WeightNorm:
    """Recompute one normalized module parameter before every forward call."""

    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def compute_weight(self, module):
        gain = getattr(module, self.name + "_g")
        direction = getattr(module, self.name + "_v")
        return direction * (gain / _norm_except_dim(direction, self.dim))

    @staticmethod
    def apply(module, name, dim):
        functions = _ensure_reparam_hook(module)
        if any(
            isinstance(function, WeightNorm) and function.name == name
            for function in functions
        ):
            raise RuntimeError(
                "Cannot register two weight_norm hooks on the same parameter {}".format(
                    name
                )
            )

        weight = getattr(module, name)
        normalized_dim = _normalize_dim(weight, dim)
        function = WeightNorm(name, normalized_dim)
        delattr(module, name)
        setattr(module, name + "_g", _norm_except_dim(weight, normalized_dim).clone())
        setattr(module, name + "_v", weight.clone())
        functions.append(function)
        function(module)
        return function

    def remove(self, module):
        weight = self.compute_weight(module).clone()
        weight.persistent = True
        delattr(module, self.name)
        delattr(module, self.name + "_g")
        delattr(module, self.name + "_v")
        setattr(module, self.name, weight)

    def __call__(self, module, inputs=None):
        weight = self.compute_weight(module)
        weight.persistent = False
        setattr(module, self.name, weight)


def weight_norm(module, name, dim):
    """Apply weight normalization to ``module.<name>``."""

    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module, name="weight"):
    """Remove the weight-normalization hook for ``module.<name>``."""

    functions = getattr(module, "_reparam_fns", ())
    for function in tuple(functions):
        if isinstance(function, WeightNorm) and function.name == name:
            function.remove(module)
            functions.remove(function)
            if not functions:
                _restore_previous_hook(module)
            return module
    raise ValueError("weight_norm of '{}' not found in {}".format(name, module))


__all__ = ["WeightNorm", "jt", "nn", "remove_weight_norm", "weight_norm"]
