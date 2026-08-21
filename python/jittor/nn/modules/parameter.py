"""Torch-compatible parameter helpers and containers."""

import collections

import jittor as jt


class ParameterList(jt.Module):
    def __init__(self, *args):
        self.params = collections.OrderedDict()
        for var in args:
            if isinstance(var, (collections.OrderedDict, dict)):
                for key, value in var.items():
                    self.add_param(key, value)
            elif isinstance(var, list):
                for value in var:
                    self.append(value)
            else:
                self.append(var)

    def __getitem__(self, idx):
        if idx not in self.params:
            return list(self.params.values())[idx]
        return self.params[idx]

    def __iter__(self):
        return self.params.values().__iter__()

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def items(self):
        return self.params.items()

    def execute(self, x):
        raise NotImplementedError("Parameters is not executable")

    def append(self, var):
        assert isinstance(var, jt.Var), f"argument <{type(var)}> is not jittor var"
        var._is_torch_parameter = True
        self.params[len(self.params)] = var

    def add_param(self, name, var):
        assert isinstance(var, jt.Var), f"argument <{type(var)}> is not jittor var"
        var._is_torch_parameter = True
        self.params[name] = var

    def __setitem__(self, name, var):
        self.add_param(name, var)

    def __len__(self):
        return len(self.params)


def _make_parameter(data, requires_grad=True):
    """Torch-compatible Parameter wrapper.

    Jittor treats a Var assigned to a Module as a parameter, so wrapping an
    existing Var only needs to set the trainable flag. Do not clone here:
    PyTorch's Parameter is a lightweight wrapper over the supplied tensor data,
    while cloning can force materialization/JIT work and makes large pretrained
    model construction unnecessarily slow.
    """
    if not isinstance(data, jt.Var):
        data = jt.array(data)
    data.requires_grad = requires_grad
    data._is_torch_parameter = True
    return data


class _ParameterMeta(type):
    def __instancecheck__(cls, obj):
        return (
            isinstance(obj, jt.Var)
            and bool(getattr(obj, "_is_torch_parameter", False))
        )

    def __call__(cls, data=None, requires_grad=True):
        return _make_parameter(data, requires_grad=requires_grad)


class Parameter(metaclass=_ParameterMeta):
    """Semantic parameter role backed by a marked :class:`jittor.Var`."""


__all__ = ["Parameter", "ParameterList"]
