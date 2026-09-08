"""Native parameter holders and containers."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

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

    def _var_attrs(self):
        """A ParameterList keeps its Vars in ``self.params``, not in ``__dict__``.

        Overriding the one accessor is what makes the traversals in Module work
        here; each of them used to carry its own
        ``if isinstance(v, ParameterList): dc = v.params``.
        """
        return [(k, v) for k, v in self.params.items() if isinstance(v, jt.Var)]

    def append(self, var):
        assert isinstance(var, jt.Var), f"argument <{type(var)}> is not jittor var"
        self.params[len(self.params)] = var

    def add_param(self, name, var):
        assert isinstance(var, jt.Var), f"argument <{type(var)}> is not jittor var"
        self.params[name] = var

    def __setitem__(self, name, var):
        self.add_param(name, var)

    def __len__(self):
        return len(self.params)

class Parameter(jt.Var):
    """A real native tensor subtype with shared data and its own leaf identity."""
    __slots__ = ()
    _frontend_result_type = jt.Var

    def __new__(cls, data=None, requires_grad=True):
        token = jt.core._set_tensor_frontend_type(cls)
        try:
            if data is None:
                value = jt.empty((0,), dtype="float32")
            else:
                source = data if isinstance(data, jt.Var) else jt.array(data)
                value = jt.Var.detach(source)
            value.requires_grad = bool(requires_grad)
            return value
        finally:
            jt.core._reset_tensor_frontend_type(token)

    def __init__(self, data=None, requires_grad=True):
        # The converter already initialized the native holder in __new__.
        pass

    def __reduce_ex__(self, protocol):
        return (_rebuild_parameter,
                (type(self), self.numpy(), _jittor_dtype_name(self.dtype), self.requires_grad),
                self.__dict__.copy())

    def __deepcopy__(self, memo):
        from copy import deepcopy
        result = _rebuild_parameter(type(self), self.numpy(), _jittor_dtype_name(self.dtype), self.requires_grad)
        memo[id(self)] = result
        result.__dict__.update(deepcopy(self.__dict__, memo))
        return result


def _rebuild_parameter(parameter_type, array, dtype, requires_grad):
    """Restore the real holder without rerunning application subclass init."""
    token = jt.core._set_tensor_frontend_type(parameter_type)
    try:
        value = jt.array(array, dtype=dtype)
        value.requires_grad = requires_grad
        return value
    finally:
        jt.core._reset_tensor_frontend_type(token)


__all__ = ["Parameter", "ParameterList"]
