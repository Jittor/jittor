"""Lazy parameter materialization for the Torch module frontend."""

import math


class UninitializedTensorMixin:
    __slots__ = ()
    _torch_uninitialized = True


class UninitializedBuffer(UninitializedTensorMixin):
    pass


def is_lazy(parameter):
    return isinstance(parameter, UninitializedTensorMixin)


def make_uninitialized_parameter(torch):
    parameter = torch.nn.Parameter

    class UninitializedParameter(UninitializedTensorMixin, parameter):
        __slots__ = ()
        cls_to_become = parameter

        def __new__(cls, requires_grad=True, device=None, dtype=None):
            data = torch.empty(0, device=device, dtype=dtype)
            return parameter.__new__(cls, data, requires_grad)

        @property
        def shape(self):
            raise RuntimeError("Cannot access the shape of an uninitialized parameter")

        def numel(self):
            raise ValueError("Cannot use numel on an uninitialized parameter")

        def detach(self):
            raise ValueError("Cannot detach an uninitialized parameter")

        def materialize(self, shape, device=None, dtype=None):
            # Storage changes in place so optimizers constructed before the
            # first forward still own this parameter.
            self.data = torch.empty(
                shape, device=self.device if device is None else device,
                dtype=self.dtype if dtype is None else dtype,
            )
            self.__class__ = self.cls_to_become

        def __deepcopy__(self, memo):
            result = type(self)(self.requires_grad, self.device, self.dtype)
            memo[id(self)] = result
            return result

        def __repr__(self):
            return "<UninitializedParameter>"

    UninitializedParameter.__module__ = "torch.nn.parameter"
    UninitializedParameter.__qualname__ = "UninitializedParameter"
    return UninitializedParameter


def make_lazy_linear(torch, uninitialized_parameter):
    linear = torch.nn.Linear
    module = torch.nn.Module

    class LazyLinear(linear):
        __slots__ = ()
        cls_to_become = linear

        def __init__(self, out_features, bias=True, device=None, dtype=None):
            # The eager Jittor constructor initializes both dimensions. A lazy
            # layer owns real parameter holders before either input is known.
            module.__init__(self)
            self.in_features = 0
            self.out_features = out_features
            self.weight = uninitialized_parameter(device=device, dtype=dtype)
            if bias:
                self.bias = uninitialized_parameter(device=device, dtype=dtype)
            else:
                self.register_parameter("bias", None)
            self._initialize_hook = self.register_forward_pre_hook(
                self._infer_parameters, with_kwargs=True,
            )
            self._load_hook = self.register_load_state_dict_pre_hook(self._materialize_for_load)

        def has_uninitialized_params(self):
            return is_lazy(self.weight) or is_lazy(self.bias)

        def reset_parameters(self):
            if not self.has_uninitialized_params() and self.in_features:
                linear.reset_parameters(self)

        def initialize_parameters(self, input):
            features = input.shape[-1]
            with torch.no_grad():
                if is_lazy(self.weight):
                    self.weight.materialize((self.out_features, features))
                    if features and self.out_features:
                        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
                elif self.weight.shape[-1] != features:
                    raise AssertionError(
                        f"Input has {features} features, but the weight has {self.weight.shape[-1]}"
                    )
                if is_lazy(self.bias):
                    self.bias.materialize((self.weight.shape[0],))
                    bound = 1 / math.sqrt(features) if features else 0
                    torch.nn.init.uniform_(self.bias, -bound, bound)
            self.in_features = features

        def _infer_parameters(self, module, args, kwargs):
            self.initialize_parameters(*args, **kwargs)
            self._initialize_hook.remove()
            self._load_hook.remove()
            del self._initialize_hook, self._load_hook
            self.__class__ = self.cls_to_become
            if "input" in kwargs:
                return (kwargs["input"],), {}

        def _materialize_for_load(self, module, state, prefix, *args):
            for name in ("weight", "bias"):
                current = getattr(self, name)
                saved = state.get(prefix + name)
                if is_lazy(current) and saved is not None and not is_lazy(saved):
                    current.materialize(saved.shape)

    LazyLinear.__module__ = "torch.nn.modules.linear"
    LazyLinear.__qualname__ = "LazyLinear"
    return LazyLinear
