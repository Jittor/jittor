"""Torch distribution owners delegating mathematics to native distributions."""

import inspect
from functools import wraps
from types import ModuleType

from .frontend import tensor_frontend


def make_distribution_frontend(native, target):
    module = ModuleType("torch.distributions")
    module.__package__ = "torch.distributions"
    module.__path__ = []
    module._native_distribution_module = native
    tensor_type = target.Var
    native_base = native.Distribution
    native_normal = native.Normal
    native_uniform = native.Uniform
    tensor_parameters = {
        "mu", "sigma", "loc", "scale", "low", "high", "concentration",
        "concentration0", "concentration1", "rate", "probs", "logits",
        "p", "temperature", "covariance_matrix",
    }

    def has_trainable_parameter(value, seen=None):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return False
        seen.add(id(value))
        if isinstance(value, tensor_type):
            return bool(value.requires_grad)
        if isinstance(value, native_base):
            return any(has_trainable_parameter(item, seen) for item in vars(value).values())
        if isinstance(value, (tuple, list)):
            return any(has_trainable_parameter(item, seen) for item in value)
        # Opaque transform/config objects can carry differentiable parameters.
        # Preserve their graph rather than guessing that they contain none.
        return value is not None and not isinstance(value, (bool, int, float, complex, str))

    def scoped(function):
        @wraps(function)
        def call(*args, **kwargs):
            with tensor_frontend(tensor_type):
                result = function(*args, **kwargs)
                if (function.__name__ == "rsample" and isinstance(result, tensor_type)
                        and not has_trainable_parameter(args[0])):
                    result = result.stop_grad()
                return result
        return call

    def constructor(original):
        signature = inspect.signature(original)

        @wraps(original)
        def initialize(self, *args, **kwargs):
            if original is native_normal.__init__:
                kwargs = dict(kwargs)
                for public, internal in (("loc", "mu"), ("scale", "sigma")):
                    if public in kwargs:
                        if internal in kwargs:
                            raise TypeError("received both %s and %s" % (public, internal))
                        kwargs[internal] = kwargs.pop(public)
            if "validate_args" in kwargs and "validate_args" not in signature.parameters:
                kwargs = dict(kwargs)
                if kwargs.pop("validate_args") not in (None, False):
                    raise NotImplementedError("this native distribution does not implement validate_args=True")
            with tensor_frontend(tensor_type):
                bound = signature.bind(self, *args, **kwargs)
                if original is native_uniform.__init__:
                    low, high = bound.arguments.get("low"), bound.arguments.get("high")
                    if isinstance(low, (int, float)) and isinstance(high, (int, float)) and high <= low:
                        raise ValueError("Uniform requires high > low")
                for name, value in tuple(bound.arguments.items()):
                    if name not in tensor_parameters or value is None:
                        continue
                    dtype = target.get_default_dtype() if isinstance(value, (int, float)) else None
                    bound.arguments[name] = target.as_tensor(value, dtype=dtype)
                original(*bound.args, **bound.kwargs)
        return initialize

    adapters = {}

    def adapt(original):
        if original in adapters:
            return adapters[original]
        bases = (original,)
        if original is not native_base:
            bases += tuple(adapt(base) for base in original.__bases__
                           if issubclass(base, native_base))
        attributes = {"__module__": "torch.distributions",
                      "__init__": constructor(original.__init__)}
        if original is native_normal:
            attributes["loc"] = property(lambda self: self.mu,
                                         lambda self, value: setattr(self, "mu", value))
            attributes["scale"] = property(lambda self: self.sigma,
                                           lambda self, value: setattr(self, "sigma", value))
        for name in dir(original):
            if name.startswith("_"):
                continue
            descriptor = inspect.getattr_static(original, name)
            if isinstance(descriptor, property):
                attributes[name] = property(
                    scoped(descriptor.fget) if descriptor.fget else None,
                    scoped(descriptor.fset) if descriptor.fset else None,
                    scoped(descriptor.fdel) if descriptor.fdel else None,
                    descriptor.__doc__,
                )
            elif inspect.isfunction(descriptor):
                attributes[name] = scoped(descriptor)
            elif isinstance(descriptor, (dict, list, set)):
                attributes[name] = descriptor.copy()
        result = type(original.__name__, bases, attributes)
        adapters[original] = result
        return result

    for name in native.__all__:
        value = getattr(native, name)
        if isinstance(value, type) and issubclass(value, native_base):
            value = adapt(value)
        elif isinstance(value, ModuleType):
            child = ModuleType("torch.distributions." + name)
            for attribute in dir(value):
                if not attribute.startswith("__"):
                    setattr(child, attribute, getattr(value, attribute))
            value = child
        elif inspect.isfunction(value):
            value = scoped(value)
        setattr(module, name, value)
    module.__all__ = list(native.__all__)
    return module
