"""Explicit distribution method/constructor adapters with per-frontend state."""
import inspect
import pickle
from types import MethodType
from typing import Any, cast

from .frontend import tensor_frontend


TENSOR_PARAMETERS = frozenset({
    "mu", "sigma", "loc", "scale", "low", "high", "concentration",
    "concentration0", "concentration1", "rate", "probs", "logits",
    "p", "temperature", "covariance_matrix",
})


def has_trainable_parameter(value, state, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, state.tensor_type):
        return bool(value.requires_grad)
    if isinstance(value, state.native.Distribution):
        return any(has_trainable_parameter(item, state, seen) for item in vars(value).values())
    if isinstance(value, (tuple, list)):
        return any(has_trainable_parameter(item, state, seen) for item in value)
    return value is not None and not isinstance(value, (bool, int, float, complex, str))


def normal_loc(instance):
    return instance.mu


def set_normal_loc(instance, value):
    instance.mu = value


def normal_scale(instance):
    return instance.sigma


def set_normal_scale(instance, value):
    instance.sigma = value


def resolve_distribution_member(class_name, name, role):
    import jittor as jt
    from .context import get_install_context
    namespace = get_install_context(jt).target_namespace.distributions
    owner = getattr(namespace, class_name)
    member = inspect.getattr_static(owner, name)
    return getattr(member, role) if role else member


class DistributionMethod:
    """Descriptor whose execution implementation is this module-owned class."""

    def __init__(self, state, implementation, name, role=""):
        self.state = state
        self.implementation = implementation
        self.__name__ = name
        self.role = role
        self.owner = None
        self.__doc__ = implementation.__doc__
        self.__signature__ = inspect.signature(implementation)

    def __get__(self, instance, owner=None):
        return self if instance is None else MethodType(self, instance)

    def __call__(_adapter, *args, **kwargs):
        with tensor_frontend(_adapter.state.tensor_type):
            result = _adapter.implementation(*args, **kwargs)
            if (_adapter.implementation.__name__ == "rsample"
                    and isinstance(result, _adapter.state.tensor_type)
                    and not has_trainable_parameter(args[0], _adapter.state)):
                result = result.stop_grad()
            return result

    def __reduce__(self):
        key = (cast(Any, self.owner).__name__, self.__name__, self.role)
        if resolve_distribution_member(*key) is not self:
            raise pickle.PicklingError("distribution member is not the currently published owner")
        return resolve_distribution_member, key


class DistributionConstructor(DistributionMethod):
    def __call__(_adapter, instance, *args, **kwargs):
        state = _adapter.state
        original = _adapter.implementation
        signature = _adapter.__signature__
        if original is state.native.Normal.__init__:
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
        with tensor_frontend(state.tensor_type):
            bound = signature.bind(instance, *args, **kwargs)
            if original is state.native.Uniform.__init__:
                low, high = bound.arguments.get("low"), bound.arguments.get("high")
                if isinstance(low, (int, float)) and isinstance(high, (int, float)) and high <= low:
                    raise ValueError("Uniform requires high > low")
            for name, value in tuple(bound.arguments.items()):
                if name not in TENSOR_PARAMETERS or value is None:
                    continue
                dtype = state.target.get_default_dtype() if isinstance(value, (int, float)) else None
                bound.arguments[name] = state.target.as_tensor(value, dtype=dtype)
            return original(*bound.args, **bound.kwargs)


class DistributionAdapterState:
    """Configuration and class identities belonging to exactly one frontend."""

    def __init__(self, native, target):
        self.native = native
        self.target = target
        self.tensor_type = target.Var
        self.adapters = {}

    def adapt(self, original):
        if original in self.adapters:
            return self.adapters[original]
        bases = (original,)
        if original is not self.native.Distribution:
            bases += tuple(self.adapt(base) for base in original.__bases__
                           if issubclass(base, self.native.Distribution))
        constructor = DistributionConstructor(self, original.__init__, "__init__")
        owned = [constructor]
        attributes = {"__module__": "torch.distributions", "__init__": constructor}
        if original is self.native.Normal:
            attributes["loc"] = property(normal_loc, set_normal_loc)
            attributes["scale"] = property(normal_scale, set_normal_scale)
        for name in dir(original):
            if name.startswith("_"):
                continue
            descriptor = inspect.getattr_static(original, name)
            if isinstance(descriptor, property):
                accessors = []
                for role in ("fget", "fset", "fdel"):
                    function = getattr(descriptor, role)
                    adapted = DistributionMethod(self, function, name, role) if function else None
                    accessors.append(adapted)
                    if adapted is not None:
                        cast(Any, owned).append(adapted)
                attributes[name] = cast(Any, property)(*accessors, doc=descriptor.__doc__)
            elif inspect.isfunction(descriptor):
                adapted = DistributionMethod(self, descriptor, name)
                attributes[name] = adapted
                cast(Any, owned).append(adapted)
            elif isinstance(descriptor, (dict, list, set)):
                attributes[name] = descriptor.copy()
        result = type(original.__name__, bases, attributes)
        for implementation in owned:
            implementation.owner = result
        self.adapters[original] = result
        return result
