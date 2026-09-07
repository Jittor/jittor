"""Installation-owned NN modules sharing native layer mathematics."""

import types
import inspect
from functools import wraps

from .frontend import tensor_frontend


def prepare_nn_namespace(context):
    target, backend = context.target_namespace, context.native_backend
    if target is backend:
        context.state["Module"] = backend.nn.Module
        return backend.nn
    existing = context.state.get("nn_frontend")
    if (existing is not None and context.state.get("nn_frontend_tensor")
            is context.state["Var"]):
        target.nn = existing
        target.Module = context.state["Module"]
        return existing

    tensor_type = context.state["Var"]
    native_module = backend.nn.Module

    class Module(native_module):
        __slots__ = ()

        def __call__(self, *args, **kwargs):
            with tensor_frontend(tensor_type):
                return native_module.__call__(self, *args, **kwargs)

    Module.__module__ = "torch.nn"
    Module.__qualname__ = "Module"
    adapters = {native_module: Module}
    modules = {}

    def external_objects(args, kwargs):
        seen = set()
        pending = [args, kwargs]
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, native_module):
                pending.extend(vars(value).values())
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
        return seen

    def adapt_class(native):
        known = adapters.get(native)
        if known is not None:
            return known
        initializer = native.__init__

        @wraps(initializer)
        def initialize(self, *args, **kwargs):
            external = external_objects(args, kwargs)
            frozen = False
            if native.__name__ == "Embedding":
                arguments = inspect.signature(initializer).bind_partial(self, *args, **kwargs)
                frozen = bool(arguments.arguments.get("_freeze", False))
            with tensor_frontend(tensor_type):
                initializer(self, *args, **kwargs)
                adopt_owned_children(self, external, frozen)

        adapted = type(native.__name__, (native, Module), {
            "__module__": "torch.nn", "__slots__": (),
            "__init__": initialize, "_torch_native_layer": native,
        })
        adapters[native] = adapted
        return adapted

    def adopt_owned_children(module, external, frozen=False):
        memo = {id(module): module}

        def adapt_value(value):
            if id(value) in external:
                return value
            if id(value) in memo:
                return memo[id(value)]
            if isinstance(value, native_module):
                if isinstance(value, Module):
                    return value
                native_type = type(value)
                if not native_type.__module__.startswith("jittor.nn.modules."):
                    return value
                # A native constructor can return a globally shared child.
                # Wrap its Python structure instead of changing its class or
                # containers in place; tensor/parameter references stay shared.
                result = object.__new__(adapt_class(native_type))
                memo[id(value)] = result
                for name, item in vars(value).items():
                    setattr(result, name, adapt_value(item))
                return result
            if isinstance(value, dict):
                result = value.copy()
                memo[id(value)] = result
                for name, item in value.items():
                    result[name] = adapt_value(item)
                return result
            if isinstance(value, list):
                result = []
                memo[id(value)] = result
                result.extend(adapt_value(item) for item in value)
                return result
            if isinstance(value, tuple):
                items = tuple(adapt_value(item) for item in value)
                if all(item is original for item, original in zip(items, value)):
                    return value
                if type(value) is tuple:
                    result = items
                elif hasattr(value, "_fields"):
                    result = type(value)(*items)
                else:
                    return value
                memo[id(value)] = result
                return result
            return value

        for name, value in tuple(vars(module).items()):
            replacement = adapt_value(value)
            if replacement is not value:
                setattr(module, name, replacement)
        # Mark this constructor's direct Tensor parameters only. Parameters of
        # copied native children keep their existing flags and object identity.
        for _, parameter, role in module._var_roles():
            if (role != "parameter" or id(parameter) in external
                    or not isinstance(parameter, tensor_type)):
                continue
            parameter._is_torch_parameter = True
            parameter._torch_parameter_class = Parameter
            if not frozen and str(parameter.dtype) in (
                    "float16", "bfloat16", "float32", "float64",
                    "complex64", "complex128"):
                parameter.start_grad()
                from .nested import _torch_register_leaf
                _torch_register_leaf(parameter)

    native_parameter = backend.nn.Parameter
    parameter_meta = type(native_parameter)

    class ParameterMeta(parameter_meta):
        def __call__(cls, *args, **kwargs):
            with tensor_frontend(tensor_type):
                return super().__call__(*args, **kwargs)

    Parameter = ParameterMeta("Parameter", (native_parameter,), {
        "__module__": "torch.nn.parameter", "_torch_compat_type": True,
    })

    def copy_module(source, name):
        known = modules.get(id(source))
        if known is not None:
            return known
        result = types.ModuleType(name)
        result.__package__ = name
        if hasattr(source, "__path__"):
            result.__path__ = []
        modules[id(source)] = result
        for key, value in vars(source).items():
            if key.startswith("__") and key not in ("__all__", "__doc__"):
                continue
            if value is native_parameter:
                value = Parameter
            elif isinstance(value, type) and issubclass(value, native_module):
                value = adapt_class(value)
            elif isinstance(value, types.ModuleType) and (
                    value.__name__.startswith("jittor.nn") or value is backend.init):
                value = copy_module(value, name + "." + key)
            elif isinstance(value, (dict, list, set)):
                value = value.copy()
            setattr(result, key, value)
        return result

    namespace = copy_module(backend.nn, "torch.nn")
    namespace.Module = Module
    # Parameter's public module may not exist in the native tree yet.
    parameter_module = types.ModuleType("torch.nn.parameter")
    parameter_module.Parameter = Parameter
    for name in ("UninitializedTensorMixin", "UninitializedParameter", "UninitializedBuffer"):
        setattr(parameter_module, name, type(name, (), {"__module__": "torch.nn.parameter"}))
    namespace.parameter = parameter_module
    context.state["Module"] = Module
    context.state["nn_frontend"] = namespace
    context.state["nn_frontend_tensor"] = tensor_type
    context.state["nn_layer_adapters"] = adapters
    target.Module = Module
    target.nn = namespace
    return namespace
