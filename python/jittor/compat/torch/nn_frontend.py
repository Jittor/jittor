"""Installation-owned NN modules sharing native layer mathematics."""

import types
import inspect
from functools import wraps

from .frontend import make_parameter_type, tensor_frontend
from .parameter_containers import make_parameter_containers


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
        _frontend_tensor_type = tensor_type

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
        protected = set(external)

        def adapt_value(value):
            if id(value) in external:
                return value
            if id(value) in memo:
                return memo[id(value)]
            if isinstance(value, native_module):
                # A child may have come from shared global state. Its tensor
                # references must not be turned into new Parameters by the
                # enclosing constructor, even when also exposed on the parent.
                protected.update(external_objects((value,), {}))
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
        # Promote only this constructor's own parameters. Role names can denote
        # ParameterList entries rather than attributes; replace by identity in
        # the actual attribute/container graph, never setattr a dotted name.
        roles = tuple(module._var_roles())
        protected.update(id(value) for _, value, role in roles
                         if role in ("buffer", "non_persistent_buffer"))
        replacements = {}
        for _, parameter, role in roles:
            if (role != "parameter" or id(parameter) in protected
                    or not isinstance(parameter, tensor_type)
                    or isinstance(parameter, Parameter)):
                continue
            if id(parameter) not in replacements:
                differentiable = str(parameter.dtype) in (
                    "float16", "bfloat16", "float32", "float64", "complex64", "complex128")
                replacements[id(parameter)] = Parameter(
                    parameter, requires_grad=not frozen and differentiable)

        rewritten = {}

        def replace_parameters(value):
            if id(value) in protected:
                return value
            if id(value) in replacements:
                return replacements[id(value)]
            if id(value) in rewritten:
                return rewritten[id(value)]
            if isinstance(value, dict):
                result = value.copy()
                rewritten[id(value)] = result
                for key, item in value.items():
                    result[key] = replace_parameters(item)
                return result
            if isinstance(value, list):
                result = []
                rewritten[id(value)] = result
                result.extend(replace_parameters(item) for item in value)
                return result
            if isinstance(value, tuple):
                items = tuple(replace_parameters(item) for item in value)
                if all(item is old for item, old in zip(items, value)):
                    return value
                if type(value) is tuple:
                    result = items
                elif hasattr(value, "_fields"):
                    result = type(value)(*items)
                else:
                    return value
                rewritten[id(value)] = result
                return result
            return value

        if replacements:
            for name, value in tuple(vars(module).items()):
                replacement = replace_parameters(value)
                if replacement is not value:
                    setattr(module, name, replacement)

    native_parameter = backend.nn.Parameter
    Parameter = make_parameter_type(backend, tensor_type)

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
    ParameterList, ParameterDict = make_parameter_containers(Module, Parameter, backend.Var)
    namespace.ParameterList = ParameterList
    namespace.ParameterDict = ParameterDict
    namespace.modules.ParameterList = ParameterList
    namespace.modules.ParameterDict = ParameterDict
    namespace.modules.parameter.ParameterList = ParameterList
    namespace.modules.parameter.ParameterDict = ParameterDict
    # Parameter's public module may not exist in the native tree yet.
    parameter_module = types.ModuleType("torch.nn.parameter")
    parameter_module.Parameter = Parameter
    parameter_module.ParameterList = ParameterList
    parameter_module.ParameterDict = ParameterDict
    source_parameter_module = getattr(backend.nn, "parameter", None)
    if source_parameter_module is not None:
        for name in ("UninitializedTensorMixin", "UninitializedParameter", "UninitializedBuffer"):
            if name in vars(source_parameter_module):
                setattr(parameter_module, name, vars(source_parameter_module)[name])
    namespace.parameter = parameter_module
    context.state["Module"] = Module
    context.state["Parameter"] = Parameter
    context.state["nn_frontend"] = namespace
    context.state["nn_frontend_tensor"] = tensor_type
    context.state["nn_layer_adapters"] = adapters
    target.Module = Module
    target.nn = namespace
    return namespace
