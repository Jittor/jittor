"""Installation-owned NN types; implementation lives outside type factories."""
import types
import inspect

from .frontend import make_parameter_type, tensor_frontend
from .parameter_containers import make_parameter_containers
from .nn_adoption import adopt_owned_children


def module_setattr(module, name, value):
    owner = type(module)._nn_frontend_owner
    attributes = vars(module)
    if attributes.get("_native_parameter_construction", False):
        return owner.native_module.__setattr__(module, name, value)
    if isinstance(value, owner.backend.Var) and not name.startswith("_"):
        parameters = attributes.setdefault("_parameter_names", set())
        non_parameters = attributes.setdefault("_non_parameter_names", set())
        if isinstance(value, owner.Parameter):
            parameters.add(name)
            non_parameters.discard(name)
            attributes.setdefault("_buffer_names", set()).discard(name)
            attributes.setdefault("_non_persistent_buffer_names", set()).discard(name)
        elif name not in attributes.get("_buffer_names", ()):
            if name not in parameters:
                non_parameters.add(name)
    object.__setattr__(module, name, value)


def module_call(module, *args, **kwargs):
    owner = type(module)._nn_frontend_owner
    with tensor_frontend(owner.tensor_type):
        return owner.native_module.__call__(module, *args, **kwargs)


class LayerInitializer:
    """Descriptor binds one native initializer to one frontend owner."""
    def __init__(self, owner, native):
        self.owner = owner
        self.native = native
        self.original = native.__init__
        self.__wrapped__ = self.original
        self.__name__ = "__init__"

    def __get__(self, instance, owner=None):
        return self if instance is None else types.MethodType(self, instance)

    def __call__(self, module, *args, **kwargs):
        owner = self.owner
        external = owner.external_objects(args, kwargs)
        frozen = False
        if self.native.__name__ == "Embedding":
            arguments = inspect.signature(self.original).bind_partial(module, *args, **kwargs)
            frozen = bool(arguments.arguments.get("_freeze", False))
        with tensor_frontend(owner.tensor_type):
            previous = vars(module).get("_native_parameter_construction")
            object.__setattr__(module, "_native_parameter_construction", True)
            try:
                self.original(module, *args, **kwargs)
                adopt_owned_children(owner, module, external, frozen)
            finally:
                if previous is None:
                    vars(module).pop("_native_parameter_construction", None)
                else:
                    object.__setattr__(module, "_native_parameter_construction", previous)


class NNFrontendOwner:
    """Own types and memoized namespace copies for a single installation."""
    def __init__(self, backend, tensor_type):
        self.backend = backend
        self.tensor_type = tensor_type
        self.native_module = backend.nn.Module
        self.Parameter = make_parameter_type(backend, tensor_type)
        self.Module = type("Module", (self.native_module,), {
            "__module__": "torch.nn", "__slots__": (),
            "_frontend_tensor_type": tensor_type, "_nn_frontend_owner": self,
            "__setattr__": module_setattr, "__call__": module_call,
        })
        self.adapters = {self.native_module: self.Module}
        self.modules = {}

    def external_objects(self, args, kwargs):
        seen = set()
        pending = [args, kwargs]
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, self.native_module):
                pending.extend(vars(value).values())
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
        return seen

    def adapt_class(self, native):
        known = self.adapters.get(native)
        if known is not None:
            return known
        adapted = type(native.__name__, (native, self.Module), {
            "__module__": "torch.nn", "__slots__": (),
            "__init__": LayerInitializer(self, native), "_torch_native_layer": native,
        })
        self.adapters[native] = adapted
        return adapted

    def copy_module(self, source, name):
        known = self.modules.get(id(source))
        if known is not None:
            return known
        result = types.ModuleType(name)
        result.__package__ = name
        if hasattr(source, "__path__"):
            result.__path__ = []
        self.modules[id(source)] = result
        for key, value in vars(source).items():
            if key.startswith("__") and key not in ("__all__", "__doc__"):
                continue
            if value is self.backend.nn.Parameter:
                value = self.Parameter
            elif isinstance(value, type) and issubclass(value, self.native_module):
                value = self.adapt_class(value)
            elif isinstance(value, types.ModuleType) and (
                    value.__name__.startswith("jittor.nn") or value is self.backend.init):
                value = self.copy_module(value, name + "." + key)
            elif isinstance(value, (dict, list, set)):
                value = value.copy()
            setattr(result, key, value)
        return result


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
    owner = NNFrontendOwner(backend, tensor_type)
    Module, Parameter = owner.Module, owner.Parameter
    namespace = owner.copy_module(backend.nn, "torch.nn")
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
    context.state["nn_layer_adapters"] = owner.adapters
    context.state["nn_class_adapter"] = owner.adapt_class
    context.state["nn_frontend_owner"] = owner
    target.Module = Module
    target.nn = namespace
    return namespace
