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
    # torch registers a submodule or parameter when one is first assigned. A
    # name that held something else until then -- diffusers writes
    # `self.mid_block = None` and builds the block later -- is registered at
    # that point, after everything registered in between. Attributes keep the
    # position of their first assignment, so move the name to the end: without
    # it `named_parameters()` listed an SD UNet's mid block before its up
    # blocks, where torch lists it after, and anything pairing parameters by
    # position paired the wrong ones.
    if name in attributes and isinstance(value, (owner.native_module, owner.Parameter)) \
            and not isinstance(attributes[name], (owner.native_module, owner.backend.Var)):
        del attributes[name]
    # Native Sequential keeps registered children in ``layers`` and its
    # named_children() only traverses that mapping. Torch permits attaching a
    # child with setattr; route it through add_module so it is visible to
    # parameters(), state_dict(), and forward traversal.
    sequential_type = getattr(owner.backend.nn, "Sequential", None)
    if (sequential_type is not None and isinstance(module, sequential_type)
            and isinstance(value, owner.native_module) and not name.startswith("_")
            and "layers" in attributes):
        module.add_module(name, value)
        return
    object.__setattr__(module, name, value)


#: The native module call (`src/bindings/pyjt/py_module_call.h`), once the
#: nn installer has bound it: the same scope and dispatch, without the frames.
_NATIVE_CALL = None


def module_call(module, *args, **kwargs):
    native = _NATIVE_CALL
    if native is not None:
        return native(module, args, kwargs)
    return python_module_call(module, *args, **kwargs)


def python_module_call(module, *args, **kwargs):
    owner = type(module)._nn_frontend_owner
    # A forward follows its inputs' device; only constructors use the default.
    # The first tensor argument is the reference, as `*_like` uses its source:
    # with none (or one without an explicit placement) allocation is left to
    # the ambient device, and never forced onto the default one.
    with tensor_frontend(owner.tensor_type, like=_first_tensor(owner, args, kwargs),
                         default_placement=False):
        return owner.native_module.__call__(module, *args, **kwargs)


def _first_tensor(owner, args, kwargs):
    # No backend, no placement to follow -- `tensor_frontend` passes those
    # straight through too.
    backend = getattr(owner.tensor_type, "_frontend_backend", None)
    if backend is None:
        return None
    var_type = backend.Var
    for value in args:
        if isinstance(value, var_type):
            return value
    for value in kwargs.values():
        if isinstance(value, var_type):
            return value
    return None


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
        padding_mode = self._take_torch_only_kwargs(kwargs)
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
                if padding_mode is not None:
                    # torch layers expose what they were built with.
                    object.__setattr__(module, "padding_mode", padding_mode)
                adopt_owned_children(owner, module, external, frozen)
            finally:
                if previous is None:
                    vars(module).pop("_native_parameter_construction", None)
                else:
                    object.__setattr__(module, "_native_parameter_construction", previous)

    def _take_torch_only_kwargs(self, kwargs):
        """Remove torch-only keyword arguments the native ``__init__`` lacks.

        Torch's convolution layers take ``padding_mode``. The native
        Conv1d/Conv2d/Conv3d implement it and receive it unchanged; a native
        layer without the parameter -- the transposed convolutions, for which
        torch itself accepts only ``'zeros'`` -- has it removed and recorded
        on the module, and any other mode is refused rather than ignored.
        """
        if "padding_mode" not in kwargs:
            return None
        if "padding_mode" in inspect.signature(self.original).parameters:
            return None
        mode = kwargs.pop("padding_mode")
        if mode != "zeros":
            raise ValueError('Only "zeros" padding mode is supported for torch.nn.%s, got %r'
                             % (self.native.__name__, mode))
        return mode


def module_list_execute(module, *args, **kwargs):
    """Torch ModuleList stores modules but has no callable forward."""
    raise NotImplementedError("ModuleList is missing the required forward function")


def make_distinct_module_list(owner, native_sequential):
    """Separate Torch ModuleList from Jittor's Sequential/ModuleList alias."""
    return type("ModuleList", (native_sequential, owner.Module), {
        "__module__": "torch.nn", "__slots__": (),
        "__init__": LayerInitializer(owner, native_sequential),
        "_torch_native_layer": native_sequential,
        "execute": module_list_execute,
    })


class NNFrontendOwner:
    """Own types and memoized namespace copies for a single installation."""
    def __init__(self, backend, tensor_type):
        self.backend = backend
        self.tensor_type = tensor_type
        self.native_module = backend.nn.Module
        self.Parameter = make_parameter_type(backend, tensor_type)
        # Torch's explicit ``super(nn.Module, self).__init__(...)`` must cross a
        # Torch-owned base before reaching the native Jittor class.  The base
        # receives the cooperative initializer without mutating jt.Module.
        self.cooperative_module = type("_TorchNativeModule", (self.native_module,), {
            "__module__": "jittor.compat.torch.nn_frontend", "__slots__": (),
        })
        self.Module = type("Module", (self.cooperative_module,), {
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
        namespace = {
            "__module__": "torch.nn", "__slots__": (),
            "__init__": LayerInitializer(self, native),
            "_torch_native_layer": native,
        }
        adapted = type(native.__name__, (native, self.Module), namespace)
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
    # Native Jittor aliases ModuleList to Sequential. Torch distinguishes them:
    # a Sequential must not match isinstance(x, nn.ModuleList), and a ModuleList
    # is only a container. Keep native Jittor's alias untouched.
    native_sequential = getattr(backend.nn, "Sequential", None)
    if native_sequential is not None and getattr(backend.nn, "ModuleList", None) is native_sequential:
        ModuleList = make_distinct_module_list(owner, native_sequential)
        namespace.ModuleList = ModuleList
        namespace.modules.ModuleList = ModuleList
        if hasattr(namespace.modules, "container"):
            namespace.modules.container.ModuleList = ModuleList
    ParameterList, ParameterDict = make_parameter_containers(Module, Parameter, backend.Var)
    namespace.ParameterList = ParameterList
    namespace.ParameterDict = ParameterDict
    namespace.modules.ParameterList = ParameterList
    namespace.modules.ParameterDict = ParameterDict
    namespace.modules.parameter.ParameterList = ParameterList
    namespace.modules.parameter.ParameterDict = ParameterDict
    # Parameter's public module may not exist in the native tree yet.
    parameter_module = types.ModuleType("torch.nn.parameter")
    setattr(parameter_module, "Parameter", Parameter)
    setattr(parameter_module, "ParameterList", ParameterList)
    setattr(parameter_module, "ParameterDict", ParameterDict)
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
