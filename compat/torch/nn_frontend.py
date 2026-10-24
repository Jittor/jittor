"""Installation-owned NN types; implementation lives outside type factories."""
import types
import inspect
import itertools
from jittor._core.module import module_protocol_members, register_module_protocol

from .frontend import first_tensor, make_parameter_type, tensor_frontend
from .parameter_containers import make_parameter_containers
from .nn_adoption import adopt_owned_children


def module_setattr(module, name, value):
    owner = type(module)._nn_frontend_owner
    attributes = vars(module)
    if attributes.get("_native_parameter_construction", False):
        return owner.native_module.__setattr__(module, name, value)
    registered_parameter = name in attributes.get("_parameter_names", ())
    if owner.backend.core._is_tensor_carrier(value) and not name.startswith("_"):
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
    if name in attributes and isinstance(value, (owner.Module, owner.Parameter)) \
            and not (isinstance(attributes[name], owner.Module)
                     or owner.backend.core._is_tensor_carrier(attributes[name])) \
            and not registered_parameter:
        del attributes[name]
    if isinstance(value, owner.Module) or value is None:
        return owner.native_module.__setattr__(module, name, value)
    object.__setattr__(module, name, value)


def module_parameter_map_get(module):
    """Retain the native write-through view, including owned None slots.

    Rebuild on every access, just like the native getter. Container overrides
    remain the exact original mapping, and custom native parameter iteration
    keeps its existing real-parameter order.
    """
    owner = type(module)._nn_frontend_owner
    override = vars(module).get("_torch_parameter_mapping_override")
    if override is not None:
        return override
    parameters = owner.native_module._parameters.fget(module)
    registered = vars(module).get("_parameter_names", ())
    before = {}
    pending = []
    for name, value in vars(module).items():
        if name in parameters:
            if pending:
                before[name] = pending
                pending = []
        elif value is None and name in registered:
            pending.append(name)
    if not before and not pending:
        return parameters
    items = []
    for name, value in parameters.items():
        items.extend((slot, None) for slot in before.get(name, ()))
        items.append((name, value))
    items.extend((slot, None) for slot in pending)
    # Bypass only the returned snapshot's mutating hooks while inserting None
    # metadata. Its concrete class and subsequent item write-through stay intact.
    dict.clear(parameters)
    dict.update(parameters, items)
    return parameters


def module_parameter_map_set(module, value):
    owner = type(module)._nn_frontend_owner
    return owner.native_module._parameters.fset(module, value)


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
    # Native kernels allocate scratch/output buffers with jt.empty, which must
    # follow the operand rather than torch's default factory device. Explicit
    # torch.tensor/empty/... calls inside forward still establish their own
    # factory scope. With no placed input, allocation remains on the ambient
    # device rather than being forced onto the frontend's constructor default.
    like = _first_tensor(owner, args, kwargs)
    with tensor_frontend(owner.tensor_type, like=like, default_placement=False):
        return owner.module_call(module, *args, **kwargs)


def _first_tensor(owner, args, kwargs):
    return first_tensor(owner.tensor_type, args, kwargs)


def _conv1d_spatial(value, name):
    """Unpack a Torch one-axis sequence for the unchanged native constructor."""
    if isinstance(value, (tuple, list)):
        if len(value) != 1:
            raise ValueError("Conv1d %s must have one spatial entry" % name)
        return value[0]
    return value


def _conv1d_execute(module, value):
    """Public one-axis state drives the original native two-axis computation."""
    inner = module._conv[0]
    for name in ("stride", "padding", "dilation"):
        scalar = _conv1d_spatial(getattr(module, name), name)
        setattr(inner, name, scalar if name == "padding" and isinstance(scalar, str)
                else (scalar, 0 if name == "padding" else 1))
    inner.groups = module.groups
    inner.padding_mode = module.padding_mode
    # Like Torch, nonzero padding uses the widths cached at construction. Do not
    # silently recompute them from later writes to the descriptive padding field.
    # kernel_size/transposed/output_padding do not change Conv1d's weight math.
    owner = type(module)._nn_frontend_owner
    return owner.backend.nn.Conv1d.execute(module, value)


def _take_construction_dtype(owner, native, original, module, args, kwargs):
    supported = tuple(getattr(owner.backend.nn, name, None)
                      for name in ("Conv1d", "Conv3d", "LayerNorm"))
    if native not in supported:
        return None
    signature = inspect.signature(original)
    explicit = kwargs.get("dtype")
    if "dtype" not in signature.parameters:
        kwargs.pop("dtype", None)
    arguments = signature.bind_partial(module, *args, **kwargs)
    if "dtype" in signature.parameters:
        explicit = arguments.arguments.get("dtype")
    if explicit is None:
        from .frontend import _default_tensor_dtype
        dtype = _default_tensor_dtype(owner.backend)
    else:
        from .types import _dtype_to_str
        dtype = _dtype_to_str(explicit)
    if dtype not in ("float16", "bfloat16", "float32", "float64"):
        raise NotImplementedError(
            "Torch-owned native layer construction requires a supported real floating dtype")
    return dtype


def _cast_owned_constructor_parameters(owner, module, external, dtype):
    """Only new public weight/bias. Never rewrite external/hidden child graphs."""
    if dtype is None:
        return
    from jittor._core.dtypes import dtype_name
    replacements = {}
    for name in ("weight", "bias"):
        value = vars(module).get(name)
        if (id(value) in external or not isinstance(value, owner.tensor_type)
                or isinstance(value, owner.Parameter)):
            continue
        source_dtype = dtype_name(value.dtype)
        if source_dtype not in ("float16", "bfloat16", "float32", "float64"):
            continue
        if source_dtype != dtype:
            if id(value) not in replacements:
                replacements[id(value)] = value.cast(dtype)
            setattr(module, name, replacements[id(value)])


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
        parameter_dtype = _take_construction_dtype(
            owner, self.native, self.original, module, args, kwargs)
        padding_mode = self._take_torch_only_kwargs(kwargs)
        external = owner.external_objects(args, kwargs)
        conv1d = None
        if self.native is getattr(owner.backend.nn, "Conv1d", None):
            conv1d = inspect.signature(self.original).bind(module, *args, **kwargs)
            conv1d.apply_defaults()
            for name in ("kernel_size", "stride", "padding", "dilation"):
                conv1d.arguments[name] = _conv1d_spatial(conv1d.arguments[name], name)
            args, kwargs = conv1d.args[1:], conv1d.kwargs
        frozen = False
        if self.native.__name__ == "Embedding":
            arguments = inspect.signature(self.original).bind_partial(module, *args, **kwargs)
            frozen = bool(arguments.arguments.get("_freeze", False))
        with tensor_frontend(owner.tensor_type):
            previous = vars(module).get("_native_parameter_construction")
            object.__setattr__(module, "_native_parameter_construction", True)
            try:
                self.original(module, *args, **kwargs)
                _cast_owned_constructor_parameters(owner, module, external, parameter_dtype)
                if padding_mode is not None:
                    # torch layers expose what they were built with.
                    object.__setattr__(module, "padding_mode", padding_mode)
                adopt_owned_children(owner, module, external, frozen)
                if conv1d is not None:
                    for name in ("kernel_size", "stride", "padding", "dilation"):
                        value = conv1d.arguments[name]
                        object.__setattr__(module, name,
                            value if name == "padding" and isinstance(value, str) else (value,))
                    object.__setattr__(module, "output_padding", (0,))
                    object.__setattr__(module, "transposed", False)
                    # The native constructor first stores a boolean bias flag,
                    # before weight. Torch owns weight then bias registration.
                    # Move only this owned instance's actual object, not its math.
                    bias = vars(module).pop("bias")
                    object.__setattr__(module, "bias", bias)
                if self.native in (getattr(owner.backend.nn, "Linear", None),
                                   getattr(owner.backend.nn, "Conv1d", None)) \
                        and "bias" in vars(module) and module.bias is None:
                    # Ordinary None attributes are not parameters. These two
                    # reviewed constructors explicitly declare a disabled bias.
                    module.register_parameter("bias", None)
                if issubclass(self.native, getattr(owner.backend.nn, "BatchNorm", ())):
                    # Native Jittor keeps a non-persistent int32 vector for its
                    # checkpoint format. Torch owns a persistent scalar int64
                    # counter instead; change only this frontend instance.
                    counter = (owner.backend.array(0, dtype="int64").stop_grad()
                               if module.track_running_stats else None)
                    module.register_buffer("num_batches_tracked", counter)
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


def _layer_public_module(native):
    """Canonical public class locations, without mutating a native class.

    Only reviewed native exports are mapped. This is not a blanket promise
    that every native layer has a Torch-equivalent type or public API.
    """
    if not native.__module__.startswith("jittor.nn.modules."):
        return "torch.nn"
    family = {
        "Linear": "linear", "GELU": "activation", "LayerNorm": "normalization",
        "Conv1d": "conv", "Conv3d": "conv", "Embedding": "sparse",
        "EmbeddingBag": "sparse", "Sequential": "container",
    }.get(native.__name__)
    return "torch.nn.modules." + family if family is not None else "torch.nn"


def _modulelist_execute(module, *args, **kwargs):
    # A list is a registered container, not a Sequential forward graph.
    raise NotImplementedError("Module [ModuleList] is missing the required 'forward' function")


def _modulelist_getitem(module, index):
    import operator
    owner = type(module)._nn_frontend_owner
    selected = owner.backend.nn.Sequential.__getitem__(module,
        index if isinstance(index, slice) else operator.index(index))
    return type(module)(selected) if isinstance(index, slice) else selected


def _convnd_init(module, *args, **kwargs):
    """Real common owner of reviewed 1D/2D/3D convolution initialization."""
    owner = type(module)._nn_frontend_owner
    native = type(module)._torch_native_layer
    dimensions = {
        owner.backend.nn.Conv1d: 1, owner.backend.nn.Conv: 2,
        owner.backend.nn.Conv3d: 3,
    }[native]
    LayerInitializer(owner, native)(module, *args, **kwargs)
    object.__setattr__(module, "transposed", False)
    object.__setattr__(module, "output_padding", (0,) * dimensions)
    if module.bias is None:
        module.register_parameter("bias", None)


def _silu_init(module, inplace=False):
    owner = type(module)._nn_frontend_owner
    LayerInitializer(owner, owner.backend.nn.SiLU)(module, inplace=inplace)
    module.inplace = bool(inplace)


def _gelu_init(module, approximate="none"):
    owner = type(module)._nn_frontend_owner
    LayerInitializer(owner, owner.backend.nn.GELU)(module, approximate=approximate)
    module.approximate = approximate


def _layer_norm_execute(module, input):
    """The owned module shares the functional argument adapter, not its math."""
    from .installers.nn.functional import _layer_norm
    backend = type(module)._nn_frontend_owner.backend
    weight = module.weight if module.elementwise_affine else None
    bias = module.bias if backend.core._is_tensor_carrier(module.bias) else None
    return _layer_norm(input, module.normalized_shape, weight, bias, module.eps)


class NNFrontendOwner:
    """Own types and memoized namespace copies for a single installation."""
    def __init__(self, backend, tensor_type):
        self.backend = backend
        self.tensor_type = tensor_type
        self.native_module = backend.nn.Module
        self.Parameter = make_parameter_type(backend, tensor_type)
        members = module_protocol_members()
        members.update({
            "__module__": "torch.nn.modules.module", "__slots__": ("__dict__", "__weakref__"),
            "_frontend_tensor_type": tensor_type, "_nn_frontend_owner": self,
            "__setattr__": module_setattr, "__call__": module_call,
            "_parameters": property(module_parameter_map_get, module_parameter_map_set),
            "_call_depth": 0, "_hook_serial": itertools.count(),
            "_hook_table_created": False,
        })
        self.Module = type("Module", (), members)
        self.protocol = register_module_protocol(self.Module, tensor_type)
        self.module_call = self.protocol.module_call
        # Unmigrated, previously supported layers keep their original native
        # method precedence. This explicit legacy bridge makes no MRO-equality
        # claim and is never used by the reviewed Qwen classes below.
        self.LegacyModule = type("_LegacyNativeModule", (self.Module, self.native_module), {
            "__module__": __name__, "__slots__": (),
        })
        self.ConvNd = type("_ConvNd", (self.Module,), {
            "__module__": "torch.nn.modules.conv", "__slots__": (),
            "__init__": _convnd_init, "padding_mode": "zeros",
        })
        self.adapters = {self.native_module: self.Module}
        self.named_adapters = {}
        self.modules = {}

    def external_objects(self, args, kwargs):
        seen = set()
        pending = [args, kwargs]
        while pending:
            value = pending.pop()
            if id(value) in seen:
                continue
            seen.add(id(value))
            if isinstance(value, (self.Module, self.native_module)):
                pending.extend(vars(value).values())
            elif isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
        return seen

    def adapt_class(self, native, public_name=None):
        if (public_name == "ModuleList"
                and native is getattr(self.backend.nn, "Sequential", None)):
            key = (native, public_name)
            if key not in self.named_adapters:
                self.named_adapters[key] = type("ModuleList", (self.Module,), {
                    "__module__": "torch.nn.modules.container", "__slots__": (),
                    "__init__": LayerInitializer(self, native),
                    "_torch_native_layer": native,
                    "execute": _modulelist_execute, "__getitem__": _modulelist_getitem,
                    "__iter__": native.__iter__, "__len__": native.__len__,
                    "append": native.append, "extend": native.extend,
                    "insert": native.insert, "add_module": native.add_module,
                    "keys": native.keys, "values": native.values, "items": native.items,
                    "dfs": native.dfs, "named_children": native.named_children,
                    "_modules": native._modules,
                    "__setattr__": native.__setattr__, "__getattr__": native.__getattr__,
                    "_independent_nn_type_supported": True,
                })
            return self.named_adapters[key]
        known = self.adapters.get(native)
        if known is not None:
            return known
        owned = {
            getattr(self.backend.nn, "Linear", None): ("Linear", "linear"),
            getattr(self.backend.nn, "Embedding", None): ("Embedding", "sparse"),
            getattr(self.backend.nn, "SiLU", None): ("SiLU", "activation"),
            getattr(self.backend.nn, "GELU", None): ("GELU", "activation"),
            getattr(self.backend.nn, "LayerNorm", None): ("LayerNorm", "normalization"),
            getattr(self.backend.nn, "Conv1d", None): ("Conv1d", "conv"),
            getattr(self.backend.nn, "Conv", None): ("Conv2d", "conv"),
            getattr(self.backend.nn, "Conv3d", None): ("Conv3d", "conv"),
        }
        selected = owned.get(native)
        if selected is None:
            # Preserve support, do not relabel legacy inheritance as accepted.
            adapted = type(native.__name__, (native, self.LegacyModule), {
                "__module__": _layer_public_module(native), "__slots__": (),
                "__init__": LayerInitializer(self, native),
                "_torch_native_layer": native,
                "_independent_nn_type_supported": False,
            })
        else:
            name, family = selected
            members = {
                "__module__": "torch.nn.modules." + family, "__slots__": (),
                "__init__": LayerInitializer(self, native),
                "execute": native.execute, "_torch_native_layer": native,
                "_independent_nn_type_supported": True,
            }
            if name in ("Linear", "Embedding", "LayerNorm"):
                members["reset_parameters"] = native.reset_parameters
            if family == "conv":
                # _ConvNd actually owns the common constructor route, rather
                # than serving as an empty/virtual inspection-only base.
                members.pop("__init__")
                if name == "Conv1d":
                    members["execute"] = _conv1d_execute
                if name == "Conv2d":
                    members["_conv_forward"] = native._conv_forward
                bases = (self.ConvNd,)
            else:
                bases = (self.Module,)
                if name == "SiLU":
                    members.update(__init__=_silu_init, _function_name="silu")
                elif name == "GELU":
                    members.update(__init__=_gelu_init, _function_name="gelu")
                elif name == "LayerNorm":
                    members["execute"] = _layer_norm_execute
            adapted = type(name, bases, members)
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
                value = self.adapt_class(value, public_name=key)
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
    ParameterList, ParameterDict = make_parameter_containers(Module, Parameter, tensor_type)
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
