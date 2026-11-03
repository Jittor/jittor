"""Explicit native mathematical delegates for the independent Torch frontend.

Each exported operation is a stable module object. Its implementation remains
the native backend's function, executed under the frontend's result-type and
autograd policy. Native-only utilities are not exposed through missing reads.
"""
from .api_delegates import bind_delegates
from .context import get_install_context
from .frontend import tensor_frontend
from .fidelity import Fidelity, register_fidelity
from .amp import _preserve_autocast_arithmetic, _cuda_autocast_fp32_input, _autocast_fp32_call
from jittor._core.dtypes import dtype_name as _jittor_dtype_name


_NATIVE_NAMES = (
    "acos", "acosh", "arccos", "arccosh", "arcsin", "arcsinh", "arctan",
    "arctan2", "arctanh", "asin", "asinh", "atan", "atan2", "atanh",
    "atleast_1d", "atleast_2d", "atleast_3d", "baddbmm", "bitwise_and",
    "bitwise_not", "bitwise_or", "bitwise_xor", "block_diag", "bmm",
    "cartesian_prod", "ceil", "chunk", "cos", "cosh", "cross", "ctc_loss",
    "cummax", "cummin", "deg2rad", "detach", "diag", "digamma",
    "einsum", "erf", "erfinv", "exp", "expm1", "flatten", "flip",
    "floor", "floor_divide", "fmod", "gather", "greater", "greater_equal", "histc",
    "hypot", "igamma", "index_add", "index_fill", "isfinite", "isinf",
    "isnan", "isneginf", "isposinf", "kthvalue", "less", "less_equal",
    "lgamma", "log", "log2", "logical_and", "logical_not", "logical_or",
    "logical_xor", "matmul", "maximum", "mean", "meshgrid", "minimum",
    "mul", "multiply", "negative", "permute", "pow", "prod", "rad2deg",
    "reshape", "roll", "round", "rsqrt", "scatter", "scatter_add",
    "scatter_reduce", "searchsorted", "sigmoid", "sin", "sinc", "sinh", "split",
    "sqrt", "squeeze", "sub", "subtract", "t", "tan", "tanh", "transpose",
    "unbind", "unique", "unique_consecutive", "unsqueeze",
)

_NATIVE_MODULE_NAMES = {
    "fft": ("fft", "ifft", "fft2", "ifft2", "fftn", "ifftn", "rfft", "irfft",
            "fftshift", "ifftshift", "fftfreq", "rfftfreq"),
    "linalg": ("svd", "svdvals", "eig", "eigh", "eigvalsh", "inv", "inv_ex", "pinv",
               "matrix_power", "matrix_rank", "matrix_norm", "vector_norm", "norm",
               "cond", "det", "slogdet", "cholesky", "solve", "qr"),
}


class NativeOperation:
    """An installation-independent identity for one native mathematical API."""

    def __init__(self, name):
        self._operation_key = name
        self.__name__ = self.__qualname__ = name.replace(".", "_")
        self.__module__ = __name__
        self.__doc__ = "Native %s under the active Torch frontend policy." % name

    #: Where torch's default differs from the native one. ``linalg.svd``
    #: returns the full decomposition unless asked otherwise; jittor's returns
    #: the thin one. Forwarding without this, ``torch.linalg.svd(B)`` on a
    #: 5x3 input answered with a (5,3) ``U`` where torch and numpy both give
    #: (5,5) -- a quietly different answer, not an error.
    _TORCH_DEFAULTS = {"linalg.svd": {"full_matrices": True}}

    def __call__(self, *args, **kwargs):
        import jittor
        context = get_install_context(jittor)
        implementation = context.state["native_torch_operations"][self._operation_key]
        if self._operation_key in ("pow", "rsqrt", "sqrt", "log", "exp"):
            if "input" in kwargs or self._operation_key == "pow" and "exponent" in kwargs:
                kwargs = dict(kwargs)
                if "input" in kwargs:
                    if args or "x" in kwargs:
                        raise TypeError("%s received input more than once" % self._operation_key)
                    args = (kwargs.pop("input"),)
                if self._operation_key == "pow" and "exponent" in kwargs:
                    if len(args) > 1 or "y" in kwargs:
                        raise TypeError("pow received exponent more than once")
                    if args:
                        args += (kwargs.pop("exponent"),)
                    else:
                        kwargs["y"] = kwargs.pop("exponent")
        if self._operation_key in ("mul", "multiply"):
            # Public aliases share the Tensor scalar-opmath owner. Keep other
            # native signatures (including unsupported out=) on their delegate.
            kwargs = dict(kwargs)
            for position, name in enumerate(("input", "other")):
                if name in kwargs:
                    if len(args) > position:
                        raise TypeError("%s received %s more than once" % (self._operation_key, name))
                    if len(args) == position:
                        args += (kwargs.pop(name),)
            if len(args) == 2 and not kwargs:
                from .installers.tensor.method_api import _cuda_scalar_multiply
                left, right = args
                if isinstance(left, context.state["Var"]):
                    result = _cuda_scalar_multiply(left, right)
                elif isinstance(right, context.state["Var"]):
                    result = _cuda_scalar_multiply(right, left)
                else:
                    result = None
                if result is not None:
                    return result
        defaults = self._TORCH_DEFAULTS.get(self._operation_key)
        if defaults:
            kwargs = dict(kwargs)
            for name, value in defaults.items():
                kwargs.setdefault(name, value)
        device = None
        if self._operation_key in ("fft.fftfreq", "fft.rfftfreq"):
            kwargs = dict(kwargs)
            device = kwargs.pop("device", None)
        # Mathematical operations follow their inputs; only frequency-vector
        # factories choose the ambient/default construction device. Native
        # composites such as FFT allocate intermediate stack buffers here.
        like = next((value for value in (*args, *kwargs.values())
                     if isinstance(value, context.native_backend.Var)), None)
        frequency_factory = self._operation_key in ("fft.fftfreq", "fft.rfftfreq")
        with tensor_frontend(context.state["Var"], device=device, like=like,
                             default_placement=frequency_factory):
            if self._operation_key in ("log", "exp"):
                # Native exp computes CPU half inputs in FP32. Torch keeps
                # the public output dtype; retain native opmath and narrow
                # only the result, without accepting extra native signatures.
                if (self._operation_key == "exp" and args
                        and _jittor_dtype_name(args[0].dtype) in ("float16", "bfloat16")
                        and str(args[0].device).split(":")[0] == "cpu"):
                    return implementation(*args, **kwargs).cast(args[0].dtype)
                return _autocast_fp32_call(implementation, *args,
                                          eligible=kwargs.get("out") is None, **kwargs)
            if isinstance(implementation, type) and issubclass(implementation, context.native_backend.Function):
                implementation = implementation()
            if self._operation_key in ("pow", "rsqrt", "sqrt", "mul", "multiply", "sub", "subtract") \
                    and _preserve_autocast_arithmetic():
                if self._operation_key in ("pow", "rsqrt"):
                    args = tuple(_cuda_autocast_fp32_input(value) for value in args)
                    kwargs = {name: _cuda_autocast_fp32_input(value)
                              for name, value in kwargs.items()}
                with jittor.flag_scope(amp_reg=jittor.amp_flags.disabled):
                    return implementation(*args, **kwargs)
            return implementation(*args, **kwargs)

    def __reduce__(self):
        return operation, (self._operation_key,)


def tensor_log(input):
    return operation("log")(input)


def tensor_rsqrt(input):
    return operation("rsqrt")(input)


def tensor_exp(input):
    return operation("exp")(input)


def _tensor_unary_inplace(input, name):
    if not input.is_floating_point() and not input.is_complex():
        raise RuntimeError("result type Float can't be cast to the desired output type "
                           + str(input.dtype))
    from .installers.tensor.method_api import _ip
    import jittor
    context = get_install_context(jittor)
    implementation = context.state["native_torch_operations"][name]
    value = _autocast_fp32_call(implementation, input, eligible=False)
    return _ip(input, value.cast(input.dtype))


def tensor_log_(input):
    return _tensor_unary_inplace(input, "log")


def tensor_rsqrt_(input):
    return _tensor_unary_inplace(input, "rsqrt")


_OPERATIONS = {name: NativeOperation(name) for name in _NATIVE_NAMES}
_OPERATIONS.update((module + "." + name, NativeOperation(module + "." + name))
                   for module, names in _NATIVE_MODULE_NAMES.items() for name in names)
globals().update((value.__name__, value) for value in _OPERATIONS.values())


def operation(name):
    return _OPERATIONS[name]


def install(context):
    """Fill declared native-backed APIs, then close the bootstrap-only fallback."""
    from .namespace import TorchNamespace
    target = context.target_namespace
    if not isinstance(target, TorchNamespace):
        return
    delegates = dict(context.state.get("native_torch_operations", {}))
    for name in _NATIVE_NAMES:
        if name in vars(target):
            continue
        implementation = getattr(context.native_backend, name, None)
        if callable(implementation):
            delegates[name] = implementation
            api = _OPERATIONS[name]
            setattr(target, name, api)
            register_fidelity("torch." + name, api, Fidelity.APPROXIMATE,
                              "Native mathematics with frontend result types; native signatures, dtype and edge-case limitations apply")
    for module_name, names in _NATIVE_MODULE_NAMES.items():
        target_module = vars(target).get(module_name)
        native_module = getattr(context.native_backend, module_name, None)
        if target_module is None or native_module is None:
            continue
        if target_module is native_module:
            raise RuntimeError("independent Torch module aliases native " + module_name)
        for name in names:
            implementation = getattr(native_module, name, None)
            if not callable(implementation):
                continue
            key = module_name + "." + name
            api = _OPERATIONS[key]
            existing = vars(target_module).get(name)
            if existing is not None and existing is not implementation and existing is not api:
                continue
            delegates[key] = implementation
            setattr(target_module, name, api)
            register_fidelity("torch." + key, api, Fidelity.APPROXIMATE,
                              "Shared native implementation in an owned module and frontend scope; native dtype/backend limitations remain")
    bind_delegates(context, "native_torch_operations", delegates)
    # Version identity must not fall through to the backend after publication.
    if "__version__" not in vars(target):
        target.__version__ = getattr(target, "__jittor_version__", "")
    target.__all__ = tuple(sorted(name for name in vars(target) if not name.startswith("_")))
    target._seal()


__all__ = [*_NATIVE_NAMES, "NativeOperation", "operation", "install"]
