"""Explicit native mathematical delegates for the independent Torch frontend.

Each exported operation is a stable module object. Its implementation remains
the native backend's function, executed under the frontend's result-type and
autograd policy. Native-only utilities are not exposed through missing reads.
"""
from .api_delegates import bind_delegates
from .context import get_install_context
from .frontend import tensor_frontend
from .fidelity import Fidelity, register_fidelity


_NATIVE_NAMES = (
    "acos", "acosh", "arccos", "arccosh", "arcsin", "arcsinh", "arctan",
    "arctan2", "arctanh", "asin", "asinh", "atan", "atan2", "atanh",
    "atleast_1d", "atleast_2d", "atleast_3d", "baddbmm", "bitwise_and",
    "bitwise_not", "bitwise_or", "bitwise_xor", "block_diag", "bmm",
    "cartesian_prod", "ceil", "chunk", "cos", "cosh", "cross", "ctc_loss",
    "cummax", "cummin", "deg2rad", "detach", "diag", "digamma", "div",
    "divide", "einsum", "erf", "erfinv", "exp", "expm1", "flatten", "flip",
    "floor", "floor_divide", "gather", "greater", "greater_equal", "histc",
    "hypot", "igamma", "index_add", "index_fill", "isfinite", "isinf",
    "isnan", "isneginf", "isposinf", "kthvalue", "less", "less_equal",
    "lgamma", "log", "log2", "logical_and", "logical_not", "logical_or",
    "logical_xor", "matmul", "maximum", "mean", "meshgrid", "minimum",
    "mul", "multiply", "negative", "permute", "pow", "prod", "rad2deg",
    "reshape", "roll", "round", "rsqrt", "scatter", "scatter_add",
    "scatter_reduce", "searchsorted", "sigmoid", "sin", "sinh", "split",
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

    def __call__(self, *args, **kwargs):
        import jittor
        context = get_install_context(jittor)
        implementation = context.state["native_torch_operations"][self._operation_key]
        device = None
        if self._operation_key in ("fft.fftfreq", "fft.rfftfreq"):
            kwargs = dict(kwargs)
            device = kwargs.pop("device", None)
        with tensor_frontend(context.state["Var"], device=device):
            if isinstance(implementation, type) and issubclass(implementation, context.native_backend.Function):
                implementation = implementation()
            return implementation(*args, **kwargs)

    def __reduce__(self):
        return operation, (self._operation_key,)


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
