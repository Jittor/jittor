"""Explicit native mathematical delegates for the independent Torch frontend.

Each exported operation is a stable module object. Its implementation remains
the native backend's function, executed under the frontend's result-type and
autograd policy. Native-only utilities are not exposed through missing reads.
"""
import numbers
from contextlib import nullcontext

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
    _AUTOMIXED_PASSTHROUGH = frozenset(
        name for name in _NATIVE_NAMES
        if name not in ("matmul", "bmm", "baddbmm", "einsum"))

    @staticmethod
    def _first_var(value, var_type):
        if isinstance(value, var_type):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                found = NativeOperation._first_var(item, var_type)
                if found is not None:
                    return found
        if isinstance(value, dict):
            for item in value.values():
                found = NativeOperation._first_var(item, var_type)
                if found is not None:
                    return found
        return None

    def __call__(self, *args, **kwargs):
        import jittor
        context = get_install_context(jittor)
        implementation = context.state["native_torch_operations"][self._operation_key]
        if self._operation_key in ("mul", "multiply"):
            kwargs = dict(kwargs)
            for position, name in enumerate(("input", "other")):
                if name in kwargs:
                    if len(args) > position:
                        raise TypeError("%s received %s more than once" % (self._operation_key, name))
                    if len(args) == position:
                        args += (kwargs.pop(name),)
            if len(args) == 2 and not kwargs:
                from .installers.tensor.method_api import _tensor_mul
                left, right = args
                if isinstance(left, context.state["Var"]):
                    return _tensor_mul(left, right)
                if isinstance(right, context.state["Var"]):
                    return _tensor_mul(right, left)
        defaults = self._TORCH_DEFAULTS.get(self._operation_key)
        if defaults:
            kwargs = dict(kwargs)
            for name, value in defaults.items():
                kwargs.setdefault(name, value)
        device = None
        if self._operation_key in ("fft.fftfreq", "fft.rfftfreq"):
            kwargs = dict(kwargs)
            device = kwargs.pop("device", None)
        unique_operation = self._operation_key in ("unique", "unique_consecutive")
        like = self._first_var((args, kwargs), context.native_backend.Var)
        with tensor_frontend(context.state["Var"], device=device, like=like,
                             default_placement=False if like is not None else True):
            if self._operation_key == "rsqrt":
                return _torch_rsqrt(context, implementation, *args, **kwargs)
            if self._operation_key in ("div", "divide"):
                return _torch_div(context, *args, **kwargs)
            if isinstance(implementation, type) and issubclass(implementation, context.native_backend.Function):
                implementation = implementation()
            if unique_operation:
                result = implementation(*args, **kwargs)
                # Torch always represents inverse indices and counts as int64.
                if isinstance(result, tuple):
                    result = (result[0],) + tuple(value.int64() for value in result[1:])
                return result
            if self._operation_key == "baddbmm":
                from .amp import _autocast_cast_inputs
                values = list(args) + list(kwargs.values())
                cast_values = _autocast_cast_inputs(*values)
                args = tuple(cast_values[:len(args)])
                kwargs = dict(zip(kwargs, cast_values[len(args):]))
            if self._operation_key in self._AUTOMIXED_PASSTHROUGH:
                from .amp import _preserve_pointwise_dtype
                amp_reg = (jittor.amp_flags.keep_reduce
                           if self._operation_key == "mean" else 0)
                scope = (jittor.flag_scope(amp_reg=amp_reg)
                         if _preserve_pointwise_dtype() else nullcontext())
                with scope:
                    return implementation(*args, **kwargs)
            from .amp import _autocast_scope_for_inputs
            with _autocast_scope_for_inputs(*args, *kwargs.values()):
                return implementation(*args, **kwargs)

    def __reduce__(self):
        return operation, (self._operation_key,)


_TORCH_RSQRT_ACL_FUNCTIONS = {}


def _torch_rsqrt_acl(context, implementation, input):
    """Use the ACL composition for forward and Torch's rsqrt VJP order."""
    native = context.native_backend
    key = (native, implementation)
    function_type = _TORCH_RSQRT_ACL_FUNCTIONS.get(key)
    if function_type is None:
        class _TorchRsqrtACL(native.Function):
            def execute(self, value):
                self.input_dtype = value.dtype
                self.output = implementation(value)
                return self.output

            def grad(self, upstream):
                with native.flag_scope(amp_reg=0):
                    cube = self.output.pow(3).cast(self.input_dtype)
                    return (-0.5 * upstream) * cube

        function_type = _TorchRsqrtACL
        _TORCH_RSQRT_ACL_FUNCTIONS[key] = function_type
    return function_type.apply(input)


def _torch_rsqrt(context, implementation, input):
    """Preserve Torch's CUDA autocast and ACL rsqrt backward contracts."""
    from jittor._core.dtypes import dtype_name
    from .amp import autocast_is_enabled
    native = context.native_backend
    if (isinstance(input, native.Var)
            and native.core.dispatch_context([input])[0] == "acl"
            and dtype_name(input.dtype) in (
                "float16", "bfloat16", "float32", "float64")):
        with native.flag_scope(amp_reg=0):
            return _torch_rsqrt_acl(context, implementation, input)
    if (hasattr(input, "dtype")
            and dtype_name(input.dtype) in ("float16", "bfloat16")
            and isinstance(input, native.Var)
            and native.core.dispatch_context([input])[0] == "cuda"
            and autocast_is_enabled("cuda")):
        input = input.float32()
    with native.flag_scope(amp_reg=0):
        return implementation(input)


def _integral_operand(value, var_type):
    if isinstance(value, var_type):
        return str(value.dtype).removeprefix("torch.").startswith(("int", "uint"))
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _torch_div(context, input, other, *, rounding_mode=None, out=None):
    if rounding_mode not in (None, "trunc", "floor"):
        raise RuntimeError(
            "div expected rounding_mode to be one of None, 'trunc', or 'floor'"
        )
    var_type = context.native_backend.Var
    input_is_var = isinstance(input, var_type)
    other_is_var = isinstance(other, var_type)
    inputs_are_integral = (
        _integral_operand(input, var_type)
        and _integral_operand(other, var_type)
    )
    if not input_is_var and not other_is_var:
        input = context.target_namespace.tensor(input)
    result = input / other
    if rounding_mode == "trunc":
        result = result.trunc()
    elif rounding_mode == "floor":
        result = result.floor()
    if rounding_mode is not None and inputs_are_integral:
        if input_is_var and other_is_var:
            dtype = context.target_namespace.promote_types(input.dtype, other.dtype)
        elif other_is_var:
            dtype = other.dtype
        else:
            dtype = input.dtype
        result = result.to(dtype=dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


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
