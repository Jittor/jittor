"""torch.nn.init on jittor.

Its own module because it is its own family: a dozen in-place initialisers that
share one assignment helper and nothing else with the rest of the nn surface.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import jittor as jt

from ..context import registry_for
from ..fidelity import Fidelity, register_api_bindings
from ...diagnostics import EXPECTED, swallowed


import types as _types_init
import jittor as _jt2
from jittor.init.scaling import _kaiming_uniform_
from ..context import get_install_context
from types import MappingProxyType
import jittor.init as _native_init

_NATIVE_INITIALIZERS = MappingProxyType({
    name: getattr(_native_init, name)
    for name in ("kaiming_normal_", "gauss_", "xavier_uniform_", "xavier_gauss_",
                 "xavier_normal_", "relu_invariant_gauss_", "invariant_uniform_")
    if hasattr(_native_init, name)
})

def _native_initializer(name, tensor, *args, **kwargs):
    was_trainable = hasattr(tensor, "is_stop_grad") and not tensor.is_stop_grad()
    result = _NATIVE_INITIALIZERS[name](tensor, *args, **kwargs)
    if was_trainable and hasattr(tensor, "start_grad"):
        tensor.start_grad()
    return result

def _assign(tensor, value):
    # Preserve the tensor's grad-tracking: jittor's .assign() adopts the
    # source var's stop_grad flag, and our `value` (jt.normal/zeros/...) is
    # stop_grad, which would silently freeze the parameter. Re-enable grad
    # unless the param was explicitly stop-grad before.
    was_trainable = not tensor.is_stop_grad()
    # Native assign owns retained-view writeback at every depth.
    tensor.assign(value)
    if was_trainable:
        tensor.start_grad()
    return tensor


def _not_var(t):
    return not isinstance(t, _jt2.Var)


def normal_(tensor, mean=0.0, std=1.0, generator=None):
    if _not_var(tensor): return tensor
    return _assign(tensor, _jt2.normal(float(mean), float(std), tensor.shape).cast(_jittor_dtype_name(tensor.dtype)))


def uniform_(tensor, a=0.0, b=1.0, generator=None, *, low=None, high=None):
    if low is not None:
        if a != 0.0 and a != low:
            raise TypeError("uniform_ received conflicting values for a and low")
        a = low
    if high is not None:
        if b != 1.0 and b != high:
            raise TypeError("uniform_ received conflicting values for b and high")
        b = high
    if _not_var(tensor): return tensor
    return _assign(tensor, (_jt2.rand(tensor.shape) * (b - a) + a).cast(_jittor_dtype_name(tensor.dtype)))


def zeros_(tensor):
    if _not_var(tensor): return tensor
    return _assign(tensor, _jt2.zeros(tensor.shape, tensor.dtype))


def kaiming_uniform_(tensor, a=0, mode='fan_in',
                     nonlinearity='leaky_relu', generator=None):
    return _kaiming_uniform_(tensor, uniform_, a, mode, nonlinearity, generator)


def ones_(tensor):
    if _not_var(tensor): return tensor
    return _assign(tensor, _jt2.ones(tensor.shape, tensor.dtype))


def constant_(tensor, val=0.0, *, value=None):
    if value is not None:
        if val != 0.0 and val != value:
            raise TypeError("constant_ received conflicting values for val and value")
        val = value
    if _not_var(tensor): return tensor
    return _assign(tensor, _jt2.ones(tensor.shape, tensor.dtype) * val)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
    if _not_var(tensor): return tensor
    import numpy as _np
    # simple clamp of a normal sample (no scipy dependency)
    x = _np.random.normal(mean, std, tensor.shape).astype("float32")
    x = _np.clip(x, mean + a * std, mean + b * std)
    return _assign(tensor, _jt2.array(x).cast(_jittor_dtype_name(tensor.dtype)))


def _fan(t):
    sh = t.shape
    if len(sh) < 2:
        return sh[0], sh[0]
    num_input_fmaps, num_output_fmaps = sh[1], sh[0]
    rf = 1
    for s in sh[2:]:
        rf *= s
    return num_input_fmaps * rf, num_output_fmaps * rf


def _calculate_correct_fan(tensor, mode):
    mode = str(mode).lower()
    if mode not in ("fan_in", "fan_out"):
        raise ValueError("Mode %s not supported, please use fan_in or fan_out" % mode)
    fan_in, fan_out = get_install_context(jt).target_namespace.nn.init._calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def _dirac(tensor, groups=1):
    """Identity ("Dirac delta") init for 3/4/5-D conv weights.

    Was `lambda t, *a, **k: t`: an outright no-op, so a model relying
    on dirac_ to start as an identity mapping started from whatever
    uninitialised memory the tensor held instead.
    """
    import numpy as _np
    shape = [int(d) for d in tensor.shape]
    if len(shape) not in (3, 4, 5):
        raise ValueError(
            "Only tensors with 3, 4, or 5 dimensions are supported")
    groups = int(groups)
    out_channels = shape[0]
    if out_channels % groups != 0:
        raise ValueError("dim 0 must be divisible by groups")
    out_per_group = out_channels // groups
    min_dim = min(out_per_group, shape[1])
    arr = _np.zeros(shape, dtype=_np.float64)
    centre = tuple(slice(d // 2, d // 2 + 1) for d in shape[2:])
    for g in range(groups):
        for d in range(min_dim):
            arr[(g * out_per_group + d, d) + centre] = 1.0
    tensor.assign(jt.array(arr).cast(_jittor_dtype_name(tensor.dtype)))
    return tensor


def _orth(t, gain=1.0):
    import numpy as _np
    sh = t.shape
    flat = (sh[0], int(t.numel() // sh[0])) if len(sh) > 1 else (sh[0], 1)
    a = _np.random.randn(*flat)
    q, r = _np.linalg.qr(a)
    q = q * _np.sign(_np.diag(r))
    if flat[0] < flat[1]:
        q = q.T
    t.assign(jt.array((gain * q).reshape(sh).astype("float32")))
    return t


def _sparse(tensor, sparsity, std=0.01, generator=None):
    """Sparse init for a 2-D weight: was a no-op, now really sparse."""
    import numpy as _np
    shape = [int(d) for d in tensor.shape]
    if len(shape) != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    rows, cols = shape
    num_zeros = int(_np.ceil(float(sparsity) * rows))
    rng = _np.random.default_rng()
    arr = rng.normal(0.0, float(std), size=(rows, cols))
    for col in range(cols):
        zero_rows = rng.permutation(rows)[:num_zeros]
        arr[zero_rows, col] = 0.0
    tensor.assign(jt.array(arr).cast(_jittor_dtype_name(tensor.dtype)))
    return tensor


class _GuardedInit(_types_init.ModuleType):
    _protected = set()
    def __setattr__(self, key, value):
        if key in self._protected:
            name = getattr(value, "__name__", "")
            if (not callable(value)) or name in ("_skip_init", "skip_init", "<lambda>"):
                return
        object.__setattr__(self, key, value)


def kaiming_normal_(tensor, *args, **kwargs):
    return _native_initializer('kaiming_normal_', tensor, *args, **kwargs)


def gauss_(tensor, *args, **kwargs):
    return _native_initializer('gauss_', tensor, *args, **kwargs)


def xavier_uniform_(tensor, *args, **kwargs):
    return _native_initializer('xavier_uniform_', tensor, *args, **kwargs)


def xavier_gauss_(tensor, *args, **kwargs):
    return _native_initializer('xavier_gauss_', tensor, *args, **kwargs)


def xavier_normal_(tensor, *args, **kwargs):
    return _native_initializer('xavier_normal_', tensor, *args, **kwargs)


def relu_invariant_gauss_(tensor, *args, **kwargs):
    return _native_initializer('relu_invariant_gauss_', tensor, *args, **kwargs)


def invariant_uniform_(tensor, *args, **kwargs):
    return _native_initializer('invariant_uniform_', tensor, *args, **kwargs)


_INITIALIZER_APIS = {
    'kaiming_normal_': kaiming_normal_,
    'kaiming_uniform_': kaiming_uniform_,
    'gauss_': gauss_,
    'xavier_uniform_': xavier_uniform_,
    'xavier_gauss_': xavier_gauss_,
    'xavier_normal_': xavier_normal_,
    'relu_invariant_gauss_': relu_invariant_gauss_,
    'invariant_uniform_': invariant_uniform_,
}

def _install_init_aliases(registry=None):
    _registry = registry_for(jt, registry)
    _modules = _registry.module_map
    _tensor_type = _registry.target_namespace.Var
    _target_nn = _registry.target_namespace.nn
    import jittor.init as _native_init
    import jittor as _jt2
    import types as _types_init
    if _registry.target_namespace is _registry.native_backend:
        _init = _native_init
    else:
        # Only the spelling/adaptation namespace is copied. Native initializer
        # callables retain their physical owners and the single mathematics.
        _init = _types_init.ModuleType("torch.nn.init")
        _init.__package__ = "torch.nn"
        for _name, _value in vars(_native_init).items():
            if not _name.startswith("__"):
                setattr(_init, _name, _value)
    from jittor.init.scaling import _kaiming_uniform_
    # torch-style in-place initializers, tolerant of torch kwargs (e.g.
    # `generator=`, which jittor ignores). Each writes into `tensor` in place.

    # in-place inits are sometimes called on a NON-Var constant: jittor represents a
    # disabled affine term (e.g. LayerNorm(bias=False) -> self.bias = 0.0) as a Python
    # scalar, and a model's _init_weights may still call init.zeros_(module.bias) on it.
    # Such a constant isn't a learnable parameter, so initializing it is a no-op.
    # override with the tolerant versions (also covers jittor's own names)
    for name, fn in [("normal_", normal_), ("uniform_", uniform_),
                     ("kaiming_uniform_", kaiming_uniform_),
                     ("zeros_", zeros_), ("ones_", ones_), ("constant_", constant_),
                     ("trunc_normal_", trunc_normal_)]:
        setattr(_init, name, fn)
    # jittor's native kaiming/xavier/gauss initializers do `var.assign(src)` without
    # re-enabling grad. Under transformers' @torch.no_grad() weight init, `src` is
    # stop_grad, so .assign() silently FREEZES the parameter -- Conv2d/Linear inited
    # with kaiming (resnet/regnet/...) end up stop_grad and get zero weight grads
    # (forward stays exact, so it's invisible until you train/check gradients). Wrap
    # them with the same grad-preserving guard used by _assign() above: a no-op for
    # already-frozen params, so it can't regress anything.
    for _nm in ("kaiming_normal_", "kaiming_uniform_", "gauss_",
                "xavier_uniform_", "xavier_gauss_", "xavier_normal_",
                "relu_invariant_gauss_", "invariant_uniform_"):
        if hasattr(_init, _nm):
            setattr(_init, _nm, _INITIALIZER_APIS[_nm])
            if hasattr(_tensor_type, _nm):
                setattr(_tensor_type, _nm, getattr(_init, _nm))
    # keep jittor's good xavier/kaiming; add torch-name aliases for the rest
    aliases = {"xavier_normal_": "xavier_gauss_"}
    for tname, jname in aliases.items():
        if not hasattr(_init, tname) and hasattr(_init, jname):
            setattr(_init, tname, getattr(_init, jname))
    # initializers torch has that jittor lacks -- best-effort implementations
    if not hasattr(_init, "_calculate_fan_in_and_fan_out"):
        _init._calculate_fan_in_and_fan_out = _fan
    if not hasattr(_init, "_calculate_correct_fan"):
        _init._calculate_correct_fan = _calculate_correct_fan
    if not hasattr(_init, "dirac_"):
        _init.dirac_ = _dirac
    if not hasattr(_init, "orthogonal_"):
        _init.orthogonal_ = _orth
    if not hasattr(_init, "sparse_"):
        _init.sparse_ = _sparse

    # torch.nn.init also exposes deprecated non-underscore spellings of the
    # in-place initializers (normal/xavier_normal/kaiming_uniform/kaiming_normal),
    # which forward to the `_` versions. Some older model code calls them. Add
    # each alias only when its `_` target exists and the alias is still missing.
    for tname in ("normal", "xavier_normal", "kaiming_uniform", "kaiming_normal"):
        target = tname + "_"
        if not hasattr(_init, tname) and hasattr(_init, target):
            setattr(_init, tname, getattr(_init, target))

    # Keep transformers/diffusers no_init_weights() from replacing jittor's
    # construction-time init functions with no-op stubs. torch.nn is jittor.nn
    # on the bare `import jittor as torch` path, and jittor.nn.Conv/Linear call
    # the same module-global init functions to allocate weights.
    guarded = _GuardedInit("torch.nn.init")
    protected = set()
    for key in dir(_init):
        if not key.startswith("__"):
            try:
                value = getattr(_init, key)
                object.__setattr__(guarded, key, value)
                if callable(value):
                    protected.add(key)
            except EXPECTED as exc:
                swallowed("torch/installers/nn_init.py _install_init_aliases: value = getattr(_init, key)", exc)
    object.__setattr__(guarded, "_protected", protected)
    _target_nn.init = guarded
    _modules["torch.nn.init"] = guarded
    register_api_bindings(
        guarded, "torch.nn.init",
        ("normal_", "uniform_", "zeros_", "ones_", "constant_", "trunc_normal_",
         "kaiming_uniform_", "kaiming_normal_", "gauss_", "xavier_uniform_",
         "xavier_gauss_", "xavier_normal_", "relu_invariant_gauss_",
         "invariant_uniform_", "_calculate_fan_in_and_fan_out",
         "_calculate_correct_fan", "dirac_", "orthogonal_", "sparse_"),
        Fidelity.APPROXIMATE,
        "Native initialization with retained trainability; generator arguments "
        "are not fully honored, truncated normal clamps samples, and some "
        "initializers use NumPy host sampling",
    )
