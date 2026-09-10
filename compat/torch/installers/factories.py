"""Torch-compatible tensor factories.

Jittor's constructors take a shape and a dtype; torch's take a shape and half a
dozen kwargs about where the result should live and whether it records a
gradient. This module wraps the former into the latter, and does the same for
the random samplers, whose torch signatures carry a ``generator=``.

Split out of the tensor installer, which it runs as part of.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import functools
import inspect
from ..context import get_install_context
from ..api_delegates import bind_delegates

import jittor as jt
import numpy as np

from ..types import (
    _DEVICE_CTX_STACK,
    _device_is_meta,
    _dtype_to_str,
    _set_meta_placeholder,
)
from ..nested import _torch_register_leaf
from ..fidelity import Fidelity, register_fidelity
from ...diagnostics import EXPECTED, swallowed
from ...transaction import set_flag


def _set_use_cuda():
    """Turn CUDA on for a torch-requested device, reversibly during install."""
    set_flag(jt.flags, "use_cuda", 1)


_FACTORY_NAMES = (
    "arange", "bernoulli", "empty", "empty_like", "full", "full_like",
    "linspace", "multinomial", "normal", "ones", "ones_like", "rand",
    "rand_like", "randint", "randn", "randn_like", "randperm", "tril",
    "triu", "zeros", "zeros_like",
)
_FACTORY_FIDELITY_DETAIL = (
    "supports Jittor-backed tensor construction but approximates or omits "
    "some Torch layout, pin-memory, out, or generator-state semantics"
)
_EMPTY_LIKE_FIDELITY_DETAIL = (
    "preserves the input shape and dtype through Jittor allocation but omits "
    "Torch layout, device, requires_grad, pin-memory, and memory-format semantics"
)


def _invoke_factory(name, args, kwargs):
    context = get_install_context(jt)
    implementation = context.state.get("factory_implementations", {}).get(name)
    if implementation is None:
        raise RuntimeError("torch.%s is not installed" % name)
    from ..frontend import tensor_frontend
    like = args[0] if args and (name.endswith("_like") or name in _TENSOR_ARGUMENT) else None
    with tensor_frontend(context.target_namespace.Var, device=kwargs.get("device"), like=like):
        return implementation(*args, **kwargs)


def arange(*args, **kwargs):
    return _invoke_factory("arange", args, kwargs)


def bernoulli(*args, **kwargs):
    return _invoke_factory("bernoulli", args, kwargs)


def empty(*args, **kwargs):
    return _invoke_factory("empty", args, kwargs)


def empty_like(*args, **kwargs):
    return _invoke_factory("empty_like", args, kwargs)


def full(*args, **kwargs):
    return _invoke_factory("full", args, kwargs)


def full_like(*args, **kwargs):
    return _invoke_factory("full_like", args, kwargs)


def linspace(*args, **kwargs):
    return _invoke_factory("linspace", args, kwargs)


def multinomial(*args, **kwargs):
    return _invoke_factory("multinomial", args, kwargs)


def normal(*args, **kwargs):
    return _invoke_factory("normal", args, kwargs)


def ones(*args, **kwargs):
    return _invoke_factory("ones", args, kwargs)


def ones_like(*args, **kwargs):
    return _invoke_factory("ones_like", args, kwargs)


def rand(*args, **kwargs):
    return _invoke_factory("rand", args, kwargs)


def rand_like(*args, **kwargs):
    return _invoke_factory("rand_like", args, kwargs)


def randint(*args, **kwargs):
    return _invoke_factory("randint", args, kwargs)


def randn(*args, **kwargs):
    return _invoke_factory("randn", args, kwargs)


def randn_like(*args, **kwargs):
    return _invoke_factory("randn_like", args, kwargs)


def randperm(*args, **kwargs):
    return _invoke_factory("randperm", args, kwargs)


def tril(*args, **kwargs):
    return _invoke_factory("tril", args, kwargs)


def triu(*args, **kwargs):
    return _invoke_factory("triu", args, kwargs)


def zeros(*args, **kwargs):
    return _invoke_factory("zeros", args, kwargs)


def zeros_like(*args, **kwargs):
    return _invoke_factory("zeros_like", args, kwargs)


FACTORY_APIS = {name: globals()[name] for name in _FACTORY_NAMES}
for _name, _api in FACTORY_APIS.items():
    register_fidelity(
        "torch." + _name,
        _api,
        Fidelity.APPROXIMATE,
        (_EMPTY_LIKE_FIDELITY_DETAIL
         if _name == "empty_like" else _FACTORY_FIDELITY_DETAIL),
    )
del _name, _api


def _publish_factory(root, name, implementation):
    context = get_install_context(root)
    delegates = dict(context.state.get("factory_implementations", {}))
    delegates[name] = implementation
    bind_delegates(context, "factory_implementations", delegates)
    setattr(root, name, FACTORY_APIS.get(name, implementation))


def _factory_implementation(value):
    name = getattr(value, "__name__", None)
    if name in FACTORY_APIS and value is FACTORY_APIS[name]:
        return get_install_context(jt).state["factory_implementations"][name]
    return value


def _empty_like_implementation(input, **kwargs):
    """Preserve the historical compiler fallback for ``torch.empty_like``."""
    selected = kwargs.get("dtype")
    return jt.empty(input.shape, dtype=input.dtype if selected is None else _dtype_to_str(selected))


def _install_empty_like(root):
    """Publish the family-owned stable object at the legacy install step."""
    if not hasattr(root, "empty_like"):
        _publish_factory(root, "empty_like", _empty_like_implementation)


_DROP = ("device", "requires_grad", "layout", "pin_memory", "memory_format", "out", "non_blocking")
_DEFAULT_FLOAT_FACTORIES = {"zeros", "ones", "empty", "rand", "randn", "eye", "linspace"}
_TENSOR_ARGUMENT = ("tril", "triu")


def _shape_dim(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, jt.Var):
        try:
            if int(np.prod(tuple(v.shape))) == 1:
                return int(v.item())
        except EXPECTED as exc:
            swallowed("torch/installers/factories.py _shape_dim: if int(np.prod(tuple(v.shape))) == 1:", exc)
    return v
def _shape_arg(v):
    if isinstance(v, jt.NanoVector):
        return tuple(int(x) for x in v)
    if isinstance(v, tuple):
        return tuple(_shape_dim(x) for x in v)
    if isinstance(v, list):
        return tuple(_shape_dim(x) for x in v)
    return _shape_dim(v)


def _constructor_adapter(name, orig, _accepts_dtype, *args, **kwargs):
    g = get_install_context(jt).target_namespace
    requested_device = kwargs.get("device")
    inherits_device = name.endswith("_like") or name in _TENSOR_ARGUMENT
    device_input = args[0] if inherits_device and args and isinstance(args[0], jt.Var) else None
    want_meta = (
        _device_is_meta(requested_device)
        or (requested_device is None and device_input is not None
            and getattr(device_input, "_jittor_torch_meta", False))
        or (requested_device is None and device_input is None
            and bool(_DEVICE_CTX_STACK))
    )
    # ACL adapters call jt.empty thousands of times; keep the FP32 fast path.
    if (name == "empty" and not want_meta and not kwargs and args and
            g.get_default_dtype() == g.float32 and
            (len(args) == 1 or all(type(dim) is int for dim in args))):
        shape = args[0]
        native_shape = isinstance(shape, jt.NanoVector) or type(shape) is int
        if type(shape) in (tuple, list):
            native_shape = all(type(dim) is int for dim in shape)
        if native_shape:
            out = orig(*args)
            out._jittor_torch_ext_mutable = True
            return out
    # _invoke_factory already established native construction placement.
    _requires_grad = bool(kwargs.get("requires_grad", False))
    for k in _DROP:
        kwargs.pop(k, None)
    # Jittor shape conversion rejects numpy scalars; normalize them.
    # Only for the factories that really take a shape: tril/triu take
    # the matrix to transform, and a 1x1 matrix holds a single element,
    # so shape conversion would collapse it into an integer dimension.
    _takes_shape = not (name.endswith("_like") or name in _TENSOR_ARGUMENT)
    if args and _takes_shape:
        args = tuple(_shape_arg(a) for a in args)
    # Jittor factories reject Size/NanoVector tuple subclasses.
    if _takes_shape and args and (isinstance(args[0], jt.NanoVector) or
                 (isinstance(args[0], tuple) and type(args[0]) is not tuple)):
        args = (tuple(int(x) for x in args[0]),) + tuple(args[1:])
    # Torch also allows shape via size=.
    if "size" in kwargs and not args:
        sz = kwargs.pop("size")
        args = (tuple(sz),) if hasattr(sz, "__len__") else (sz,)
    # torch.full(size, fill_value=...) / full_like(input, fill_value=...):
    # jittor's full(shape, val) / full_like(x, val) take the value as the 2nd
    # positional. transformers' beam scorer passes fill_value= as a keyword, so
    # map it onto the next positional slot (only full/full_like ever get it).
    if "fill_value" in kwargs:
        args = tuple(args) + (kwargs.pop("fill_value"),)
    _cast_to = None  # cast after construction when needed for torch dtype semantics
    if "dtype" not in kwargs and name in _DEFAULT_FLOAT_FACTORIES:
        default_dtype = _dtype_to_str(g.get_default_dtype())
        if _jittor_dtype_name(default_dtype) != "float32":
            if _accepts_dtype:
                kwargs["dtype"] = default_dtype
            else:
                _cast_to = default_dtype
    if "dtype" in kwargs:
        if kwargs["dtype"] is None:
            # torch.empty/zeros(..., dtype=None) -> the default dtype.
            # jittor's factories reject dtype=None, so resolve it.
            if _accepts_dtype:
                try:
                    kwargs["dtype"] = _dtype_to_str(g.get_default_dtype())
                except EXPECTED as exc:
                    swallowed("torch/installers/factories.py wrapped: kwargs['dtype'] = _dtype_to_str(g.get_default_dtype())", exc)
                    kwargs.pop("dtype")
            else:
                kwargs.pop("dtype")
        elif _accepts_dtype:
            kwargs["dtype"] = _dtype_to_str(kwargs["dtype"])
            _cast_to = kwargs["dtype"]
        else:
            # ones_like / tril / triu have no dtype param in jittor; torch
            # accepts one. Pop it and cast the result instead.
            _cast_to = _dtype_to_str(kwargs.pop("dtype"))
    out = orig(*args, **kwargs)
    if _cast_to is not None:
        out = out.cast(_cast_to)
    out._jittor_torch_ext_mutable = True
    if want_meta:
        _set_meta_placeholder(out)
    out.requires_grad_(_requires_grad)
    if _requires_grad:
        _torch_register_leaf(out)
    return out


def _wrap_constructors(g):
    for name in ("zeros", "ones", "empty", "full", "arange", "rand", "randn",
                 "randint", "eye", "linspace", "zeros_like", "ones_like",
                 "empty_like", "full_like", "randn_like", "rand_like", "tril",
                 "triu", "normal"):
        original = getattr(g, name, None)
        if original is None or original is FACTORY_APIS.get(name):
            continue
        try:
            signature = inspect.signature(original)
            accepts_dtype = ("dtype" in signature.parameters or any(
                p.kind == p.VAR_KEYWORD for p in signature.parameters.values()))
        except (ValueError, TypeError):
            accepts_dtype = True
        _publish_factory(g, name, functools.partial(
            _constructor_adapter, name, original, accepts_dtype))


def _linspace_adapter(original, *args, dtype=None, **kwargs):
    # torch.linspace(start, end, steps): mmdet (DETR reference points) passes
    # tensor/float scalars for start/end and a tensor for steps; jittor needs
    # python float start/end and an int steps.
    args = list(args)
    if len(args) >= 1 and hasattr(args[0], "item"): args[0] = float(args[0])
    if len(args) >= 2 and hasattr(args[1], "item"): args[1] = float(args[1])
    if len(args) >= 3 and not isinstance(args[2], int): args[2] = int(args[2])
    if "steps" in kwargs and not isinstance(kwargs["steps"], int):
        kwargs["steps"] = int(kwargs["steps"])
    r = original(*args, **kwargs)
    if dtype is not None:
        r = r.cast(_dtype_to_str(dtype))
    return r


def _seed_from(gen):
    if gen is None:
        return
    s = None
    for attr in ("initial_seed", "seed"):
        fn = getattr(gen, attr, None)
        if callable(fn):
            try:
                s = fn()
                break
            except EXPECTED as exc:
                swallowed("torch/installers/factories.py _seed_from: s = fn()", exc)
                s = None
    if s is None:
        s = getattr(gen, "_seed", None)
    if s is not None and hasattr(jt, "set_global_seed"):
        jt.set_global_seed(int(s))


def _random_adapter(original, *args, generator=None, **kwargs):
    _seed_from(generator)
    return original(*args, **kwargs)


def _install_random_and_linspace(g):
    original = _factory_implementation(getattr(g, "linspace", None))
    if original is not None:
        _publish_factory(g, "linspace", functools.partial(_linspace_adapter, original))
    for name in ("randn", "rand", "randint", "randperm", "normal",
                 "randn_like", "rand_like", "multinomial", "bernoulli"):
        original = _factory_implementation(getattr(g, name, None))
        if original is not None:
            _publish_factory(g, name, functools.partial(_random_adapter, original))
