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

from ..types import _dtype_to_str
from ..nested import _torch_register_leaf
from ..fidelity import Fidelity, register_fidelity
from ...diagnostics import EXPECTED, swallowed
from ...transaction import set_flag


def _set_use_cuda():
    """Turn CUDA on for a torch-requested device, reversibly during install."""
    set_flag(jt.flags, "use_cuda", 1)


#: Every factory published through `_invoke_factory`, which is what carries a
#: `device=` into the native placement scope. A name that `_wrap_constructors`
#: adapts but that is missing here reaches `_constructor_adapter` directly,
#: and that function *drops* `device` (it is in `_DROP`) on the assumption
#: that the placement is already established -- so the tensor would be built
#: on the ambient device with no error.
#:
#: `eye` is deliberately **not** here: its owner is
#: `installers.numerical.eye`, which `_bind_missing` publishes as the module
#: level `torch.eye` and which `test_torch_numerical_fidelity` pins by
#: identity. Routing it through this module would rebind `torch.eye` to a
#: wrapper and break that ownership, so `device=` is honoured in the owner
#: instead -- it enters its own `tensor_frontend(..., device=device)`.
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
    if kwargs.get("layout") is not None and kwargs["layout"] is context.target_namespace.sparse_coo:
        raise NotImplementedError("factory does not support sparse COO layout; use Tensor.to_sparse")
    implementation = context.state.get("factory_implementations", {}).get(name)
    if implementation is None:
        raise RuntimeError("torch.%s is not installed" % name)
    from ..frontend import tensor_frontend
    like = args[0] if args and (name.endswith("_like") or name in _TENSOR_FIRST_ARGUMENT) else None
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
#: Factories whose first positional argument is the tensor the result should
#: follow, so it is the placement reference when no `device=` is given. The
#: samplers belong here as much as `tril`/`triu` do: `torch.multinomial` builds
#: its own working buffers, and with no reference they landed on the ambient
#: device -- `torch.multinomial(weights_on_cuda1, 2)` died with "Expected all
#: tensor inputs on the same backend and device" instead of sampling.
_TENSOR_FIRST_ARGUMENT = _TENSOR_ARGUMENT + ("multinomial", "bernoulli")


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
    # ACL adapters call jt.empty thousands of times; keep the FP32 fast path.
    if (name == "empty" and not kwargs and args and
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
    out.requires_grad_(_requires_grad)
    if _requires_grad:
        _torch_register_leaf(out)
    return out


def _wrap_constructors(g):
    # Keep this a subset of _FACTORY_NAMES, minus `eye` (see the note there):
    # a name here but not there is published without the placement wrapper,
    # and a name there but not here hands torch's `device=`/`requires_grad=`
    # straight to a jittor factory that has no such parameter
    # (`torch.randperm(4, device="cuda:1")` raised "randperm() got an
    # unexpected keyword argument 'device'").
    for name in ("zeros", "ones", "empty", "full", "arange", "rand", "randn",
                 "randint", "randperm", "linspace", "zeros_like",
                 "ones_like", "empty_like", "full_like", "randn_like",
                 "rand_like", "tril", "triu", "normal"):
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


#: What a seeded draw of an integer factory must come back as. torch's own
#: default is int64, but these have to match what the *same call without a
#: generator* returns -- jittor's randint is int32 -- because a generator picks
#: the stream a draw comes from and nothing else.
_INTEGER_DRAWS = {"randperm": "int64", "randint": "int32"}


def _is_dim(value):
    """True for something torch accepts as one dimension of a shape."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value):
    """True for a python/numpy scalar -- not a tensor, which draws elementwise."""
    return (isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, bool))


def _shape_tuple(value):
    """``value`` read as a shape, or None when it is not one.

    torch spells one shape four ways -- ``randn(2, 3)``, ``randn((2, 3))``,
    ``randn(t.shape)`` and ``randn(size=(2, 3))`` -- and under this shim a
    ``torch.Size`` is a jittor ``NanoVector``, which is neither tuple nor list.
    A spelling missed here is not a small loss: the call falls back to the
    *global* stream, which is exactly the rank-dependent draw a passed
    generator says it must not use.
    """
    if isinstance(value, jt.NanoVector):
        return tuple(int(s) for s in value)
    if isinstance(value, (tuple, list)) and all(_is_dim(s) for s in value):
        return tuple(int(s) for s in value)
    return None


def _draw_from_generator(name, generator, args, kwargs):
    """Draw from the generator's own stream, or None if this call is not covered.

    `Generator.manual_seed` builds this stream, so a request's latents are
    reproducible and -- the part that matters for TP -- identical in every rank
    that seeds the same generator, no matter what that rank did before.

    Covering a call is all-or-nothing: what comes back has to be the tensor the
    same call returns *without* a generator -- same shape, same dtype, same
    requires_grad -- because a generator chooses which stream a draw comes from
    and nothing else. Arguments this cannot read that way return None, and the
    original factory runs.
    """
    import numpy as _np
    rng = getattr(generator, "_rng", None)
    if rng is None:
        return None
    src = None
    values_args = ()
    if name.endswith("_like"):
        src = args[0] if args else kwargs.get("input")
        if src is None or not hasattr(src, "shape"):
            return None
        shape = tuple(int(s) for s in src.shape)
    elif name == "randperm":
        n = args[0] if args else kwargs.get("n")
        if not _is_dim(n):
            return None
        shape = (int(n),)
    elif name in ("normal", "randint"):
        # the value arguments come first -- normal(mean, std, size) and
        # randint([low, ] high, size) -- and both also take size= by keyword.
        values_args = list(args)
        size = kwargs.get("size")
        if size is None and values_args:
            size = values_args.pop()
        shape = _shape_tuple(size)
    else:
        shape = None
        if len(args) == 1:
            shape = _shape_tuple(args[0])
        if shape is None and args and all(_is_dim(a) for a in args):
            shape = tuple(int(a) for a in args)
        if shape is None and not args:
            shape = _shape_tuple(kwargs.get("size"))
    if shape is None:
        return None
    if name in ("randn", "randn_like"):
        values = rng.standard_normal(shape)
    elif name in ("rand", "rand_like"):
        values = rng.random(shape)
    elif name == "normal":
        mean = kwargs.get("mean", values_args[0] if len(values_args) > 0 else 0.0)
        std = kwargs.get("std", values_args[1] if len(values_args) > 1 else 1.0)
        if not (_is_number(mean) and _is_number(std)):
            return None     # normal(mean_tensor, std_tensor) draws elementwise
        values = rng.normal(float(mean), float(std), size=shape)
    elif name == "randint":
        if "high" in kwargs:
            low = kwargs.get("low", values_args[0] if values_args else 0)
            high = kwargs["high"]
        elif len(values_args) >= 2:
            low, high = values_args[0], values_args[1]
        elif len(values_args) == 1:
            low, high = kwargs.get("low", 0), values_args[0]
        else:
            return None
        if not (_is_dim(low) and _is_dim(high)):
            return None
        values = rng.integers(int(low), int(high), size=shape)
    elif name == "randperm":
        values = rng.permutation(int(shape[0]))
    else:
        return None
    # dtype: the caller's, else the one the plain call would have produced --
    # the source's for *_like (jittor promotes a non-float source to float32),
    # the integer width for randperm/randint, the default dtype otherwise.
    dtype = kwargs.get("dtype")
    if dtype is not None:
        cast_to = _dtype_to_str(dtype)
    elif name.endswith("_like"):
        like = _dtype_to_str(src.dtype)
        cast_to = like if "float" in str(like) else "float32"
    elif name in _INTEGER_DRAWS:
        cast_to = _INTEGER_DRAWS[name]
    else:
        cast_to = _dtype_to_str(get_install_context(jt).target_namespace.get_default_dtype())
    integral = name in _INTEGER_DRAWS
    t = jt.array(_np.ascontiguousarray(
        values, dtype=_np.int64 if integral else _np.float32))
    if tuple(t.shape) != tuple(shape):
        t = t.reshape(shape)
    t = t.cast(cast_to)
    t._jittor_torch_ext_mutable = True
    requires_grad = bool(kwargs.get("requires_grad", False))
    t.requires_grad_(requires_grad)
    if requires_grad:
        _torch_register_leaf(t)
    return t


def _random_adapter(original, *args, generator=None, _name=None, **kwargs):
    drawn = _draw_from_generator(_name, generator, args, kwargs) if _name else None
    if drawn is not None:
        return drawn
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
            _publish_factory(g, name,
                             functools.partial(_random_adapter, original, _name=name))
