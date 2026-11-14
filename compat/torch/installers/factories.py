"""Torch-compatible tensor factories.

Jittor's constructors take a shape and a dtype; torch's take a shape and half a
dozen kwargs about where the result should live and whether it records a
gradient. This module wraps the former into the latter, and does the same for
the random samplers, whose torch signatures carry a ``generator=``.

Split out of the tensor installer, which it runs as part of.
"""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._core import var as _native_creation

import functools
import inspect
from ..context import get_install_context
from ..api_delegates import bind_delegates

import jittor as jt
import numpy as np

from ..types import (
    _device_is_cpu, _device_is_cuda, _dtype_to_str, _make_cpu_resident,
    _make_cuda_resident,
)
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
    "preserves input shape, dtype, device and leaf-gradient options but "
    "approximates or omits Torch layout, pin-memory and memory-format semantics"
)


def _is_meta_device_request(value):
    """Whether a ``device=`` argument names the meta device.

    Jittor has no meta backend; callers that honour the request do so through
    the observable-meta contract (see ``device.__enter__``), not by silently
    picking a real device.
    """
    if value is None:
        return False
    name = getattr(value, "type", None)
    if name is None:
        name = str(value).split(":", 1)[0]
    return name == "meta"


def _invoke_factory(name, args, kwargs):
    context = get_install_context(jt)
    implementation = context.state.get("factory_implementations", {}).get(name)
    if implementation is None:
        raise RuntimeError("torch.%s is not installed" % name)
    from ..frontend import tensor_frontend
    like = args[0] if args and (name.endswith("_like") or name in _TENSOR_FIRST_ARGUMENT) else None
    if name == "normal":
        mean = kwargs.get("mean", args[0] if args else None)
        std = kwargs.get("std", args[1] if len(args) > 1 else None)
        like = mean if isinstance(mean, jt.Var) else std if isinstance(std, jt.Var) else None
    requested = kwargs.get("device")
    # ``torch.arange(tensor_bound)`` inherits the bound tensor's device when no
    # explicit device is supplied. This matters for CUDA scalar bounds used by
    # multimodal position-grid construction. Keep the general factory default
    # for ordinary Python bounds and never let a tensor step decide placement.
    if requested is None and name == "arange":
        for value in args[:2]:
            if isinstance(value, jt.Var):
                requested = "cuda" if value.is_cuda else "cpu"
                break
    meta = _is_meta_device_request(requested)
    # torch's meta tensors carry shape/dtype only; Jittor has no meta backend,
    # so the same contract `with torch.device("meta")` uses applies here: the
    # Var is really allocated (on the host) and reports "meta" via the
    # `_jittor_torch_meta` mark that `Tensor.device`/`is_meta` read.
    with tensor_frontend(context.target_namespace.Var,
                         device="cpu" if meta else requested, like=like):
        result = implementation(*args, **kwargs)
    if meta:
        try:
            result._jittor_torch_meta = True
        except (AttributeError, TypeError) as exc:
            swallowed("torch/installers/factories.py _invoke_factory: meta mark", exc)
    return result


def arange(*args, **kwargs):
    # Torch infers int64 for all-integer bounds and the current default
    # floating dtype for any floating bound. Explicit dtype=None has the same
    # inference semantics as omitting dtype, unlike the size factories.
    if kwargs.get("dtype") is None:
        bounds = tuple(args[:3]) + tuple(kwargs[k] for k in ("start", "end", "step") if k in kwargs)
        def is_floating(value):
            if isinstance(value, jt.Var):
                return _jittor_dtype_name(value.dtype).startswith(("float", "bfloat"))
            return isinstance(value, (float, np.floating))
        kwargs = dict(kwargs)
        kwargs["dtype"] = (get_install_context(jt).target_namespace.get_default_dtype()
                           if any(is_floating(value) for value in bounds) else "int64")
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


def _allocate_empty_like(input):
    return jt.empty(input.shape, dtype=input.dtype)


def _copy_like(input):
    return input.clone()


def _materialize_meta_like(input, device, **kwargs):
    """Copy an emulated-meta buffer's values onto the device the call asked for.

    The copy has to be made explicitly: ``input.clone()`` keeps the input's
    device, so a buffer built under the ``torch.device("meta")`` emulation (real
    host storage) stayed on CPU even when the factory was called with
    ``device="cuda"``. transformers materializes Qwen2's non-persistent rotary
    buffers this way while loading with ``device_map="cuda"``; the parameters
    moved to CUDA, those buffers did not, and DDP's NCCL broadcast then had no
    CPU kernel. A fresh allocation under the factory's ambient placement would
    follow the request, but it holds no values, so copy first and then move.
    """
    copied = _constructor_adapter("empty_like", _copy_like, False, input, **kwargs)
    if _device_is_cpu(device):
        return _make_cpu_resident(copied)
    if _device_is_cuda(device):
        return _make_cuda_resident(copied, force=True, device=device)
    return copied


def _empty_like_implementation(input, **kwargs):
    """Apply the shared factory options to the compiler's empty allocation."""
    if getattr(input, "_jittor_torch_meta_data", False):
        # The shim's meta context allocates real host storage, so a buffer
        # built there already holds the values from __init__.  transformers'
        # _move_missing_keys_from_meta_to_device materializes non-persistent
        # buffers with torch.empty_like; returning uninitialized memory makes
        # it lose those values, and the later _init_weights restore is skipped
        # for modules that are already initialized (CLIP position_ids,
        # SpeechT5 sinusoidal weights).  Materializing the emulated meta
        # storage therefore copies it instead of exposing garbage -- onto the
        # device the factory call requested.
        device = kwargs.pop("device", None)
        return _materialize_meta_like(input, device, **kwargs)
    return _constructor_adapter("empty_like", _allocate_empty_like, False, input, **kwargs)


def _install_empty_like(root):
    """Publish the family-owned stable object at the legacy install step."""
    if not hasattr(root, "empty_like"):
        _publish_factory(root, "empty_like", _empty_like_implementation)


_DROP = ("device", "requires_grad", "layout", "pin_memory", "memory_format", "out", "non_blocking")
_DEFAULT_FLOAT_FACTORIES = {"zeros", "ones", "empty", "rand", "randn", "eye", "linspace"}
_FRESH_LEAF_FACTORIES = frozenset({
    "empty", "zeros", "ones", "full", "arange", "linspace",
    "empty_like", "zeros_like", "ones_like", "full_like",
    "rand", "randn", "randint", "randperm", "rand_like", "randn_like",
})
# Identity proves that these allocate from shape/scalar metadata only. A
# replaced factory may return an existing tensor, which still needs detach.
_FRESH_CONSTANT_LIKE = {
    "zeros_like": _native_creation.zeros_like,
    "ones_like": _native_creation.ones_like,
    "full_like": _native_creation.full_like,
}
_TENSOR_ARGUMENT = ("tril", "triu")
#: Factories whose first positional argument is the tensor the result should
#: follow, so it is the placement reference when no `device=` is given. The
#: samplers belong here as much as `tril`/`triu` do: `torch.multinomial` builds
#: its own working buffers, and with no reference they landed on the ambient
#: device -- `torch.multinomial(weights_on_cuda1, 2)` died with "Expected all
#: tensor inputs on the same backend and device" instead of sampling.
_TENSOR_FIRST_ARGUMENT = _TENSOR_ARGUMENT + ("multinomial", "bernoulli")


def _creates_fresh_leaf(name, args, kwargs):
    if name in _FRESH_LEAF_FACTORIES:
        return True
    if name == "normal":
        # Tensor mean/std have their own sampling/autograd contract. Only the
        # scalar overload creates a leaf with no input tensor history.
        values = args[:2] + (kwargs.get("mean"), kwargs.get("std"))
        return not any(isinstance(value, jt.Var) for value in values)
    return False


def _draw_inputs_require_grad(args, kwargs):
    values = list(args) + [kwargs.get("mean"), kwargs.get("std"),
                           kwargs.get("input")]
    return any(isinstance(value, jt.Var) and bool(value.requires_grad)
               for value in values)


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
    # Native integer arange already constructs a fresh, stopped int64 Var.
    # Placement is established by _invoke_factory; no input normalization,
    # dtype conversion or detach is needed for this common position-ID case.
    # An overridden factory may return an existing tensor, so it must retain
    # the general constructor's conversion and leaf handling below.
    if (name == "arange" and orig is jt.arange and _accepts_dtype and
            1 <= len(args) <= 3 and all(type(value) is int for value in args) and
            kwargs.get("requires_grad", False) is False and
            kwargs.keys() <= {"dtype", "device", "requires_grad"}):
        dtype = kwargs.get("dtype")
        if ((type(dtype) is str and dtype == "int64") or
                dtype is get_install_context(jt).target_namespace.int64):
            out = orig(*args, dtype="int64")
            out._jittor_torch_ext_mutable = True
            out.requires_grad_(False)
            return out
    if (orig is _FRESH_CONSTANT_LIKE.get(name)
            and len(args) == (2 if name == "full_like" else 1)
            and isinstance(args[0], jt.Var)
            and (name != "full_like" or type(args[1]) in (int, float, bool))
            and kwargs.get("dtype") is None and kwargs.get("device") is None
            and kwargs.get("requires_grad", False) is False
            and kwargs.keys() <= {"dtype", "device", "requires_grad"}):
        # Placement is already scoped; native like factories inherit dtype.
        # Stop the fresh result's internal constant/broadcast history without
        # creating a second holder through detach or changing the input leaf.
        out = orig(*args).stop_grad()
        out._jittor_torch_ext_mutable = True
        return out
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
            out.requires_grad_(False)
            return out
    # _invoke_factory already established native construction placement.
    _requires_grad = bool(kwargs.get("requires_grad", False))
    fresh_like = name in ("empty_like", "zeros_like", "ones_like", "full_like")
    fresh_leaf = _creates_fresh_leaf(name, args, kwargs)
    input_requires_grad = not fresh_leaf and any(
        isinstance(value, jt.Var) and value.requires_grad
        for value in tuple(args) + tuple(kwargs.values()))
    for k in _DROP:
        kwargs.pop(k, None)
    # Jittor shape conversion rejects numpy scalars; normalize them.
    # Only for the factories that really take a shape: tril/triu take
    # the matrix to transform, and a 1x1 matrix holds a single element,
    # so shape conversion would collapse it into an integer dimension.
    _takes_shape = not (name.endswith("_like") or name in _TENSOR_ARGUMENT)
    if args and name == "arange":
        # arange takes bounds and a step, not dimensions: a 0-D float tensor
        # step (`1 / count`) must stay fractional, which `_shape_arg` would
        # truncate to 0. Only numpy scalars need unwrapping; Vars are read by
        # the native arange itself.
        args = tuple(a.item() if isinstance(a, np.generic) else a for a in args)
    elif args and _takes_shape:
        args = tuple(_shape_arg(a) for a in args)
    # Jittor factories reject Size/NanoVector tuple subclasses.
    if _takes_shape and args and (isinstance(args[0], jt.NanoVector) or
                 (isinstance(args[0], tuple) and type(args[0]) is not tuple)):
        args = (tuple(int(x) for x in args[0]),) + tuple(args[1:])
    # Torch also allows shape via size=.
    if name == "normal" and "size" in kwargs and not args:
        size = _shape_arg(kwargs.pop("size"))
        mean = kwargs.pop("mean", 0.0)
        std = kwargs.pop("std", 1.0)
        args = (mean, std, size)
    elif "size" in kwargs and not args:
        sz = kwargs.pop("size")
        # Route the keyword spelling through the same scalar-dimension
        # normalization as the positional spelling. Multimodal audio encoders
        # commonly compute a padded length as a CUDA 0-D tensor and pass it
        # through ``torch.full(size=(..., length))``.
        args = (_shape_arg(sz),)
    # torch.full(size, fill_value=...) / full_like(input, fill_value=...):
    # jittor's full(shape, val) / full_like(x, val) take the value as the 2nd
    # positional. transformers' beam scorer passes fill_value= as a keyword, so
    # map it onto the next positional slot (only full/full_like ever get it).
    if "fill_value" in kwargs:
        args = tuple(args) + (kwargs.pop("fill_value"),)
    _cast_to = None  # cast after construction when needed for torch dtype semantics
    # PyTorch's integer sampler defaults to long, while jt.randint defaults
    # to int32. Distributed peers may allocate torch.long receive buffers.
    if name == "randint" and kwargs.get("dtype") is None:
        kwargs["dtype"] = "int64"
    if "dtype" not in kwargs and name in _DEFAULT_FLOAT_FACTORIES:
        default_dtype = _dtype_to_str(g.get_default_dtype())
        if _jittor_dtype_name(default_dtype) != "float32":
            if _accepts_dtype:
                kwargs["dtype"] = default_dtype
            else:
                _cast_to = default_dtype
    if fresh_like and kwargs.get("dtype", False) is None:
        # Unlike size factories, explicit None follows the input dtype.
        kwargs.pop("dtype")
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
    # rand_like/randn_like inherit the input dtype when dtype= is omitted, so
    # the complex check must see that inherited dtype, not just an explicit one.
    _draw_dtype = _cast_to
    if (_draw_dtype is None and name.endswith("_like") and args
            and isinstance(args[0], jt.Var)):
        _inherited = _jittor_dtype_name(args[0].dtype)
        if _inherited in _COMPLEX_COMPONENTS:
            _draw_dtype = _inherited
    if _draw_dtype in _COMPLEX_COMPONENTS and name in _COMPLEX_RANDOM_FACTORIES:
        shape = _complex_random_shape(name, args)
        if shape is None:
            raise NotImplementedError(
                "torch.%s on a complex dtype needs a resolvable shape for the "
                "two-draw composition" % name)
        out = _complex_random(name, shape, _draw_dtype)
    elif _cast_to in _COMPLEX_COMPONENTS and name == "normal":
        _complex_normal_refusal()
    else:
        out = orig(*args, **kwargs)
    # Typed factories can already return the requested dtype. A final cast
    # should convert values, rather than repeat frontend dispatch unchanged.
    if (_cast_to is not None and
            _jittor_dtype_name(out.dtype) != _jittor_dtype_name(_cast_to)):
        out = out.cast(_cast_to)
    if fresh_leaf or not (_requires_grad or input_requires_grad):
        # Internal allocation/cast nodes do not create public history for a
        # fresh factory, or for an operation whose tensor inputs are stopped.
        out = out.detach()
    out._jittor_torch_ext_mutable = True
    if fresh_leaf or (_requires_grad and not out.requires_grad):
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
    if dtype is not None:
        # The constructor casts before detaching/registering the fresh leaf.
        # A cast here creates another holder after that registration.
        kwargs["dtype"] = dtype
    return original(*args, **kwargs)


#: A generator changes the random stream, not the public default dtype.
_INTEGER_DRAWS = {"randperm": "int64", "randint": "int64"}

#: Native complex dtypes and the real dtype each part is drawn from.
_COMPLEX_COMPONENTS = {"complex64": "float32", "complex128": "float64"}

#: Random factories defined over a complex dtype. torch draws the real
#: and imaginary parts independently over the same shape and distribution.
_COMPLEX_RANDOM_FACTORIES = frozenset({
    "rand", "randn", "rand_like", "randn_like",
})


def _complex_random_shape(name, args):
    """The shape a complex random factory would draw (None if unresolvable)."""
    if name.endswith("_like"):
        return (tuple(int(s) for s in args[0].shape) if args else None)
    if len(args) == 1:
        resolved = _shape_tuple(args[0])
        if resolved is not None:
            return resolved
    if args and all(_is_dim(a) for a in args):
        return tuple(int(a) for a in args)
    return None


def _complex_random(name, shape, dtype, generator=None):
    """Draw a complex tensor from two independent real native draws.

    Jittor instantiates its random kernels over real scalar types, so a
    complex dtype request never compiles. torch defines rand/randn on a
    complex dtype as two independent draws over the same shape, one for
    the real part and one for the imaginary part; composing them here
    keeps the draw on the native real kernels instead of narrowing it to
    a dtype the core does not compute.
    """
    component = _COMPLEX_COMPONENTS[dtype]
    distribution = "normal" if name.startswith("randn") else "uniform"
    real = jt.random(list(shape), component, distribution, generator=generator)
    imag = jt.random(list(shape), component, distribution, generator=generator)
    return jt.nn.view_as_complex(jt.stack([real, imag], dim=-1))


def _complex_normal_refusal():
    raise NotImplementedError(
        "torch.normal does not sample complex dtypes: the core has no "
        "complex normal kernel; use torch.randn for a complex normal draw")


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


def _native_generator(generator):
    """Validate placement before a native stream reserves or draws anything."""
    if generator is None:
        return None
    context = get_install_context(jt)
    default = getattr(context.target_namespace, "default_generator", None)
    native = getattr(generator, "_native", None)
    if generator is not default and not isinstance(native, jt.Generator):
        raise TypeError("generator must be a torch.Generator")
    placement = jt.core._current_tensor_placement()
    backend, ordinal = placement if placement is not None else (0, 0)
    selected = str(getattr(native, "device", "cpu"))
    kind = selected.split(":", 1)[0]
    expected = "cpu" if backend == 0 else "cuda" if backend == 1 else "unsupported"
    if kind != expected:
        raise RuntimeError("Expected a '%s' device type for generator but found '%s'" % (expected, kind))
    if kind == "cuda":
        index = int(selected.split(":", 1)[1])
        if index != ordinal:
            raise RuntimeError("Generator and output must use the same CUDA device")
    return native


def _draw_from_generator(name, generator, args, kwargs):
    """Adapt signatures to existing native samplers, retaining their PRNG owner."""
    native = _native_generator(generator)
    if native is None:
        return None
    if kwargs.get("out") is not None:
        raise NotImplementedError("explicit-generator factories do not support out yet")
    src = None
    values_args = ()
    if name.endswith("_like"):
        src = args[0] if args else kwargs.get("input")
        shape = tuple(int(s) for s in src.shape)
    elif name in ("multinomial", "bernoulli"):
        src = args[0] if args else kwargs.get("input")
        shape = tuple(int(s) for s in src.shape)
    elif name == "randperm":
        n = args[0] if args else kwargs.get("n")
        if not _is_dim(n):
            raise TypeError("randperm expects an integer n")
        shape = (int(n),)
    elif name in ("normal", "randint"):
        values_args = list(args)
        size = kwargs.get("size")
        if name == "normal":
            mean = kwargs.get("mean", values_args[0] if values_args else 0.0)
            std = kwargs.get("std", values_args[1] if len(values_args) > 1 else 1.0)
            src = mean if isinstance(mean, jt.Var) else std if isinstance(std, jt.Var) else None
            if size is None and len(values_args) > 2:
                size = values_args[2]
            shape = tuple(int(s) for s in src.shape) if src is not None and size is None else _shape_tuple(size)
        else:
            if size is None and values_args:
                size = values_args.pop()
            shape = _shape_tuple(size)
    else:
        shape = _shape_tuple(args[0]) if len(args) == 1 else None
        if shape is None and args and all(_is_dim(a) for a in args):
            shape = tuple(int(a) for a in args)
        if shape is None and not args:
            shape = _shape_tuple(kwargs.get("size"))
    if shape is None:
        raise TypeError("%s requires a valid size for an explicit generator" % name)
    dtype = kwargs.get("dtype")
    if dtype is not None:
        cast_to = _dtype_to_str(dtype)
    elif src is not None:
        cast_to = _dtype_to_str(src.dtype)
    elif name in _INTEGER_DRAWS:
        cast_to = _INTEGER_DRAWS[name]
    else:
        cast_to = _dtype_to_str(get_install_context(jt).target_namespace.get_default_dtype())
    if name in ("randn", "randn_like", "rand", "rand_like"):
        if cast_to in _COMPLEX_COMPONENTS:
            t = _complex_random(name, shape, cast_to, generator=native)
        else:
            distribution = "normal" if name.startswith("randn") else "uniform"
            t = jt.random(shape, cast_to, distribution, generator=native)
    elif name == "normal":
        if cast_to in _COMPLEX_COMPONENTS:
            _complex_normal_refusal()
        t = jt.normal(mean, std, size=shape, dtype=cast_to, generator=native)
    elif name == "randint":
        if "high" in kwargs:
            low = kwargs.get("low", values_args[0] if values_args else 0)
            high = kwargs["high"]
        elif len(values_args) >= 2:
            low, high = values_args[0], values_args[1]
        elif len(values_args) == 1:
            low, high = kwargs.get("low", 0), values_args[0]
        else:
            raise TypeError("randint requires high")
        t = jt.randint(low, high, shape, dtype=cast_to, generator=native)
    elif name == "randperm":
        t = jt.randperm(shape[0], dtype=cast_to, generator=native)
    elif name == "multinomial":
        count = args[1] if len(args) > 1 else kwargs.get("num_samples")
        replacement = args[2] if len(args) > 2 else kwargs.get("replacement", False)
        t = jt.multinomial(src, count, replacement, generator=native)
    elif name == "bernoulli":
        t = jt.bernoulli(src, generator=native)
    else:
        raise NotImplementedError("%s does not support an explicit native generator" % name)
    if _creates_fresh_leaf(name, args, kwargs):
        t = t.detach()
    t._jittor_torch_ext_mutable = True
    requires_grad = bool(kwargs.get("requires_grad", False))
    if requires_grad:
        if not bool(t.requires_grad):
            t.requires_grad_(True)
        _torch_register_leaf(t)
    elif bool(t.requires_grad) and not _draw_inputs_require_grad(args, kwargs):
        # The native draw flags its result even when no input asks for a
        # gradient; torch's factory result is a leaf that does not require
        # one. A non-leaf refuses the flag change, so cut it outright.
        if t.is_backward_leaf:
            t.requires_grad_(False)
        else:
            t.stop_grad()
    return t


def _bernoulli_adapter(original, *args, generator=None, **kwargs):
    """torch.bernoulli(input, p=...): sample with one constant probability.

    jittor's bernoulli draws each element with the probability stored in that
    element and takes no ``p``; transformers' SpeechT5 decoder calls the scalar
    torch form ``torch.bernoulli(inputs_embeds[0], p=p)``.
    """
    p = kwargs.pop("p", None)
    if p is not None:
        reference = args[0] if args else kwargs.pop("input", None)
        if reference is None:
            raise TypeError("bernoulli() missing required argument: 'input'")
        probs = jt.full_like(reference, float(p))
        return jt.bernoulli(probs) if generator is None else jt.bernoulli(probs, generator=generator)
    if generator is None:
        return original(*args, **kwargs)
    drawn = _draw_from_generator("bernoulli", generator, args, kwargs)
    return original(*args, **kwargs) if drawn is None else drawn


def _randint_adapter(original, *args, generator=None, **kwargs):
    """Accept torch.randint's (low[, high], size) spellings.

    torch accepts ``randint(high, size)``, ``randint(low, high, size)`` and the
    keyword spellings ``high=``/``low=``/``size=`` with size positional or a
    keyword; jittor's sampler is ``randint(low, high, shape)``.  The generic
    constructor adapter routes a keyword ``size`` into the first positional
    slot, which jittor reads as ``low`` and fails with ``high - low`` TypeError.
    torchmetrics' classification fixtures and transformers' DETR determinism
    test use exactly that keyword form.
    """
    size = kwargs.pop("size", None)
    low = kwargs.pop("low", None)
    high = kwargs.pop("high", None)
    positional = list(args)
    if size is None and len(positional) >= 2:
        size = positional.pop()
    if high is None:
        if len(positional) == 1:
            high = positional.pop()
        elif len(positional) == 2:
            low, high = positional[0], positional[1]
            positional.clear()
        elif len(positional) > 2:
            raise TypeError("randint() takes at most 3 positional arguments")
    elif positional:
        if len(positional) == 1 and low is None:
            low = positional.pop()
        else:
            raise TypeError("randint() got multiple values for an argument")
    if high is None:
        raise TypeError("randint() missing required argument 'high'")
    if size is None:
        raise TypeError("randint() missing required argument 'size'")
    if low is None:
        low = 0

    def _scalar(value):
        return value.item() if isinstance(value, jt.Var) else int(value)

    shape = tuple(int(dim) for dim in size) if isinstance(size, (tuple, list)) else (int(size),)
    return _random_adapter(original, _scalar(low), _scalar(high), shape,
                           generator=generator, _name="randint", **kwargs)


def _random_adapter(original, *args, generator=None, _name=None, **kwargs):
    if _name == "bernoulli":
        return _bernoulli_adapter(original, *args, generator=generator, **kwargs)
    if generator is None:
        return original(*args, **kwargs)
    drawn = _draw_from_generator(_name, generator, args, kwargs)
    return original(*args, **kwargs) if drawn is None else drawn


def _install_random_and_linspace(g):
    original = _factory_implementation(getattr(g, "linspace", None))
    if original is not None:
        _publish_factory(g, "linspace", functools.partial(_linspace_adapter, original))
    for name in ("randn", "rand", "randint", "randperm", "normal",
                 "randn_like", "rand_like", "multinomial", "bernoulli"):
        original = _factory_implementation(getattr(g, name, None))
        if original is None:
            continue
        if name == "randint":
            # torch's size=/high= spellings need their own normalization before
            # the constructor adapter sees a shape in the low slot.
            _publish_factory(g, name, functools.partial(_randint_adapter, original))
        else:
            _publish_factory(g, name,
                             functools.partial(_random_adapter, original, _name=name))
