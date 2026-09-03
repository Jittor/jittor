"""Torch-compatible tensor factories.

Jittor's constructors take a shape and a dtype; torch's take a shape and half a
dozen kwargs about where the result should live and whether it records a
gradient. This module wraps the former into the latter, and does the same for
the random samplers, whose torch signatures carry a ``generator=``.

Split out of the tensor installer, which it runs as part of.
"""

import functools

import jittor as jt
import numpy as np

from ..types import (
    _device_is_cpu, _device_is_cuda, _dtype_to_str, _make_cuda_resident,
    _cuda_index_of,
)
from ..nested import _torch_register_leaf
from ..fidelity import Fidelity, register_fidelity
from ...diagnostics import EXPECTED, swallowed


_FACTORY_NAMES = (
    "arange", "bernoulli", "empty", "full", "full_like", "linspace",
    "multinomial", "normal", "ones", "ones_like", "rand", "rand_like",
    "randint", "randn", "randn_like", "randperm", "tril", "triu", "zeros",
    "zeros_like",
)
_FACTORY_FIDELITY_DETAIL = (
    "supports Jittor-backed tensor construction but approximates or omits "
    "some Torch layout, pin-memory, out, or generator-state semantics"
)


class _FactoryAPI:
    """Stable public callable whose internal adapter is bound at install time."""

    def __init__(self, name):
        self.__name__ = name
        self.__qualname__ = name
        self.__module__ = __name__
        self._implementation = None

    @property
    def implementation(self):
        return self._implementation

    def bind(self, implementation):
        if implementation is self:
            return self
        self._implementation = implementation
        self.__wrapped__ = implementation
        if getattr(implementation, "__doc__", None):
            self.__doc__ = implementation.__doc__
        return self

    def __call__(self, *args, **kwargs):
        if self._implementation is None:
            raise RuntimeError("torch.%s is not installed" % self.__name__)
        return self._implementation(*args, **kwargs)


FACTORY_APIS = {name: _FactoryAPI(name) for name in _FACTORY_NAMES}
globals().update(FACTORY_APIS)
for _name, _api in FACTORY_APIS.items():
    register_fidelity(
        "torch." + _name,
        _api,
        Fidelity.APPROXIMATE,
        _FACTORY_FIDELITY_DETAIL,
    )
del _name, _api


def _publish_factory(root, name, implementation):
    api = FACTORY_APIS.get(name)
    if api is None:
        setattr(root, name, implementation)
        return
    api.bind(implementation)
    setattr(root, name, api)


def _factory_implementation(value):
    if isinstance(value, _FactoryAPI):
        return value.implementation
    return value


def _wrap_constructors(g):
    """Wrap jittor tensor constructors to accept torch kwargs (device=,
    requires_grad=, layout=, pin_memory=, out=) and torch dtype objects."""
    import functools, inspect
    _DROP = ("device", "requires_grad", "layout", "pin_memory", "memory_format",
             "out", "non_blocking")
    _DEFAULT_FLOAT_FACTORIES = {
        "zeros", "ones", "empty", "rand", "randn", "eye", "linspace",
    }

    def wrap(name):
        orig = getattr(g, name, None)
        if orig is None:
            return
        if isinstance(orig, _FactoryAPI):
            return
        # Some jittor factories have no dtype param; cast their result instead.
        try:
            _sig = inspect.signature(orig)
            _accepts_dtype = ("dtype" in _sig.parameters or
                              any(p.kind == p.VAR_KEYWORD
                                  for p in _sig.parameters.values()))
        except (ValueError, TypeError):
            _accepts_dtype = True  # builtins w/o introspectable sig: assume ok
        # Wrapped alongside the factories for their kwargs, but their first
        # argument is data rather than a shape.
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
        @functools.wraps(orig)
        def wrapped(*args, **kwargs):
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
            # torch device='cpu' must produce a host-resident Var (native exts
            # check tensor.is_cpu()). Capture the device before dropping it and,
            # when CPU is requested, build the Var under use_cuda=0 so its
            # allocator is the host allocator (Var.location()=='cpu').
            _requested_device = kwargs.get("device")
            _want_cpu = _device_is_cpu(_requested_device)
            _want_cuda = _device_is_cuda(_requested_device)
            _cuda_index = None
            if _want_cuda:
                jt.flags.use_cuda = 1
                # device="cuda:N" means "create it on N", not "create it here
                # and copy it there": the copy would be a wasted transfer and,
                # for a big weight, twice the peak memory.
                _cuda_index = _cuda_index_of(_requested_device)
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
                if default_dtype != "float32":
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
            if _want_cpu and jt.flags.use_cuda:
                # Build on the host so the result is genuinely CPU-resident.
                with jt.flag_scope(use_cuda=0):
                    out = orig(*args, **kwargs)
                    if _cast_to is not None:
                        out = out.cast(_cast_to)
                    out.sync()
                try:
                    out._jittor_torch_ext_mutable = True
                    out._jittor_torch_force_cpu = True
                except (AttributeError, TypeError) as exc:
                    swallowed("torch/installers/factories.py wrapped: out._jittor_torch_ext_mutable = True", exc)
                if _requires_grad:
                    out.requires_grad_(True)
                    _torch_register_leaf(out)
                return out
            if _cuda_index is not None and int(_cuda_index) >= 0:
                with jt.flag_scope(device_id=int(_cuda_index)):
                    out = orig(*args, **kwargs)
            else:
                out = orig(*args, **kwargs)
            if _cast_to is not None:
                out = out.cast(_cast_to)
            if _want_cuda:
                out = _make_cuda_resident(out, force=True)
            try:
                out._jittor_torch_ext_mutable = True
            except (AttributeError, TypeError) as exc:
                swallowed("torch/installers/factories.py wrapped: out._jittor_torch_ext_mutable = True", exc)
            if _requires_grad:
                out.requires_grad_(True)
                _torch_register_leaf(out)
            return out
        wrapped._torch_wrapped = True
        _publish_factory(g, name, wrapped)

    for name in ("zeros", "ones", "empty", "full", "arange", "rand", "randn",
                 "randint", "eye", "linspace", "zeros_like", "ones_like",
                 "empty_like", "full_like", "randn_like", "rand_like", "tril",
                 "triu", "normal"):
        wrap(name)


def _install_random_and_linspace(g):
    """torch-compat for linspace(dtype=) and the random samplers' generator= arg.

    Runs AFTER _wrap_constructors, so it wraps the already-kwarg-tolerant
    versions. jittor's linspace has no `dtype` and its random ops have no
    `generator`, so torch code passing either currently raises TypeError.
    """
    import functools

    # torch.linspace(..., dtype=) -- jittor's linspace has no dtype param. Pop
    # it and cast the result, matching torch (default float32 stays unchanged).
    _lin = _factory_implementation(getattr(g, "linspace", None))
    if _lin is not None:
        @functools.wraps(_lin)
        def linspace(*args, dtype=None, **kwargs):
            # torch.linspace(start, end, steps): mmdet (DETR reference points) passes
            # tensor/float scalars for start/end and a tensor for steps; jittor needs
            # python float start/end and an int steps.
            args = list(args)
            if len(args) >= 1 and hasattr(args[0], "item"): args[0] = float(args[0])
            if len(args) >= 2 and hasattr(args[1], "item"): args[1] = float(args[1])
            if len(args) >= 3 and not isinstance(args[2], int): args[2] = int(args[2])
            if "steps" in kwargs and not isinstance(kwargs["steps"], int):
                kwargs["steps"] = int(kwargs["steps"])
            r = _lin(*args, **kwargs)
            if dtype is not None:
                r = r.cast(_dtype_to_str(dtype))
            return r
        _publish_factory(g, "linspace", linspace)

    # torch.randn/rand/randint(..., generator=) -- jittor samplers seed off the
    # global RNG and reject `generator`. When a Generator is given, seed the
    # global RNG from it (initial_seed()/seed) so the draw is reproducible,
    # then restore nothing (matches torch users who pass a seeded generator for
    # determinism). Without a generator, behavior is unchanged.
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

    def wrap_gen(name):
        orig = _factory_implementation(getattr(g, name, None))
        if orig is None:
            return
        @functools.wraps(orig)
        def wrapped(*args, generator=None, **kwargs):
            _seed_from(generator)
            return orig(*args, **kwargs)
        _publish_factory(g, name, wrapped)

    for name in ("randn", "rand", "randint", "randperm", "normal",
                 "randn_like", "rand_like", "multinomial", "bernoulli"):
        wrap_gen(name)
