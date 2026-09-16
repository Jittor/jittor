"""Gradient mode, clipping, autocast, and loss-scaling compatibility."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import numpy as np
from typing import Any, Dict, List, cast
import jittor as jt
from .. import collectives as _collectives


import functools as _functools


class _GradDecoratorCtx:
    """Mimics torch.no_grad/enable_grad: usable as a context manager, a bare
    decorator (@torch.no_grad), and a called decorator (@torch.no_grad())."""

    def __init__(self, scope_factory, func=None):
        self._scope_factory = scope_factory
        self._func = func if callable(func) else None

    def __call__(self, *args, **kwargs):
        # used as @torch.no_grad() returning a decorator, then applied to a func
        if self._func is None and len(args) == 1 and callable(args[0]) and not kwargs:
            func = args[0]
            @_functools.wraps(func)
            def wrapped(*a, **k):
                with self._scope_factory():
                    return func(*a, **k)
            return wrapped
        # used as @torch.no_grad (bare): self._func was set at construction
        if self._func is not None:
            with self._scope_factory():
                return self._func(*args, **kwargs)
        raise TypeError("no_grad/enable_grad misuse")

    def __get__(self, obj, objtype=None):
        # Descriptor protocol: when @torch.no_grad wraps a *method*, this instance
        # replaces the method in the class dict. Without __get__, `inst.method`
        # returns this object unbound, so `self` is never passed and the first
        # real arg wrongly binds to the method's `self` (e.g. transformers'
        # @torch.no_grad ConversionOps.convert -> "missing 'input_dict'"). Bind
        # the instance like a normal function descriptor. Plain context-manager
        # instances (no wrapped func) are never class attributes -> return self.
        if self._func is None or obj is None:
            return self
        import types as _types
        return _types.MethodType(self, obj)

    def __enter__(self):
        self._scope = self._scope_factory()
        return self._scope.__enter__()

    def __exit__(self, *exc):
        return self._scope.__exit__(*exc)


# Jittor's auto-mixed-precision control registers (src/type/nano_string.h).
# amp_reg drives dtype inference for every op created while it is set.
_AMP_PREFER32 = 1
_AMP_PREFER16 = 2
_AMP_KEEP_REDUCE = 4
_AMP_KEEP_WHITE = 8
_AMP_ARRAY_PREFER = 16
_AMP_PREFER_BFLOAT16 = 64

# Thread-local nesting state so `torch.is_autocast_enabled()` answers truthfully
# and nested/`enabled=False` regions restore the enclosing setting.
import threading as _threading

_autocast_state = _threading.local()


def _autocast_stack():
    stack = getattr(_autocast_state, "stack", None)
    if stack is None:
        stack = _autocast_state.stack = []
    return stack


def autocast_is_enabled(device_type=None):
    """True inside an *enabled* torch.autocast region (torch.is_autocast_enabled)."""
    for entry in reversed(_autocast_stack()):
        if device_type is None or entry["device_type"] == str(device_type):
            return bool(entry["enabled"])
    return False


def autocast_dtype(device_type=None):
    """dtype of the innermost enabled autocast region, else None."""
    for entry in reversed(_autocast_stack()):
        if device_type is None or entry["device_type"] == str(device_type):
            return entry["dtype"] if entry["enabled"] else None
    return None


def _autocast_default_dtype(device_type):
    # torch: float16 on cuda, bfloat16 on cpu.
    return "bfloat16" if str(device_type) in ("cpu", "") else "float16"


class _AutocastContext:
    """torch.autocast, implemented on jittor's amp registers.

    It used to be a total no-op: every argument was accepted, nothing changed,
    and a script that asked for mixed precision silently trained in float32 with
    none of the promised memory or speed -- while ``is_autocast_enabled()``
    agreed it was off, so nothing in the program could notice.

    The region now sets ``jt.flags.amp_reg`` so op dtype inference actually
    prefers the low-precision type (matmul/conv in fp16, ``exp``/``pow`` and
    reductions kept in fp32, mirroring torch's autocast lists), and restores the
    previous register on exit.  It is still BOTH a context manager and a
    decorator -- accelerate does ``new_forward = autocast(model_forward)``.

    BF16 uses an explicit native preference, so FP32 master parameters need
    not be mutated or permanently cast to select BF16 computation.
    """

    def __init__(self, device_type=None, dtype=None, enabled=True,
                 cache_enabled=None, *a, **k):
        # torch.cuda.amp.autocast()/torch.cpu.amp.autocast() omit device_type.
        if device_type is None:
            device_type = k.pop("device", None) or "cuda"
        self.device_type = str(device_type)
        self.enabled = bool(enabled)
        self.cache_enabled = cache_enabled
        if dtype is None:
            dtype = _autocast_default_dtype(self.device_type)
        name = getattr(dtype, "__name__", None) or _jittor_dtype_name(dtype)
        name = name.split(".")[-1]
        self.fast_dtype = name
        self._saved = None
        self._entered = 0
        if self.enabled and name not in ("float16", "half", "bfloat16",
                                         "float32", "float", "double",
                                         "float64"):
            from ..stub_policy import unimplemented
            unimplemented(
                "torch.autocast(dtype=%s)" % name,
                "accept an autocast dtype jittor cannot express and silently "
                "keep computing in the tensors' original dtype",
                "Use torch.float16, torch.bfloat16 or torch.float32.")

    def _amp_reg_for(self):
        if self.fast_dtype in ("float32", "float", "double", "float64"):
            return _AMP_PREFER32
        if self.fast_dtype == "bfloat16":
            return _AMP_PREFER16 | _AMP_PREFER_BFLOAT16
        return _AMP_PREFER16

    def __enter__(self):
        _autocast_stack().append({"device_type": self.device_type,
                                  "enabled": self.enabled,
                                  "dtype": self.fast_dtype})
        self._entered += 1
        self._saved = int(getattr(jt.flags, "amp_reg", 0))
        jt.flags.amp_reg = self._amp_reg_for() if self.enabled else 0
        return self

    def __exit__(self, *exc):
        if self._entered:
            self._entered -= 1
            stack = _autocast_stack()
            if stack:
                stack.pop()
            if self._saved is not None:
                jt.flags.amp_reg = self._saved
                self._saved = None
        return False

    def __call__(self, func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with type(self)(self.device_type, dtype=self.fast_dtype,
                            enabled=self.enabled,
                            cache_enabled=self.cache_enabled):
                return func(*args, **kwargs)
        return wrapper


def _amp_passthrough_decorator(fn=None, **kwargs):
    if fn is not None and callable(fn):
        return fn
    return lambda f: f


def _reduce_norm_group(value, how, group):
    if group is None or group.size() == 1:
        return value
    if group.ranks is None:
        return _collectives._reduce_scalar(value, how)
    if group.rank() < 0:
        raise RuntimeError("gradient norm called by a process-group nonmember")
    kind = group._get_backend_name()
    ops = getattr(jt.compile_extern, kind + "_ops", None)
    gather = getattr(ops, kind + "_all_gather", None)
    if gather is None or group._backend_handle is None:
        raise RuntimeError("gradient norm requires the mesh communicator")
    gathered = gather(value.reshape((1,)), group._backend_handle)
    return getattr(gathered, how)()


def _get_total_norm_device(grads, norm_type=2.0, error_if_nonfinite=False,
                           shard_reduce=False):
    """Compute the total norm for a list of gradient Vars on device.

    ``shard_reduce`` says these gradients are *shards*: each rank holds a
    different slice of the same logical gradient, so the norm has to be
    combined across ranks before the root is taken. Without it every rank
    clipped by its own slice's norm -- always smaller than the true one -- so
    each rank scaled by a different, too-large coefficient and the training
    trajectory silently diverged from torch's.

    It must stay off for DDP, where every rank already holds the *same*
    averaged gradient: reducing there would count the same norm N times.
    """
    import math as _math

    grads = [g for g in grads if isinstance(g, jt.Var)]
    if not grads:
        return jt.array(0.0)

    # Each shard set has its own communicator. Replicated gradients and
    # ordinary parameters are counted once; combining all through WORLD
    # would duplicate the replica axis and mix independent meshes.
    if shard_reduce is True and any(hasattr(g, "_fsdp_norm_group") for g in grads):
        groups: Dict[Any, List[Any]] = {}
        for grad in grads:
            group = getattr(grad, "_fsdp_norm_group", None)
            groups.setdefault(group, []).append(grad)
        norms = [_get_total_norm_device(values, norm_type, False,
                                       shard_reduce=group if group is not None else False)
                 for group, values in groups.items()]
        values = jt.concat([norm.reshape((1,)) for norm in norms])
        p = float(norm_type)
        if p == float("inf"):
            total = values.max()
        elif p == float("-inf"):
            total = values.min()
        elif p == 0:
            total = values.sum()
        else:
            total = (values ** p).sum() ** (1 / p)
        if error_if_nonfinite and not _math.isfinite(float(total.item())):
            raise RuntimeError("The total norm for gradients is non-finite")
        return total

    def _across(value, how):
        if not shard_reduce:
            return value
        if shard_reduce is not True:
            return _reduce_norm_group(value, how, shard_reduce)
        return _collectives._reduce_scalar(value, how)
    p = float(norm_type)
    acc_dtype = "float64" if any(_jittor_dtype_name(g.dtype) == "float64" for g in grads) else "float32"

    if p == 0.0:
        # torch first computes each tensor's zero-norm, then the zero-norm of
        # those scalars: this counts tensors containing at least one nonzero.
        nonempty = []
        for g in grads:
            x = g.abs() if "complex" in _jittor_dtype_name(g.dtype) else g
            nonempty.append((_across((x != 0).sum(), "max") != 0).reshape((1,)))
        total = jt.concat(nonempty).sum().cast(acc_dtype)
    else:
        parts = []
        for g in grads:
            x = g.abs() if "complex" in _jittor_dtype_name(g.dtype) else g
            parts.append(x.cast(acc_dtype).reshape((-1,)))
        flat = jt.concat(parts)
        ax = flat.abs()
        if p == float("inf"):
            local = ax.max() if int(flat.numel()) else jt.array(float("-inf")).cast(acc_dtype)
            total = _across(local, "max")
        elif p == float("-inf"):
            local = ax.min() if int(flat.numel()) else jt.array(float("inf")).cast(acc_dtype)
            total = _across(local, "min")
        elif p == 1.0:
            total = _across(ax.sum(), "sum")
        elif p == 2.0:
            # The cross-rank sum has to happen on the sum of squares, before
            # the square root -- combining per-rank norms afterwards would be
            # a different (and wrong) quantity.
            total = jt.sqrt(_across((flat * flat).sum(), "sum"))
        else:
            total = _across((ax ** p).sum(), "sum") ** (1.0 / p)

    if error_if_nonfinite:
        total_value = float(total.item())
        if not _math.isfinite(total_value):
            raise RuntimeError(
                "The total norm of order %s for gradients is non-finite, so it "
                "cannot be clipped. To disable this error set "
                "error_if_nonfinite=False." % norm_type
            )
    return total


def _clip_grads_with_norm_device(grads, max_norm, total_norm):
    """Scale gradient Vars using an already-computed total norm."""
    grads = [g for g in grads if isinstance(g, jt.Var)]
    if not grads:
        return

    acc_dtype = "float64" if _jittor_dtype_name(total_norm.dtype) == "float64" else "float32"
    limit = float(max_norm)
    if limit == float("inf"):
        return
    scalar_type = np.float64 if _jittor_dtype_name(acc_dtype) == "float64" else np.float32
    raw_coef = scalar_type(limit) / (total_norm + scalar_type(1e-6))
    coef = jt.minimum(cast(Any, raw_coef), jt.array(float(scalar_type(1.0))).cast(acc_dtype))
    # CUDA fmin-style minimum may select the finite operand for NaN. Torch
    # propagates a NaN total norm into every gradient when errors are disabled.
    coef = jt.ternary(jt.isnan(raw_coef), raw_coef, coef)
    for g in grads:
        g.update(g * coef.cast(_jittor_dtype_name(g.dtype)))


def _clip_grad_norm_device(grads, max_norm, norm_type=2.0,
                           error_if_nonfinite=False, shard_reduce=False):
    """Clip a list of gradient Vars without a host-side coefficient branch.

    A per-gradient reduction is mathematically equivalent for finite p-norms,
    but it creates one small CUDA reduction per parameter tensor. Transformers
    commonly have hundreds of tensors, making those launches much more costly
    than the single flat reduction used here. The device coefficient removes
    the per-step D2H sync previously caused by ``total.item()``.
    """
    grads = [g for g in grads if isinstance(g, jt.Var)]
    total = _get_total_norm_device(grads, norm_type, error_if_nonfinite,
                                   shard_reduce=shard_reduce)
    _clip_grads_with_norm_device(grads, max_norm, total)
    return total


class _GradScaler:
    """Functional fp16 dynamic loss scaler (matches torch.cuda.amp.GradScaler).
    Works with the jittor optimizer bridge: scale(loss).backward() routes scaled
    grads into the optimizer; step() unscales, SKIPS the step on inf/nan, and
    update() grows/backs off the scale. bf16 doesn't need scaling but this is
    correct (and required) for fp16 mixed-precision training."""
    def __init__(self, *args, **kwargs):
        local_args: Any = list(args)
        self._device = str(kwargs.pop("device", "cuda"))
        if local_args and (isinstance(local_args[0], str) or
                     local_args[0].__class__.__name__ in ("device", "_Device")):
            self._device = str(local_args.pop(0))
        names = ("init_scale", "growth_factor", "backoff_factor", "growth_interval", "enabled")
        defaults = (2.0 ** 16, 2.0, 0.5, 2000, True)
        if len(local_args) > len(names):
            raise TypeError("GradScaler received too many positional arguments")
        values = []
        for index, (name, default) in enumerate(zip(names, defaults)):
            if index < len(local_args):
                if name in kwargs:
                    raise TypeError("GradScaler got multiple values for argument '%s'" % name)
                values.append(local_args[index])
            else:
                values.append(kwargs.pop(name, default))
        if kwargs:
            raise TypeError("GradScaler got an unexpected keyword argument '%s'" % next(iter(kwargs)))
        init_scale, growth_factor, backoff_factor, growth_interval, enabled = values
        self._enabled = bool(enabled)
        if self._enabled:
            assert growth_factor > 1.0, "The growth factor must be > 1.0."
            assert backoff_factor < 1.0, "The backoff factor must be < 1.0."
        self._init_scale = float(init_scale)
        self._scale = None
        self._scale_value = None
        self._scale_copies = {}
        self._growth_factor = float(growth_factor)
        self._backoff_factor = float(backoff_factor)
        self._growth_interval = int(growth_interval)
        self._growth_tracker = 0
        self._per_optimizer_states = {}

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        if not self._enabled:
            return 1.0
        return self._init_scale if self._scale is None else self._scale_value

    def _new_scale_like(self, value, reference):
        from .context import get_install_context
        from .frontend import tensor_frontend
        tensor_type = get_install_context(jt).target_namespace.Var
        with tensor_frontend(tensor_type, like=reference), jt.flag_scope(amp_reg=0):
            scale = jt.array(value, dtype="float32").reshape(())
        return scale.stop_grad()

    def _set_scale(self, value):
        self._scale_value = float(np.float32(value))
        self._scale.update(self._new_scale_like(self._scale_value, self._scale))
        self._scale_copies.clear()

    def scale(self, outputs):
        if not self._enabled:
            return outputs
        if isinstance(outputs, jt.Var):
            if self._scale is None:
                self._scale = self._new_scale_like(self._init_scale, outputs)
                self._scale_value = float(np.float32(self._init_scale))
            scale = self._scale
            placement = jt.core.dispatch_context([outputs])
            if jt.core.dispatch_context([scale]) != placement:
                if placement not in self._scale_copies:
                    self._scale_copies[placement] = self._new_scale_like(self._scale_value, outputs)
                scale = self._scale_copies[placement]
            return outputs * scale
        from collections.abc import Iterable
        if isinstance(outputs, Iterable):
            scaled = map(self.scale, outputs)
            return type(outputs)(scaled) if isinstance(outputs, (list, tuple)) else scaled
        raise ValueError("outputs must be a Tensor or an iterable of Tensors")

    def _check_initialized(self, name):
        assert self._scale is not None, (
            "Attempted %s but _scale is None. This may indicate your script did not "
            "use scaler.scale(loss or outputs) earlier in the iteration." % name)

    def _optimizer_state(self, optimizer):
        return self._per_optimizer_states.setdefault(
            id(optimizer), {"stage": "ready", "found_inf": False, "checked": False})

    def _grads(self, opt):
        gs = []
        for pg in getattr(opt, "param_groups", []):
            gradients = pg.get("grads")
            if gradients is None:
                gradients = [getattr(p, "grad", None) for p in pg.get("params", ())]
            for g in gradients:
                if g is not None:
                    gs.append(g)
        return gs

    def unscale_(self, opt):
        if not self._enabled:
            return
        self._check_initialized("unscale_")
        state = self._optimizer_state(opt)
        if state["stage"] == "unscaled":
            raise RuntimeError("unscale_() has already been called on this optimizer since the last update().")
        if state["stage"] == "stepped":
            raise RuntimeError("unscale_() is being called after step().")
        grads = self._grads(opt)
        if any(_jittor_dtype_name(g.dtype) == "float16" for g in grads):
            raise ValueError("Attempting to unscale FP16 gradients.")
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            inv = np.float32(np.float64(1.0) / np.float64(self._scale_value))
        flattened = []
        for g in grads:
            if not g.numel():
                continue
            check = g
            if (_jittor_dtype_name(g.dtype) == "float64"
                    and jt.core.dispatch_context([g])[0] == "cuda"):
                # Torch's CUDA foreach AMP check detects non-finites in FP32,
                # while its unscale calculation and CPU check retain FP64.
                check = g.float32()
            flattened.append(jt.isfinite(check).reshape((-1,)))
            unscaled = g * inv
            if _jittor_dtype_name(unscaled.dtype) != _jittor_dtype_name(g.dtype):
                unscaled = unscaled.cast(_jittor_dtype_name(g.dtype))
            g.update(unscaled)
        # Optimizer.step still needs a host decision to skip state updates, but
        # one flat reduction avoids both per-gradient reductions and per-gradient
        # D2H syncs. Torch checks the scaled gradients before multiplying by
        # the reciprocal. Boolean masks retain each backend's checking domain.
        state["found_inf"] = (
            not bool(jt.concat(flattened).all().item())
            if flattened else False
        )
        state["checked"] = bool(grads)
        state["stage"] = "unscaled"

    def step(self, opt, *a, **k):
        if not self._enabled:
            return opt.step(*a, **k)
        if "closure" in k:
            raise RuntimeError("Closure use is not currently supported if GradScaler is enabled.")
        self._check_initialized("step")
        state = self._optimizer_state(opt)
        if state["stage"] == "stepped":
            raise RuntimeError("step() has already been called since the last update().")
        if state["stage"] == "ready":
            self.unscale_(opt)
        assert state["checked"], "No inf checks were recorded for this optimizer."
        result = None if state["found_inf"] else opt.step(*a, **k)
        state["stage"] = "stepped"
        return result

    def update(self, new_scale=None):
        if not self._enabled:
            return
        self._check_initialized("update")
        if new_scale is not None:
            if isinstance(new_scale, jt.Var):
                assert new_scale.numel() == 1 and not new_scale.requires_grad
                assert jt.core.dispatch_context([new_scale]) == jt.core.dispatch_context([self._scale])
                new_scale = float(new_scale.item())
            self._set_scale(float(new_scale))
        else:
            checked = [s for s in self._per_optimizer_states.values() if s["checked"]]
            assert checked, "No inf checks were recorded prior to update."
            self._update_scale(any(s["found_inf"] for s in checked))
        self._per_optimizer_states.clear()

    def _update_scale(self, found_inf):
        value = self.get_scale()
        if found_inf:
            value = np.float32(value) * np.float32(self._backoff_factor)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                with np.errstate(over="ignore"):
                    grown = np.float32(value) * np.float32(self._growth_factor)
                if np.isfinite(grown):
                    value = grown
                self._growth_tracker = 0
        self._set_scale(value)

    def get_growth_factor(self):
        return self._growth_factor

    def set_growth_factor(self, value):
        self._growth_factor = value

    def get_backoff_factor(self):
        return self._backoff_factor

    def set_backoff_factor(self, value):
        self._backoff_factor = value

    def get_growth_interval(self):
        return self._growth_interval

    def set_growth_interval(self, value):
        self._growth_interval = value

    def state_dict(self):
        if not self._enabled:
            return {}
        return {"scale": self.get_scale(), "growth_factor": self._growth_factor,
                "backoff_factor": self._backoff_factor, "growth_interval": self._growth_interval,
                "_growth_tracker": self._growth_tracker}

    def load_state_dict(self, sd):
        if not self._enabled:
            return
        if not sd:
            raise RuntimeError("The source state dict is empty, possibly because it was saved from a disabled instance of GradScaler.")
        self._init_scale = float(sd["scale"])
        if self._scale is not None:
            self._set_scale(self._init_scale)
        else:
            self._scale_copies.clear()
        self._growth_factor = sd.get("growth_factor", self._growth_factor)
        self._backoff_factor = sd.get("backoff_factor", self._backoff_factor)
        self._growth_interval = sd.get("growth_interval", self._growth_interval)
        self._growth_tracker = sd.get("_growth_tracker", sd.get("growth_tracker", 0))
