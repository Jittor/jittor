"""Gradient mode, clipping, autocast, and loss-scaling compatibility."""

import numpy as np
import jittor as jt


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


# Jittor's auto-mixed-precision control registers (src/misc/nano_string.h).
# amp_reg drives dtype inference for every op created while it is set.
_AMP_PREFER32 = 1
_AMP_PREFER16 = 2
_AMP_KEEP_REDUCE = 4
_AMP_KEEP_WHITE = 8
_AMP_ARRAY_PREFER = 16

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

    Known difference from torch, warned about once: jittor's register selects
    *bfloat16* only when an operand already is bfloat16, so an all-float32 model
    under ``autocast(dtype=torch.bfloat16)`` computes in float16 -- same
    mantissa-or-better, narrower exponent range.
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
        name = getattr(dtype, "__name__", None) or str(dtype)
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
            from ..stub_policy import degraded
            degraded(
                "torch.autocast(dtype=torch.bfloat16)",
                "jittor's amp register keeps bfloat16 only when an operand "
                "already is bfloat16; an all-float32 region computes in "
                "float16 instead",
                "Cast the module with .to(torch.bfloat16) to stay in bfloat16.")
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


def _get_total_norm_device(grads, norm_type=2.0, error_if_nonfinite=False):
    """Compute the total norm for a list of gradient Vars on device."""
    import math as _math

    grads = [g for g in grads if isinstance(g, jt.Var)]
    if not grads:
        return jt.array(0.0)
    p = float(norm_type)
    acc_dtype = "float64" if any(str(g.dtype) == "float64" for g in grads) else "float32"

    if p == 0.0:
        # torch first computes each tensor's zero-norm, then the zero-norm of
        # those scalars: this counts tensors containing at least one nonzero.
        nonempty = []
        for g in grads:
            x = g.abs() if "complex" in str(g.dtype) else g
            nonempty.append((x != 0).sum().reshape((1,)))
        total = (jt.concat(nonempty) != 0).sum().cast(acc_dtype)
    else:
        parts = []
        for g in grads:
            x = g.abs() if "complex" in str(g.dtype) else g
            parts.append(x.cast(acc_dtype).reshape((-1,)))
        flat = jt.concat(parts)
        ax = flat.abs()
        if p == float("inf"):
            total = ax.max()
        elif p == float("-inf"):
            total = ax.min()
        elif p == 1.0:
            total = ax.sum()
        elif p == 2.0:
            total = jt.sqrt((flat * flat).sum())
        else:
            total = (ax ** p).sum() ** (1.0 / p)

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

    acc_dtype = "float64" if str(total_norm.dtype) == "float64" else "float32"
    limit = float(max_norm)
    if limit == float("inf"):
        return
    scalar_type = np.float64 if acc_dtype == "float64" else np.float32
    raw_coef = scalar_type(limit) / (total_norm + scalar_type(1e-6))
    coef = jt.minimum(raw_coef, scalar_type(1.0))
    # CUDA fmin-style minimum may select the finite operand for NaN. Torch
    # propagates a NaN total norm into every gradient when errors are disabled.
    coef = jt.ternary(jt.isnan(raw_coef), raw_coef, coef)
    for g in grads:
        g.update(g * coef.cast(str(g.dtype)))


def _clip_grad_norm_device(grads, max_norm, norm_type=2.0,
                           error_if_nonfinite=False):
    """Clip a list of gradient Vars without a host-side coefficient branch.

    A per-gradient reduction is mathematically equivalent for finite p-norms,
    but it creates one small CUDA reduction per parameter tensor. Transformers
    commonly have hundreds of tensors, making those launches much more costly
    than the single flat reduction used here. The device coefficient removes
    the per-step D2H sync previously caused by ``total.item()``.
    """
    grads = [g for g in grads if isinstance(g, jt.Var)]
    total = _get_total_norm_device(grads, norm_type, error_if_nonfinite)
    _clip_grads_with_norm_device(grads, max_norm, total)
    return total


class _GradScaler:
    """Functional fp16 dynamic loss scaler (matches torch.cuda.amp.GradScaler).
    Works with the jittor optimizer bridge: scale(loss).backward() routes scaled
    grads into the optimizer; step() unscales, SKIPS the step on inf/nan, and
    update() grows/backs off the scale. bf16 doesn't need scaling but this is
    correct (and required) for fp16 mixed-precision training."""
    def __init__(self, *args, **kwargs):
        # torch >=2.3 changed the signature to GradScaler(device="cuda",
        # init_scale=..., ...); accelerate/transformers call GradScaler("cuda").
        # The legacy torch.cuda.amp.GradScaler took init_scale first. Detect a
        # leading device positional (a str like "cuda" or a device object) and
        # shift it out, so BOTH signatures work.
        args = list(args)
        if args and (isinstance(args[0], str) or
                     args[0].__class__.__name__ in ("device", "_Device")):
            args = args[1:]                     # drop the device positional
        kwargs.pop("device", None)
        init_scale = kwargs.pop("init_scale", args[0] if len(args) > 0 else 2.0 ** 16)
        growth_factor = kwargs.pop("growth_factor", args[1] if len(args) > 1 else 2.0)
        backoff_factor = kwargs.pop("backoff_factor", args[2] if len(args) > 2 else 0.5)
        growth_interval = kwargs.pop("growth_interval", args[3] if len(args) > 3 else 2000)
        enabled = kwargs.pop("enabled", args[4] if len(args) > 4 else True)
        self._enabled = enabled
        self._scale = float(init_scale)
        self._growth_factor = growth_factor
        self._backoff_factor = backoff_factor
        self._growth_interval = growth_interval
        self._growth_tracker = 0
        self._found_inf = False
        self._unscaled = False

    def is_enabled(self):
        return self._enabled

    def get_scale(self):
        return self._scale if self._enabled else 1.0

    def scale(self, outputs):
        return outputs * self._scale if self._enabled else outputs

    def _grads(self, opt):
        gs = []
        for pg in getattr(opt, "param_groups", []):
            for g in (pg.get("grads", []) or []):
                if g is not None:
                    gs.append(g)
        return gs

    def unscale_(self, opt):
        if not self._enabled:
            return
        inv = np.float32(1.0 / self._scale)
        flattened = []
        for g in self._grads(opt):
            if not g.numel():
                continue
            unscaled = g * inv
            if str(unscaled.dtype) != str(g.dtype):
                unscaled = unscaled.cast(str(g.dtype))
            g.update(unscaled)
            flattened.append(unscaled.cast("float32").reshape((-1,)))
        # Optimizer.step still needs a host decision to skip state updates, but
        # one flat reduction avoids both per-gradient reductions and per-gradient
        # D2H syncs. The finite check consumes the values actually assigned back
        # to the gradients, including any low-precision overflow from unscaling.
        self._found_inf = (
            not bool(jt.isfinite(jt.concat(flattened)).all().item())
            if flattened else False
        )
        self._unscaled = True

    def step(self, opt, *a, **k):
        if not self._enabled:
            return opt.step(*a, **k)
        if not self._unscaled:
            self.unscale_(opt)
        self._unscaled = False
        if self._found_inf:
            return None  # skip optimizer step on overflow
        return opt.step(*a, **k)

    def update(self, new_scale=None):
        if not self._enabled:
            return
        if new_scale is not None:
            self._scale = float(new_scale); return
        if self._found_inf:
            self._scale = max(1.0, self._scale * self._backoff_factor)
            self._growth_tracker = 0
        else:
            self._growth_tracker += 1
            if self._growth_tracker >= self._growth_interval:
                self._scale *= self._growth_factor
                self._growth_tracker = 0
        self._found_inf = False

    def state_dict(self):
        return {"scale": self._scale, "growth_tracker": self._growth_tracker}

    def load_state_dict(self, sd):
        self._scale = sd.get("scale", self._scale)
        self._growth_tracker = sd.get("growth_tracker", 0)
