"""Stable Torch autograd owners over the native Var/Op differentiation graph.

Installation publishes these objects. Native delegates belong to InstallContext;
Function input/output bookkeeping belongs to each native one-shot call context.
"""
import contextlib
import jittor as jt
import numpy as np
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from .context import get_install_context
from ._placeholder_context import _PlaceholderContext
from ..diagnostics import EXPECTED, swallowed


def _state():
    return get_install_context(jt).state["autograd_api"]


def save_for_backward(self, *tensors):
    # torch stores a tuple; a single un-tupled call still yields a tuple
    self._saved_tensors = tuple(tensors)
    # torch also records each saved tensor's version counter and raises
    # if the tensor was modified in place before backward reads it.
    # Without that, an in-place edit between forward and backward is
    # used silently: the backward computes with the NEW values and
    # returns a gradient for a forward that never happened.
    # jittor has no version counter, but Var.id is one in practice --
    # it is stable across reads, sync and numpy(), and changes on
    # update()/assign()/setitem/augmented assignment.
    #
    # Scope: jittor's Function.__call__ tapes its inputs, so what a
    # forward saves is the taped Var, not the caller's object. This
    # therefore catches an in-place edit of the tensor the forward
    # actually saved (the usual case: an intermediate computed inside
    # forward), not an edit of the caller's original made afterwards.
    self._saved_versions = tuple(
        (t.id if isinstance(t, jt.Var) else None) for t in tensors)


def _saved_tensors(self):
    saved = getattr(self, "_saved_tensors", ())
    versions = getattr(self, "_saved_versions", None)
    if versions:
        for tensor, version in zip(saved, versions):
            if isinstance(tensor, jt.Var) and version is not None \
                    and tensor.id != version:
                raise RuntimeError(
                    "one of the variables needed for gradient "
                    "computation has been modified by an inplace "
                    "operation")
    return saved


def _sum_grad_to(grad, shape):
    if grad is None or shape is None or not isinstance(grad, jt.Var):
        return grad
    gshape = grad.shape
    if list(gshape) == list(shape):
        return grad
    # Incompatible element counts: the returned grad does not correspond to
    # this input's true gradient. A custom Function may return a fully-shaped
    # grad for an input it actually ignores (3DGS's rasterizer returns a
    # [P,C] grad for the EMPTY placeholder inputs colors_precomp / cov3Ds_precomp
    # that aren't requires_grad in torch). torch discards grads for such
    # inputs; emulate by returning a correctly-shaped zero (jittor still tapes
    # the placeholder Var, so it needs a shape-matching grad, not None).
    tgt_items = 1
    for s in shape: tgt_items *= int(s)
    g_items = 1
    for s in gshape: g_items *= int(s)
    if tgt_items == 0 or (tgt_items != g_items and g_items % max(tgt_items, 1) != 0):
        # The returned grad cannot be reduced to this input's shape by any
        # broadcast rule, so it is almost certainly a mistake in the
        # Function's backward. Substituting zeros keeps the 3DGS-style
        # placeholder case working, but doing it SILENTLY hides a genuinely
        # wrong backward -- the gradient just becomes zero and the model
        # quietly stops learning through that input.
        import warnings as _warnings
        _warnings.warn(
            "a custom autograd Function returned a gradient of shape %s "
            "for an input of shape %s; the element counts (%d vs %d) are "
            "not broadcast-compatible, so a zero gradient is used for that "
            "input. Check the backward's return order."
            % (tuple(gshape), tuple(shape), g_items, tgt_items),
            RuntimeWarning, stacklevel=3)
        return jt.zeros([int(s) for s in shape], dtype=grad.dtype)
    # drop leading dims that the input doesn't have (broadcast prepended them)
    extra = len(gshape) - len(shape)
    if extra > 0:
        grad = grad.sum(dims=tuple(range(extra)))
        gshape = grad.shape
    # sum over dims where the input was size-1 but grad is larger (keepdim)
    reduce_dims = [i for i in range(len(shape))
                   if int(shape[i]) == 1 and int(gshape[i]) != 1]
    if reduce_dims:
        grad = grad.sum(dims=tuple(reduce_dims), keepdims=True)
    if list(grad.shape) != list(shape):
        grad = grad.reshape(tuple(int(s) for s in shape))
    return grad


def _call_record_inputs(self, *args, **kw):
    # Native Jittor Functions provide grad() directly and do not need any
    # torch ctx bookkeeping. ACL uses many such Functions in model hot
    # paths, so avoid recording shapes and requires-grad state for them.
    if getattr(type(self), "backward", None) is None:
        return _state()["native_function_call"](self, *args, **kw)
    # Everything below is recorded on the CALL CONTEXT, never on the
    # Function instance. jittor's Function runs each call against a
    # one-shot context (a shallow copy of the instance), which is the
    # object handed to forward() and to backward() as torch's `ctx`.
    # Recording on `self` used to look like it worked because the context
    # is copied from the instance at the start of the call -- but only for
    # the values written BEFORE the call. `_fwd_outputs`, written after it,
    # never reached the backward at all, so materialize_grads silently did
    # nothing and a Function with an unused output got None where torch
    # gives zeros. Building the context here and running the call against
    # it puts all three on the object that backward() actually reads.
    ctx = self._new_call_context()
    # capture forward input shapes (positional only -- jittor only tapes those)
    try:
        ctx._fwd_input_shapes = [
            (tuple(v.shape) if isinstance(v, jt.Var) else None) for v in args]
    except EXPECTED as exc:
        swallowed("torch/installers/autograd.py _call_record_inputs: ctx._fwd_input_shapes = [", exc)
        ctx._fwd_input_shapes = None
    # torch.autograd.Function exposes `ctx.needs_input_grad`: one bool per
    # argument PASSED to apply(), True iff it is a tensor requiring grad.
    # Custom Functions branch on it (e.g. flex_gemm spconv:
    # `need_grad = any(ctx.needs_input_grad)`).
    #
    # Checked against real torch 2.12: `apply(a, b, 3.0)` gives three flags
    # and `apply(a, b)` on a `forward(ctx, a, b, c=1.0)` gives *two* -- the
    # tuple follows the call, not the signature -- and `apply()` rejects
    # keyword arguments outright. The positional tuple below therefore
    # already matches torch; what did not match was the failure mode of a
    # keyword call, which fell through to jittor's Function.__call__ and
    # raised "Function.__call__() got an unexpected keyword argument".
    # Reject it here with torch's own wording instead.
    if kw:
        raise TypeError("apply() takes no keyword arguments")
    try:
        ctx.needs_input_grad = tuple(
            bool(isinstance(v, jt.Var) and v.requires_grad) for v in args)
    except EXPECTED as exc:
        swallowed("torch/installers/autograd.py _call_record_inputs: ctx.needs_input_grad = tuple(", exc)
        ctx.needs_input_grad = tuple(isinstance(v, jt.Var) for v in args)
    out = ctx._run_call(*args, **kw)
    # Capture each forward OUTPUT's (shape, dtype) so the grad bridge can
    # materialize a zeros grad for outputs that don't reach the backward'd
    # scalar (torch's materialize_grads=True; see grad() below).
    try:
        outs = out if isinstance(out, (tuple, list)) else (out,)
        ctx._fwd_outputs = [
            (tuple(o.shape), _jittor_dtype_name(o.dtype)) if isinstance(o, jt.Var) else None
            for o in outs]
    except EXPECTED as exc:
        swallowed("torch/installers/autograd.py _call_record_inputs: outs = out if isinstance(out, (tuple, list)) else (out,)", exc)
        ctx._fwd_outputs = None
    return out


def set_materialize_grads(self, value):
    self._materialize_grads = bool(value)


def function_grad(self, *grad_outputs):
    bw = getattr(type(self), "backward", None)
    if bw is None:
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute 'grad'")
    # materialize_grads (torch default True): jittor hands None for a taped
    # output that doesn't reach the backward'd scalar, but torch passes
    # zeros_like(output) for FLOATING-point outputs (int/bool ones stay
    # None — non-differentiable). 3DGS's rasterizer returns (color, radii,
    # depth); a colour-only loss leaves depth's grad None, yet the C++
    # backward requires a real zero tensor for it.
    if getattr(self, "_materialize_grads", True) and any(
            g is None for g in grad_outputs):
        outs = getattr(self, "_fwd_outputs", None)
        if outs is not None:
            go = list(grad_outputs)
            for i in range(min(len(go), len(outs))):
                if go[i] is None and outs[i] is not None:
                    shp, dt = outs[i]
                    if not any(t in dt for t in ("int", "bool", "uint")):
                        go[i] = jt.zeros(shp, dtype=dt)
            grad_outputs = tuple(go)
    ret = bw(self, *grad_outputs)
    shapes = getattr(self, "_fwd_input_shapes", None)
    if shapes is None:
        return ret
    single = not isinstance(ret, (tuple, list))
    grads = [ret] if single else list(ret)
    # reduce each input-grad to its forward input shape (torch broadcast bwd)
    for i in range(min(len(grads), len(shapes))):
        grads[i] = _sum_grad_to(grads[i], shapes[i])
    return grads[0] if single else tuple(grads)


def _as_list(x):
    if isinstance(x, jt.Var):
        return [x]
    return list(x)


def grad(outputs, inputs, grad_outputs=None, retain_graph=None,
         create_graph=False, only_inputs=True, allow_unused=None,
         is_grads_batched=False, materialize_grads=False, **kw):
    # torch.autograd.grad(outputs, inputs, ...) -> tuple of grads, one per
    # input. jittor's jt.grad takes a single scalar loss; when several
    # outputs (or grad_outputs weights) are given, reduce them to one scalar
    # via sum(grad_outputs * output), matching torch's vector-Jacobian product.
    outs = _as_list(outputs)
    ins = _as_list(inputs)
    if grad_outputs is None:
        # torch: "grad can be implicitly created only for scalar outputs".
        # This used to sum every output element, i.e. silently assume a
        # grad_output of ones -- a different vector-Jacobian product from
        # the one the caller forgot to specify, with no error.
        non_scalar = [tuple(o.shape) for o in outs
                      if isinstance(o, jt.Var) and o.numel() != 1]
        if non_scalar:
            raise RuntimeError(
                "grad can be implicitly created only for scalar outputs "
                "(got output shape(s) %s); pass grad_outputs="
                % ", ".join(str(shape) for shape in non_scalar))
        loss = outs[0].sum() if len(outs) == 1 else sum(o.sum() for o in outs)
    else:
        gos = _as_list(grad_outputs)
        loss = sum((o * w).sum() for o, w in zip(outs, gos))
    # torch keeps these two separate: retain_graph decides whether the graph
    # survives the call, create_graph decides whether the RETURNED grads are
    # themselves differentiable. Folding create_graph into retain_graph made
    # create_graph=False still hand back differentiable tensors.
    rg = bool(create_graph) if retain_graph is None else bool(retain_graph)
    if materialize_grads and allow_unused is False:
        raise ValueError(
            "Expected allow_unused to be True or not passed when "
            "materialize_grads=True, but got: allow_unused=False.")
    allow_unused = bool(materialize_grads) if allow_unused is None \
        else bool(allow_unused)
    gs = list(jt.core.grad_optional(loss, ins, rg))
    missing = [i for i, value in enumerate(gs) if value is None]
    if missing and materialize_grads:
        for i in missing:
            gs[i] = jt.zeros_like(ins[i])
            if create_graph:
                gs[i].start_grad()
            else:
                gs[i].stop_grad()
    elif missing and not allow_unused:
        raise RuntimeError(
            "One of the differentiated Tensors appears to not have been "
            "used in the graph. Set allow_unused=True if this is desired.")
    if not create_graph:
        # detach() severs the graph edge but leaves jittor's requires_grad
        # flag set, and torch reports requires_grad=False here; clear it on
        # the detached copy so the *returned* tensor answers like torch's
        # without touching the graph the caller may still be using.
        detached = []
        for g in gs:
            if isinstance(g, jt.Var):
                g = g.detach()
                g.stop_grad()
            detached.append(g)
        gs = detached
    return tuple(gs)


def backward(tensors, grad_tensors=None, retain_graph=None,
             create_graph=False, inputs=None, **kw):
    # torch.autograd.backward(tensors, ...) accumulates grads into leaf
    # .grad. Route each tensor through Var.backward (the optimizer bridge /
    # no-optimizer leaf path installed on Var).
    ts = _as_list(tensors)
    gts = None if grad_tensors is None else _as_list(grad_tensors)
    for i, t in enumerate(ts):
        gt = None if gts is None else gts[i]
        t.backward(gradient=gt, retain_graph=retain_graph)
    return None


class EventList(list):
    def table(self, *args, **kwargs):
        return ""
    def export_chrome_trace(self, *args, **kwargs):
        return None


class _RecordFunction(_PlaceholderContext):
    """Autograd record scope placeholder; no events are collected."""


class profile(_RecordFunction):
    def function_events(self):
        return EventList()
    def key_averages(self, *args, **kwargs):
        return EventList()
    def export_chrome_trace(self, *args, **kwargs):
        return None


class saved_tensors_hooks:
    def __init__(self, pack_hook=None, unpack_hook=None):
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class save_on_cpu(saved_tensors_hooks):
    def __init__(self, pin_memory=False, device_type="cuda"):
        super().__init__(None, None)


def _normalize_index(idx):
    if isinstance(idx, jt.Var):
        arr = np.asarray(idx.detach().cpu().numpy())
        if arr.ndim == 0:
            return bool(arr.item()) if arr.dtype == np.bool_ else int(arr.item())
        if arr.dtype == np.bool_:
            return [int(i) for i in np.flatnonzero(arr)]
        return [int(x) for x in arr.reshape(-1)]
    if isinstance(idx, tuple):
        return tuple(_normalize_index(i) for i in idx)
    if isinstance(idx, list):
        return [_normalize_index(i) for i in idx]
    return idx


_call_record_inputs._torch_records_inputs = True


class Function(jt.Function):
    """Torch-style callbacks with native per-call tape and context ownership."""

    def execute(self, *args, **kwargs):
        return type(self).forward(self, *args, **kwargs)

    __call__ = _call_record_inputs
    save_for_backward = save_for_backward
    saved_tensors = property(_saved_tensors)
    set_materialize_grads = set_materialize_grads
    grad = function_grad


class FunctionCtx:
    save_for_backward = save_for_backward
    saved_tensors = property(_saved_tensors)
    set_materialize_grads = set_materialize_grads


class Node:
    """Metadata placeholder; native connectivity is exposed by Tensor.grad_fn."""


def once_differentiable(function):
    return function


def set_detect_anomaly(*args, **kwargs):
    return contextlib.nullcontext()


def detect_anomaly(*args, **kwargs):
    return contextlib.nullcontext()


def record_function(*args, **kwargs):
    return _RecordFunction()


def emit_nvtx(*args, **kwargs):
    return _RecordFunction()


def kineto_available():
    return False


class _Engine:
    @staticmethod
    def queue_callback(*args, **kwargs):
        return None


class Variable:
    _execution_engine = _Engine()


def _tensordict_getitem(self, index):
    return _state()["tensordict_getitem"](self, _normalize_index(index))


def _lazy_tensordict_getitem(self, index):
    return _state()["lazy_tensordict_getitem"](self, _normalize_index(index))
