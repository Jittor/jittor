"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt
import numpy as np

from ..context import registry_for
from ...diagnostics import EXPECTED, swallowed


def _install_tensordict_compat():
    """Patch tensordict indexing for jittor Vars used as torch-style indices."""
    try:
        from tensordict.base import TensorDictBase
        from tensordict._lazy import LazyStackedTensorDict
    except EXPECTED as exc:
        swallowed("torch/installers/autograd.py _install_tensordict_compat: from tensordict.base import TensorDictBase", exc)
        return

    if getattr(TensorDictBase, "_jittor_index_compat", False):
        return

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

    _td_getitem = TensorDictBase.__getitem__
    def _getitem(self, index):
        return _td_getitem(self, _normalize_index(index))
    TensorDictBase.__getitem__ = _getitem
    TensorDictBase.__getitems__ = _getitem

    _lazy_getitem = LazyStackedTensorDict.__getitem__
    def _lazy_index(self, index):
        return _lazy_getitem(self, _normalize_index(index))
    LazyStackedTensorDict.__getitem__ = _lazy_index

    TensorDictBase._jittor_index_compat = True


def _install_autograd_function(g):
    """torch.autograd.Function exposes ctx.save_for_backward(*tensors) in
    forward() and a ctx.saved_tensors tuple in backward(). jittor's Function
    stores backward state via plain `self.<attr> = ...`, so it lacks both
    (bloom's GeLUFunction calls them). Add them to the Function class.
    """
    Fn = getattr(g, "Function", None)
    if Fn is None:
        return
    if not hasattr(Fn, "save_for_backward"):
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
        Fn.save_for_backward = save_for_backward
    if "saved_tensors" not in getattr(Fn, "__dict__", {}):
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
        Fn.saved_tensors = property(_saved_tensors)
    # torch's autograd engine reduces (sums) each grad a Function.backward returns
    # down to the shape of the corresponding *input* whenever forward broadcast that
    # input (e.g. TOOD's SigmoidGeometricMean multiplies cls_logits [N,80,H,W] by
    # cls_prob [N,1,H,W]; backward returns grad_y at [N,80,H,W]). jittor performs no
    # such reduction and raises "dvar->num != var->num". Record the forward input
    # shapes on __call__, then sum-to-shape each returned grad in the grad bridge.
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

    _orig_fn_call = Fn.__call__
    def _call_record_inputs(self, *args, **kw):
        # Native Jittor Functions provide grad() directly and do not need any
        # torch ctx bookkeeping. ACL uses many such Functions in model hot
        # paths, so avoid recording shapes and requires-grad state for them.
        if getattr(type(self), "backward", None) is None:
            return _orig_fn_call(self, *args, **kw)
        # capture forward input shapes (positional only -- jittor only tapes those)
        try:
            self._fwd_input_shapes = [
                (tuple(v.shape) if isinstance(v, jt.Var) else None) for v in args]
        except EXPECTED as exc:
            swallowed("torch/installers/autograd.py _call_record_inputs: self._fwd_input_shapes = [", exc)
            self._fwd_input_shapes = None
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
            self.needs_input_grad = tuple(
                bool(isinstance(v, jt.Var) and v.requires_grad) for v in args)
        except EXPECTED as exc:
            swallowed("torch/installers/autograd.py _call_record_inputs: self.needs_input_grad = tuple(", exc)
            self.needs_input_grad = tuple(isinstance(v, jt.Var) for v in args)
        out = _orig_fn_call(self, *args, **kw)
        # Capture each forward OUTPUT's (shape, dtype) so the grad bridge can
        # materialize a zeros grad for outputs that don't reach the backward'd
        # scalar (torch's materialize_grads=True; see grad() below).
        try:
            outs = out if isinstance(out, (tuple, list)) else (out,)
            self._fwd_outputs = [
                (tuple(o.shape), str(o.dtype)) if isinstance(o, jt.Var) else None
                for o in outs]
        except EXPECTED as exc:
            swallowed("torch/installers/autograd.py _call_record_inputs: outs = out if isinstance(out, (tuple, list)) else (out,)", exc)
            self._fwd_outputs = None
        return out
    if getattr(Fn.__call__, "_torch_records_inputs", False) is not True:
        _call_record_inputs._torch_records_inputs = True
        Fn.__call__ = _call_record_inputs

    # torch.autograd.Function defines `@staticmethod backward(ctx, *grad_outputs)`;
    # jittor's Function.__call__ tapes self._grad, which calls `self.grad(*grads)`.
    # The shim maps execute->forward and save_for_backward/saved_tensors, but never
    # bridged backward->grad, so a torch-style custom Function (e.g. bloom's
    # GeLUFunction) raised "'GeLUFunction' object has no attribute 'grad'" in the
    # backward pass. Add a base `grad` that routes to a torch-style `backward` with
    # the instance as ctx. Gated on the base lacking its own grad; every native
    # jittor Function subclass (ACL ops, EMD, ...) defines grad(), which MRO-shadows
    # this, so they're untouched.
    # torch.autograd.Function defaults to materialize_grads=True; a Function may
    # opt out via ctx.set_materialize_grads(False). Store the flag on the ctx.
    if not hasattr(Fn, "set_materialize_grads"):
        def set_materialize_grads(self, value):
            self._materialize_grads = bool(value)
        Fn.set_materialize_grads = set_materialize_grads
    if "grad" not in getattr(Fn, "__dict__", {}):
        def grad(self, *grad_outputs):
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
        Fn.grad = grad


def _install_autograd(g, registry=None):
    """Expose torch.autograd.grad / torch.autograd.backward (jittor lacks the
    `torch.autograd` namespace functions; it only has jt.grad). These wrap
    jt.grad so `import jittor as torch; torch.autograd.grad(out, inputs)` works.
    """
    _modules = registry_for(g, registry).module_map
    import types as _types
    import jittor as _jt
    autograd = getattr(g, "autograd", None)
    if autograd is None or not isinstance(autograd, _types.ModuleType):
        autograd = _types.ModuleType("torch.autograd")
    # carry over the symbols other layers expect on torch.autograd
    if not hasattr(autograd, "Function"):
        autograd.Function = getattr(_jt, "Function", object)
    if not hasattr(autograd, "no_grad"):
        autograd.no_grad = getattr(g, "no_grad", _jt.no_grad)
    if not hasattr(autograd, "enable_grad"):
        autograd.enable_grad = getattr(g, "enable_grad", _jt.enable_grad)

    def _as_list(x):
        if isinstance(x, _jt.Var):
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
                          if isinstance(o, _jt.Var) and o.numel() != 1]
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
        gs = list(_jt.core.grad_optional(loss, ins, rg))
        missing = [i for i, value in enumerate(gs) if value is None]
        if missing and materialize_grads:
            for i in missing:
                gs[i] = _jt.zeros_like(ins[i])
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
                if isinstance(g, _jt.Var):
                    g = g.detach()
                    g.stop_grad()
                detached.append(g)
            gs = detached
        return tuple(gs)
    autograd.grad = grad

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
    autograd.backward = backward

    if not hasattr(autograd, "Variable"):
        autograd.Variable = g.Tensor
    # torch.autograd.set_detect_anomaly / detect_anomaly — debug hooks jittor
    # lacks; 3DGS train.py calls set_detect_anomaly(args.detect_anomaly) at start.
    import contextlib as _ctxlib
    autograd.set_detect_anomaly = lambda *a, **k: _ctxlib.nullcontext()
    autograd.detect_anomaly = lambda *a, **k: _ctxlib.nullcontext()
    g.autograd = autograd
    _modules["torch.autograd"] = autograd
    autograd.__path__ = getattr(autograd, "__path__", [])
    from jittor.autograd import functional

    _modules["torch.autograd.functional"] = functional
    autograd.functional = functional
    if "torch.autograd.profiler" not in _modules:
        _prof = _types.ModuleType("torch.autograd.profiler")
        class EventList(list):
            def table(self, *args, **kwargs):
                return ""
            def export_chrome_trace(self, *args, **kwargs):
                return None
        class _RecordFunction:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
        class profile(_RecordFunction):
            def function_events(self):
                return EventList()
            @property
            def key_averages(self):
                return lambda *args, **kwargs: EventList()
            def export_chrome_trace(self, *args, **kwargs):
                return None
        _prof.EventList = EventList
        _prof.record_function = lambda *args, **kwargs: _RecordFunction()
        _prof.profile = profile
        _prof.emit_nvtx = lambda *args, **kwargs: _RecordFunction()
        _prof.kineto_available = lambda: False
        _modules["torch.autograd.profiler"] = _prof
    autograd.profiler = _modules["torch.autograd.profiler"]


def install(ctx):
    g = ctx.jittor_module
    _install_autograd_function(g)
    _install_autograd(g, ctx.registry)


def install_tensordict(ctx):
    _install_tensordict_compat()


def install_parity(ctx):
    g = ctx.jittor_module
    registry = ctx.registry
    def module(name):
        return registry.ensure(name)
    autograd = g.autograd
    function = module("torch.autograd.function")
    function.Function = autograd.Function
    function.FunctionCtx = getattr(function, "FunctionCtx", type("FunctionCtx", (), {}))
    function.once_differentiable = getattr(
        function, "once_differentiable", lambda fn: fn
    )
    autograd.function = function
    autograd.once_differentiable = function.once_differentiable

    graph = module("torch.autograd.graph")
    if not hasattr(graph, "saved_tensors_hooks"):
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

        graph.saved_tensors_hooks = saved_tensors_hooks
        graph.save_on_cpu = save_on_cpu
        graph.Node = type("Node", (), {})
    autograd.graph = graph

    variable = module("torch.autograd.variable")
    if not hasattr(variable, "Variable"):
        engine = type(
            "_Engine", (), {"queue_callback": staticmethod(lambda *args, **kwargs: None)}
        )()
        variable.Variable = type("Variable", (), {"_execution_engine": engine})
    autograd.variable = variable
