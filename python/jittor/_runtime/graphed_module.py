"""Trainable layer-level differentiable CUDA Graphs (`jt.graphed_module`).

Wraps an `nn.Module` (or a sequence of modules sharing one private workspace
pool) into a differentiable paired Forward + Backward CUDA Graph via a custom
`jt.Function`.

Unlike whole-step capture (`jt.capture_step`), which requires the entire
training step including loss and optimizer to stay inside one captured graph
with no host readbacks, `jt.graphed_module` graphs individual modules or
backbone layers while preserving a standard Jittor autograd boundary:

  - Every differentiable activation input and every trainable parameter of the
    wrapped module is passed through `jt.Function`, so native `jt.grad`,
    `optimizer.step`, Torch `loss.backward()`, gradient accumulation across
    microbatches, `clip_grad_norm`, `loss.item()`, dynamic outer loss, AMP,
    and multi-layer chaining work automatically.
  - Forward intermediates stay pinned in the module's private `_GraphPool`
    (`keep_graph = 1`) at fixed addresses so the paired Backward CUDA Graph
    reads the exact saved activations without recomputing any forward op.
  - In-place non-gradient buffer updates during forward (such as `BatchNorm`
    `running_mean` / `running_var`) are captured via `_state_capture_begin`
    and replayed in-graph via `jt.write_back`.
  - CUDA dropout / random draws use `step_capture`'s live Philox RNG stream,
    advancing on every forward call while reusing the exact forward mask in
    the paired backward pass.
  - Overlapping forwards before backward (e.g. weight-tied recurrent calls),
    host readbacks, opaque barriers, non-rebuildable outputs, or unsupported
    CPU/mixed paths fall back cleanly to eager without corrupting saved state.
"""

from contextlib import nullcontext as _nullcontext
import threading
import weakref

import jittor_core as _core
from .._core.function import Function as _Function
from .._core.module import Module as _Module

_BOUND = False


def _bind_runtime():
    global _BOUND, jt, _graph_replay, _step_capture
    global GraphReplay, _RERECORD_LIMIT, _Unreplayable, _capture_barriers
    global _dense, _empty_like, _graph_has_nondeterministic_op, _input_vars
    global _map_inputs, _module_vars, _native_dtype, _no_auto, _object_ids
    global _output_template, _rebuild, _signature, _training, release_auto_replay
    global _GIVE_UP_AFTER, _GraphPool, _commit_captured_state, _pool_scope
    global _unique, _write_back_in_graph
    if _BOUND:
        return
    import jittor as _jt
    from . import graph_replay as _gr
    from . import step_capture as _sc
    jt = _jt
    _graph_replay = _gr
    _step_capture = _sc
    GraphReplay = _gr.GraphReplay
    _RERECORD_LIMIT = _gr._RERECORD_LIMIT
    _Unreplayable = _gr._Unreplayable
    _capture_barriers = _gr._capture_barriers
    _dense = _gr._dense
    _empty_like = _gr._empty_like
    _graph_has_nondeterministic_op = _gr._graph_has_nondeterministic_op
    _input_vars = _gr._input_vars
    _map_inputs = _gr._map_inputs
    _module_vars = _gr._module_vars
    _native_dtype = _gr._native_dtype
    _no_auto = _gr._no_auto
    _object_ids = _gr._object_ids
    _output_template = _gr._output_template
    _rebuild = _gr._rebuild
    _signature = _gr._signature
    _training = _gr._training
    release_auto_replay = _gr.release_auto_replay
    _GIVE_UP_AFTER = _sc._GIVE_UP_AFTER
    _GraphPool = _sc._GraphPool
    _commit_captured_state = _sc._commit_captured_state
    _pool_scope = _sc._pool_scope
    _unique = _sc._unique
    _write_back_in_graph = _sc._write_back_in_graph
    _BOUND = True


_LOCAL = threading.local()


def _capturing_depth():
    return getattr(_LOCAL, "depth", 0)


def _enter_capture_depth():
    _LOCAL.depth = _capturing_depth() + 1


def _exit_capture_depth():
    _LOCAL.depth = max(0, _capturing_depth() - 1)


def _is_cpu_only_var(var):
    return int(var.placement_backend) == 0


def _ready_aliases(outputs):
    aliases = []
    for o in outputs:
        try:
            aliases.append(_core._ready_step_gradient(o))
        except Exception:
            aliases.append(o)
    return aliases


def _snapshot_forward_outputs(cap, sources):
    """Copy materialized forward outputs into fresh Vars preserving requires_grad."""
    snap = _core._try_snapshot_cuda_step_output
    outs = []
    with jt.no_grad():
        for tpl, src in zip(cap.fwd_outputs, sources):
            out = snap(src)
            if out is None:
                out = _empty_like(tpl)
                out._copy_into(src, False)
            outs.append(out)
    for need_grad, out in zip(cap.fwd_out_req_grad, outs):
        if need_grad and out.is_stop_grad():
            out.start_grad()
    return tuple(outs)


def _stage_buffers(targets, sources):
    """Copy materialized `sources` into fixed-address `targets` on stream.

    Must never run `saved_storage_readers(targets)` when backward nodes have
    had their `_kept` bit cleared for an executor/recording pass — doing so
    would prematurely execute those backward nodes against the *previous*
    step's stale buffer contents and mark them `_kept = 1`.
    """
    if not targets:
        return
    with jt.no_grad():
        if _core._try_stage_cuda_step_inputs(targets, sources, False):
            return
        # Fallback for cleared-kept / host / non-contiguous inputs: materialize
        # contiguous sources outside `keep_graph`, then copy raw bytes via
        # `_copy_into_async` + `_copy_wait` (which never triggers
        # `saved_storage_readers`).
        prepared = []
        pending = []
        for target, source in zip(targets, sources):
            src = source
            if not src._storage_is_contiguous():
                with jt.flag_scope(keep_graph=0):
                    src = src.contiguous()
            if not src.is_finished:
                pending.append(src)
            prepared.append((target, src))
        if pending:
            with jt.flag_scope(keep_graph=0):
                jt.sync(pending, False, False)
        staged = []
        for target, src in prepared:
            if target.numel() == 0:
                continue
            _core._copy_into_async(target, src)
            staged.append(target)
        for target in staged:
            _core._copy_wait(target)


class _SeedGrad(_Function):
    """Inject static gradient buffers directly as `dout` of captured forward outputs."""

    def __init__(self, seeds):
        self.seeds = seeds

    def execute(self, *outputs):
        return jt.zeros((1,), "float32")

    def grad(self, _grad_dummy):
        return self.seeds


class _BackwardCapture:
    __slots__ = (
        "grad_mask",
        "static_grad_outs",
        "active_indices",
        "active_static_grad_outs",
        "target_slots",
        "outputs",
        "ready_outputs",
        "roots",
        "guards",
        "cuda_graph",
        "graph_refused",
        "replayed",
        "record_at",
        "rerecords",
    )

    def __init__(self, grad_mask):
        self.grad_mask = grad_mask
        self.static_grad_outs = []
        self.active_indices = []
        self.active_static_grad_outs = []
        self.target_slots = []
        self.outputs = []
        self.ready_outputs = None
        self.roots = []
        self.guards = []
        self.cuda_graph = 0
        self.graph_refused = None
        self.replayed = 0
        self.record_at = 1
        self.rerecords = 0


class _ForwardCapture:
    """State of one captured training signature.

    Carries the same protocol attributes as `step_capture._Capture` so that
    `step_capture.active()`, live Philox RNG (`random_draw` / `random_keep`),
    `live_array`, `guard`, `on_replay`, and `refuse` cooperate seamlessly.
    """

    __slots__ = (
        "prologue",
        "guards",
        "refused",
        "random",
        "exact",
        "published_gradients",
        "inputs",
        "host_inputs",
        "fwd_outputs",
        "fwd_out_req_grad",
        "fwd_ready_outputs",
        "template",
        "fwd_roots",
        "state",
        "param_holders",
        "param_olds",
        "trainable_params",
        "trainable_param_olds",
        "n_inputs",
        "bwd_targets",
        "bwd_target_slots",
        "signature",
        "fwd_cuda_graph",
        "fwd_graph_refused",
        "fwd_replayed",
        "fwd_record_at",
        "fwd_rerecords",
        "bwd_caps",
        "active_ctx_ref",
        "call_seq",
        "invalidated",
    )

    def __init__(self):
        self.prologue = []
        self.guards = []
        self.refused = None
        self.random = None
        self.exact = False
        self.published_gradients = []
        self.inputs = []
        self.host_inputs = False
        self.fwd_outputs = []
        self.fwd_out_req_grad = ()
        self.fwd_ready_outputs = None
        self.template = None
        self.fwd_roots = []
        self.state = []
        self.param_holders = []
        self.param_olds = []
        self.trainable_params = []
        self.trainable_param_olds = []
        self.n_inputs = 0
        self.bwd_targets = []
        self.bwd_target_slots = []
        self.signature = None
        self.fwd_cuda_graph = 0
        self.fwd_graph_refused = None
        self.fwd_replayed = 0
        self.fwd_record_at = 1
        self.fwd_rerecords = 0
        self.bwd_caps = {}
        self.active_ctx_ref = None
        self.call_seq = 0
        self.invalidated = False


class _GraphedApply(_Function):
    """Differentiable autograd bridge for a `GraphedModule` forward/backward pair."""

    def __init__(self, owner, cap, call_inputs=None, precomputed_outputs=None):
        self.owner = owner
        self.cap = cap
        self.call_inputs = call_inputs
        self.precomputed_outputs = precomputed_outputs
        self.backward_ran = False
        self.ran_fwd_executor = False
        self.bwd_ready_for_record = False
        self.call_id = 0

    def execute(self, *flat_vars):
        cap = self.cap
        cap.call_seq += 1
        self.call_id = cap.call_seq
        cap.active_ctx_ref = weakref.ref(self)
        call_inputs = self.call_inputs
        self.call_inputs = None
        if self.precomputed_outputs is not None:
            outs = self.precomputed_outputs
            self.precomputed_outputs = None
            self.ran_fwd_executor = True
            return outs
        input_vars = call_inputs if call_inputs is not None else flat_vars[:cap.n_inputs]
        return self.owner._execute_forward(self, cap, input_vars)

    def grad(self, *grad_outputs):
        return self.owner._execute_backward(self, self.cap, grad_outputs)


class GraphedModule(_Module):
    """Differentiable CUDA Graph wrapper for an `nn.Module`. See module docstring."""

    def __init__(self, module, *, warmup=2, pool=None, record=True):
        _bind_runtime()
        if not callable(module):
            raise TypeError("graphed_module expects a callable nn.Module")
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        super().__init__()
        if isinstance(module, jt.Module):
            release_auto_replay(module)
        object.__setattr__(self, "_wrapped_module", module)
        object.__setattr__(self, "_warmup", int(warmup))
        object.__setattr__(self, "_external_pool", pool)
        object.__setattr__(self, "_pool", pool)
        object.__setattr__(self, "_record", bool(record))
        object.__setattr__(self, "_capture", None)
        object.__setattr__(self, "_eval_replay", None)
        object.__setattr__(self, "_refused", None)
        object.__setattr__(self, "_seen_sig", None)
        object.__setattr__(self, "_warmup_count", 0)
        object.__setattr__(self, "_recaptures", 0)
        object.__setattr__(self, "last_invalidated", None)
        object.__setattr__(
            self,
            "stats",
            {
                "eager": 0,
                "captured": 0,
                "fwd_replayed": 0,
                "bwd_replayed": 0,
                "fwd_graph": 0,
                "bwd_graph": 0,
                "eval_replayed": 0,
                "eval_graph": 0,
            },
        )
        from jittor.profiling import register_replay_source
        register_replay_source(self)

    def __del__(self):
        try:
            self.invalidate()
        except Exception:
            pass

    @property
    def module(self):
        return self._wrapped_module

    @property
    def refused(self):
        """Why training graph capture was refused, or None."""
        return self._refused

    @property
    def graph_refused(self):
        """Why device graph recording was refused for the active capture, or None."""
        cap = self._capture
        if cap is None:
            return None
        if cap.fwd_graph_refused is not None:
            return cap.fwd_graph_refused
        for bwd in cap.bwd_caps.values():
            if bwd.graph_refused is not None:
                return bwd.graph_refused
        return None

    def is_training(self):
        return _training(self._wrapped_module)

    @property
    def training(self):
        return _training(self._wrapped_module)

    def dfs(self, parents, k, callback, callback_leave=None, recurse=True):
        wrapped = self._wrapped_module
        if isinstance(wrapped, jt.Module):
            return wrapped.dfs(parents, k, callback, callback_leave, recurse)

    def __getattr__(self, name):
        wrapped = object.__getattribute__(self, "_wrapped_module")
        return getattr(wrapped, name)

    # -- lifecycle / cleanup -------------------------------------------
    def _drop_cuda_graphs(self, cap, *, record_next=False):
        if cap.fwd_cuda_graph:
            _core.graph_release(cap.fwd_cuda_graph)
            cap.fwd_cuda_graph = 0
        cap.fwd_ready_outputs = None
        for bwd in cap.bwd_caps.values():
            if bwd.cuda_graph:
                _core.graph_release(bwd.cuda_graph)
                bwd.cuda_graph = 0
            bwd.ready_outputs = None
            if record_next:
                bwd.rerecords += 1
                if bwd.rerecords >= _RERECORD_LIMIT:
                    bwd.graph_refused = "the leaves a recording reads kept moving"
                else:
                    bwd.record_at = bwd.replayed + 1
        if record_next:
            cap.fwd_rerecords += 1
            if cap.fwd_rerecords >= _RERECORD_LIMIT:
                cap.fwd_graph_refused = "the leaves a recording reads kept moving"
            else:
                cap.fwd_record_at = cap.fwd_replayed + 1

    def invalidate(self):
        """Release all captured forward/backward graphs and reset state."""
        cap = self._capture
        object.__setattr__(self, "_capture", None)
        if cap is not None:
            cap.invalidated = True
            self._drop_cuda_graphs(cap)
            for bwd in cap.bwd_caps.values():
                for g in bwd.guards:
                    if not g.is_finished:
                        g._release_kept()
                for root in bwd.roots:
                    if not root.is_finished:
                        root._release_kept()
            for root in cap.fwd_roots:
                if not root.is_finished:
                    root._release_kept()
            cap.bwd_caps.clear()
        eval_replay = self._eval_replay
        object.__setattr__(self, "_eval_replay", None)
        if eval_replay is not None:
            eval_replay.invalidate()
        if self._external_pool is None:
            object.__setattr__(self, "_pool", None)

    def reset(self):
        """Release captures and clear any prior refusal so capture can retry."""
        self.invalidate()
        object.__setattr__(self, "_refused", None)
        object.__setattr__(self, "_seen_sig", None)
        object.__setattr__(self, "_warmup_count", 0)
        object.__setattr__(self, "_recaptures", 0)
        object.__setattr__(self, "last_invalidated", None)

    # -- helpers -------------------------------------------------------
    def _ensure_pool(self):
        if (
            self._pool is None
            and self._record
            and jt.flags.use_cuda
            and _core.graph_capture_supported()
        ):
            object.__setattr__(self, "_pool", _GraphPool())
        return self._pool

    def _signature_for(self, args, kwargs, input_vars, module_vars):
        return (
            _signature(args, kwargs),
            tuple(bool(v.requires_grad) for v in input_vars),
            tuple(bool(v.requires_grad) for v in module_vars if isinstance(v, jt.Var)),
            _training(self._wrapped_module),
            int(jt.flags.amp_reg),
            int(jt.flags.cuda_allow_tf32),
        )

    def _stale(self, cap, sig):
        if _core._first_finished(cap.fwd_roots) >= 0:
            return "the captured forward graph was finished"
        for bwd in cap.bwd_caps.values():
            if _core._first_finished(bwd.roots) >= 0:
                return "the captured backward graph was finished"
        if sig != cap.signature:
            return "inputs, gradient policy, training mode, or math flags changed"
        for read, value in cap.guards:
            if read() != value:
                return "a guarded value baked into the module changed"
        # Check for parameter/buffer rebinding (e.g. optimizer update or load_state_dict).
        # Fused in-place optimizers like fused_adamw rebind the VarHolder to an
        # output alias sharing the exact same device memory (`_mem_ptr_now`), so
        # batching pending syncs and skipping redundant `_copy_into` avoids both
        # extra DtoD copies and per-parameter flag_scope overhead.
        holders, olds = cap.param_holders, cap.param_olds
        rebound_pairs = []
        start = 0
        n_holders = len(holders)
        while start < n_holders:
            rebound = _core._first_rebound(holders[start:], olds[start:])
            if rebound < 0:
                break
            idx = start + rebound
            holder, old = holders[idx], olds[idx]
            if (
                tuple(holder.shape) != tuple(old.shape)
                or str(_native_dtype(holder)) != str(_native_dtype(old))
            ):
                return "a module parameter or buffer changed shape or dtype"
            rebound_pairs.append((holder, old))
            start = idx + 1
        if rebound_pairs:
            pending = [h for h, _ in rebound_pairs if not h.is_finished]
            if pending:
                with jt.flag_scope(keep_graph=0):
                    jt.sync(pending, False, False)
            before = jt.flags.keep_graph
            jt.flags.keep_graph = 1
            try:
                with jt.no_grad():
                    for holder, old in rebound_pairs:
                        if holder._mem_ptr_now != old._mem_ptr_now:
                            old._copy_into(holder, False)
                        holder._update(old)
            finally:
                jt.flags.keep_graph = before
        return None

    def _clear_bwd_kept(self, cap):
        for bwd in cap.bwd_caps.values():
            for root in bwd.roots:
                if not root.is_finished:
                    root._release_kept()

    def _write_back(self, cap):
        with jt.no_grad():
            for _, old, new, inplace in cap.state:
                if not inplace:
                    old._copy_into(new, False)

    def _bwd_all_graphable(self, cap):
        for bwd in cap.bwd_caps.values():
            if bwd.graph_refused is not None:
                return False
        return True

    def _can_launch_fwd_graph(self, cap):
        if not cap.fwd_cuda_graph:
            return False
        if not self._bwd_all_graphable(cap):
            return False
        for bwd in cap.bwd_caps.values():
            if not bwd.cuda_graph:
                # Backward not yet recorded: keep forward on executor so backward
                # sees cleanly separated `_kept` marks when it records.
                return False
        return True

    # -- eager / eval paths --------------------------------------------
    def _run_eager(self, args, kwargs, *, probe_barriers=False):
        self.stats["eager"] += 1
        if not probe_barriers:
            with _no_auto():
                return self._wrapped_module(*args, **kwargs)
        readbacks = _core._host_readback_count()
        with _capture_barriers() as barriers, _no_auto():
            result = self._wrapped_module(*args, **kwargs)
        if barriers:
            object.__setattr__(
                self, "_refused",
                "the warm-up contains an opaque operation: " + barriers[0],
            )
        elif _core._host_readback_count() != readbacks:
            object.__setattr__(
                self, "_refused",
                "the warm-up read a value back to the host, so its path depends "
                "on tensor values and cannot be graphed",
            )
        return result

    def _run_eval(self, args, kwargs):
        if not self._record or not jt.flags.use_cuda or not _core.graph_capture_supported():
            self.stats["eager"] += 1
            with _no_auto():
                return self._wrapped_module(*args, **kwargs)
        replay = self._eval_replay
        if replay is None:
            replay = GraphReplay(self._wrapped_module, measure=False, weak=False)
            object.__setattr__(self, "_eval_replay", replay)
        before_replayed = replay.stats["replayed"]
        before_graph = replay.stats["graph"]
        out = replay(*args, **kwargs)
        self.stats["eval_replayed"] += replay.stats["replayed"] - before_replayed
        self.stats["eval_graph"] += replay.stats["graph"] - before_graph
        return out

    # -- forward capture -----------------------------------------------
    def _capture_forward(self, args, kwargs, input_vars, module_vars, sig):
        # Materialize all inputs and module leaves before capture so they are
        # finished leaves of the kept graph.
        to_sync = [v for v in input_vars + module_vars if isinstance(v, jt.Var) and not v.is_finished]
        if to_sync:
            with jt.flag_scope(keep_graph=0):
                jt.sync(to_sync, False, False)

        def make_static(value):
            buf = _empty_like(value)
            with jt.no_grad():
                buf._copy_into(value, False)
            return buf

        static_args, static_kwargs = _map_inputs((args, kwargs), make_static)
        static_inputs = _input_vars((static_args, static_kwargs), [])

        param_holders = [v for v in module_vars if isinstance(v, jt.Var)]
        param_olds = [_core._snapshot_step_gradient(v) for v in param_holders]
        trainable_indices = [i for i, v in enumerate(param_holders) if bool(v.requires_grad)]
        trainable_ids = {id(param_holders[i]) for i in trainable_indices}

        cap = _ForwardCapture()
        cap.inputs = static_inputs
        cap.host_inputs = any(int(v.device_id) < 0 for v in static_inputs)
        cap.param_holders = param_holders
        cap.param_olds = param_olds
        cap.trainable_params = [param_holders[i] for i in trainable_indices]
        cap.trainable_param_olds = [param_olds[i] for i in trainable_indices]
        cap.n_inputs = len(static_inputs)
        cap.signature = sig

        bwd_targets = []
        bwd_target_slots = []
        for i, v in enumerate(static_inputs):
            if bool(v.requires_grad):
                bwd_targets.append(v)
                bwd_target_slots.append(i)
        for j, old_p in enumerate(cap.trainable_param_olds):
            bwd_targets.append(old_p)
            bwd_target_slots.append(cap.n_inputs + j)
        cap.bwd_targets = bwd_targets
        cap.bwd_target_slots = bwd_target_slots

        prev_active = _step_capture._ACTIVE
        before_keep = jt.flags.keep_graph
        before_flush = jt.flags.auto_flush_ops
        before_async = jt.flags.async_flush_ops
        readbacks = _core._host_readback_count()
        records = []
        raw_out = None
        fwd_outputs = []
        template = None
        _step_capture._ACTIVE = cap
        _graph_replay._TRACING[0] += 1
        _enter_capture_depth()
        _core._state_capture_begin()
        jt.flags.keep_graph = 1
        jt.flags.auto_flush_ops = 0
        jt.flags.async_flush_ops = 0
        try:
            with _capture_barriers() as barriers, _no_auto():
                raw_out = self._wrapped_module(*static_args, **static_kwargs)
                try:
                    template = _output_template(raw_out, fwd_outputs, {}, _object_ids((args, kwargs)))
                except _Unreplayable as exc:
                    cap.refused = cap.refused or str(exc)
                if barriers:
                    cap.refused = cap.refused or (
                        "the module contains an opaque operation: " + barriers[0]
                    )
        except Exception:
            try:
                records = _core._state_capture_end()
            finally:
                jt.flags.async_flush_ops = before_async
                jt.flags.auto_flush_ops = before_flush
                jt.flags.keep_graph = before_keep
                _exit_capture_depth()
                _graph_replay._TRACING[0] -= 1
                _step_capture._ACTIVE = prev_active
            with jt.no_grad():
                for holder, old, new in records:
                    if not new.is_finished:
                        new._release_kept()
                    holder._update(old)
            raise
        else:
            try:
                records = _core._state_capture_end()
            finally:
                jt.flags.async_flush_ops = before_async
                jt.flags.auto_flush_ops = before_flush
                jt.flags.keep_graph = before_keep
                _exit_capture_depth()
                _graph_replay._TRACING[0] -= 1
                _step_capture._ACTIVE = prev_active

        if _core._host_readback_count() != readbacks:
            cap.refused = cap.refused or (
                "the module read a value back to the host, so its path depends "
                "on tensor values and cannot be graphed"
            )

        if cap.refused is None and not fwd_outputs:
            cap.refused = "the module returned no Var"

        if cap.refused is None:
            for holder, _, _ in records:
                if id(holder) in trainable_ids:
                    cap.refused = "the module mutated a trainable parameter in place during forward"
                    break

        if cap.refused is None and _graph_has_nondeterministic_op():
            cap.refused = "the module draws from an uncapturable host random generator"

        if cap.refused is not None:
            # Undo any speculative state rebinds and release pending traced nodes,
            # then run the call eagerly once on the caller's real inputs.
            with jt.no_grad():
                for holder, old, new in records:
                    if not new.is_finished:
                        new._release_kept()
                    holder._update(old)
                for o in fwd_outputs:
                    if not o.is_finished:
                        o._release_kept()
            object.__setattr__(self, "_refused", cap.refused)
            return None, self._run_eager(args, kwargs)

        # Densify outputs & state writes inside the kept forward graph.
        jt.flags.keep_graph = 1
        try:
            fwd_outputs[:] = [_dense(o) for o in fwd_outputs]
            records = [
                (h, o, n if n._same_step_storage(o) else _dense(n))
                for h, o, n in records
            ]
            fwd_roots = _unique(fwd_outputs + [n for _, _, n in records])
        finally:
            jt.flags.keep_graph = before_keep

        # Execute forward ONCE under keep_graph=1 inside the private graph pool
        # so every saved activation stays pinned at its fixed address.
        pool = self._ensure_pool()
        jt.flags.keep_graph = 1
        try:
            with _pool_scope(pool, allocate=True):
                jt.sync(fwd_roots, False, False)
            state = _commit_captured_state(records)
        finally:
            jt.flags.keep_graph = before_keep
        with jt.no_grad():
            state = _write_back_in_graph(state, fwd_roots)

        cap.fwd_outputs = fwd_outputs
        cap.fwd_out_req_grad = tuple(bool(o.requires_grad) for o in fwd_outputs)
        cap.fwd_ready_outputs = _ready_aliases(fwd_outputs)
        cap.template = template
        cap.fwd_roots = fwd_roots
        cap.state = state
        if not self._record or not _core.graph_capture_supported():
            cap.fwd_graph_refused = "recording disabled or unsupported"

        out_copies = _snapshot_forward_outputs(cap, cap.fwd_ready_outputs)
        cap.fwd_replayed = 1
        object.__setattr__(self, "_capture", cap)
        self.stats["captured"] += 1
        self.stats["fwd_replayed"] += 1
        return cap, out_copies

    # -- forward execution (called inside _GraphedApply.execute) -------
    def _record_fwd_cuda_graph(self, cap):
        if not self._record or not _core.graph_capture_supported():
            cap.fwd_graph_refused = "recording disabled or unsupported"
            return False
        host = _core.graph_host_work(cap.fwd_roots)
        if host:
            cap.fwd_graph_refused = "part of forward runs on host (%s)" % host
            return False
        pool = self._ensure_pool()
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 1
        handle = 0
        try:
            _core.graph_wait()
            with _pool_scope(pool, allocate=True):
                began = _core.graph_capture_begin()
                if not began:
                    cap.fwd_graph_refused = "device refused to start forward recording"
                    return False
                try:
                    jt.sync(cap.fwd_roots, False, False)
                    self._write_back(cap)
                finally:
                    handle = _core.graph_capture_end()
        except Exception as exc:
            cap.fwd_graph_refused = "forward recording raised %s: %s" % (type(exc).__name__, exc)
            if handle:
                _core.graph_release(handle)
            return False
        finally:
            jt.flags.keep_graph = before
        if not handle:
            refusal = _core.graph_capture_refusal()
            if refusal.startswith("retryable: "):
                cap.fwd_rerecords += 1
                if cap.fwd_rerecords < _RERECORD_LIMIT:
                    cap.fwd_record_at = cap.fwd_replayed + 1
                    return False
                refusal = refusal[len("retryable: "):]
            cap.fwd_graph_refused = refusal or "forward recording contained no device work"
            return False
        cap.fwd_cuda_graph = handle
        cap.fwd_ready_outputs = _ready_aliases(cap.fwd_outputs)
        _core.graph_bind_leaves(handle, cap.fwd_roots)
        return True

    def _execute_forward(self, ctx, cap, input_vars):
        if cap.host_inputs and cap.fwd_cuda_graph:
            _core.graph_wait()
        # Materialize any pending inputs BEFORE entering keep_graph so outer
        # producers are never kept.
        pending = [v for v in input_vars if not v.is_finished]
        if pending:
            with jt.flag_scope(keep_graph=0):
                jt.sync(pending, False, False)
        for hook in cap.prologue:
            hook()
        _stage_buffers(cap.inputs, input_vars)

        pool = self._ensure_pool()

        # Check if existing CUDA graph leaves moved (e.g. param migrated to CPU).
        if cap.fwd_cuda_graph and _core.graph_leaves_moved(cap.fwd_cuda_graph):
            self._drop_cuda_graphs(cap, record_next=True)

        # Steady-state CUDA graph launch when both forward and backward graphs are ready.
        if self._can_launch_fwd_graph(cap):
            with _pool_scope(pool):
                _core.graph_launch(cap.fwd_cuda_graph)
            outs = _snapshot_forward_outputs(cap, cap.fwd_ready_outputs)
            cap.fwd_replayed += 1
            self.stats["fwd_replayed"] += 1
            self.stats["fwd_graph"] += 1
            return outs

        # Clear `_kept` on backward roots BEFORE running/recording forward so
        # forward re-marks all forward nodes `_kept=1` while backward nodes stay `_kept=0`.
        self._clear_bwd_kept(cap)

        # Record forward CUDA graph if eligible (after warmup executor replay).
        should_record = (
            not cap.fwd_cuda_graph
            and cap.fwd_graph_refused is None
            and self._record
            and self._bwd_all_graphable(cap)
            and cap.fwd_replayed >= cap.fwd_record_at
        )
        if should_record and self._record_fwd_cuda_graph(cap):
            with _pool_scope(pool):
                _core.graph_launch(cap.fwd_cuda_graph)
            ctx.bwd_ready_for_record = True
            outs = _snapshot_forward_outputs(cap, cap.fwd_ready_outputs)
            cap.fwd_replayed += 1
            self.stats["fwd_replayed"] += 1
            self.stats["fwd_graph"] += 1
            return outs

        # Executor replay path.
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 1
        try:
            with _pool_scope(pool, allocate=True):
                jt.sync(cap.fwd_roots, False, False)
                self._write_back(cap)
        finally:
            jt.flags.keep_graph = before
        ctx.ran_fwd_executor = True
        cap.fwd_ready_outputs = _ready_aliases(cap.fwd_outputs)
        outs = _snapshot_forward_outputs(cap, cap.fwd_ready_outputs)
        cap.fwd_replayed += 1
        self.stats["fwd_replayed"] += 1
        return outs

    # -- backward capture / recording / launch -------------------------
    def _build_backward(self, cap, grad_mask, active_douts):
        bwd = _BackwardCapture(grad_mask)
        static_grad_outs = []
        active_indices = []
        active_static = []
        dout_iter = iter(active_douts)
        for i, (is_active, fwd_out) in enumerate(zip(grad_mask, cap.fwd_outputs)):
            if is_active:
                g = next(dout_iter)
                buf = _empty_like(fwd_out)
                buf.stop_grad()
                with jt.no_grad(), jt.flag_scope(keep_graph=1):
                    buf._copy_into(g, False)
                static_grad_outs.append(buf)
                active_indices.append(i)
                active_static.append(buf)
            else:
                static_grad_outs.append(None)
        bwd.static_grad_outs = static_grad_outs
        bwd.active_indices = active_indices
        bwd.active_static_grad_outs = active_static

        before_keep = jt.flags.keep_graph
        before_flush = jt.flags.auto_flush_ops
        before_async = jt.flags.async_flush_ops
        jt.flags.keep_graph = 1
        jt.flags.auto_flush_ops = 0
        jt.flags.async_flush_ops = 0
        try:
            with _no_auto():
                dummy = _SeedGrad(tuple(static_grad_outs))(*cap.fwd_outputs)
                raw_grads = jt.grad_optional(dummy, cap.bwd_targets, True)
                target_slots = []
                outputs = []
                for slot, g in zip(cap.bwd_target_slots, raw_grads):
                    if g is not None:
                        outputs.append(_dense(g))
                        target_slots.append(slot)
        finally:
            jt.flags.async_flush_ops = before_async
            jt.flags.auto_flush_ops = before_flush
            jt.flags.keep_graph = before_keep

        bwd.target_slots = target_slots
        bwd.outputs = outputs
        bwd.roots = _unique(outputs)
        if not self._record or not _core.graph_capture_supported():
            bwd.graph_refused = "recording disabled or unsupported"

        if bwd.roots:
            pool = self._ensure_pool()
            jt.flags.keep_graph = 1
            try:
                with _pool_scope(pool, allocate=True):
                    _core._materialize_step_readouts(bwd.roots, False)
            finally:
                jt.flags.keep_graph = before_keep
            # Shield backward roots from weak sync / sync_all / auto_flush even
            # while their `_kept` flag is cleared between forward and backward:
            # attaching a kept TapeOp consumer makes every root a non-sink, and
            # the kept TapeOp itself is skipped by all bystander sweepers.
            guards = []
            for root in bwd.roots:
                if not root.is_finished:
                    guard_var = root.tape()
                    guard_var._mark_kept()
                    guards.append(guard_var)
            bwd.guards = guards
        bwd.ready_outputs = _ready_aliases(bwd.outputs)

        # A newly attached backward mask may consume forward intermediates that
        # were previously fused away when `fwd_cuda_graph` was recorded. Drop
        # the forward recording so the next step re-records with all backward
        # consumers visible to the fuser.
        if cap.fwd_cuda_graph:
            _core.graph_release(cap.fwd_cuda_graph)
            cap.fwd_cuda_graph = 0
            cap.fwd_ready_outputs = None
            cap.fwd_record_at = cap.fwd_replayed

        cap.bwd_caps[grad_mask] = bwd
        return bwd

    def _record_bwd_cuda_graph(self, cap, bwd):
        if not self._record or not _core.graph_capture_supported():
            bwd.graph_refused = "recording disabled or unsupported"
            return False
        if not bwd.roots:
            bwd.graph_refused = "backward has no outputs"
            return False
        host = _core.graph_host_work(bwd.roots)
        if host:
            bwd.graph_refused = "part of backward runs on host (%s)" % host
            return False
        pool = self._ensure_pool()
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 1
        handle = 0
        try:
            _core.graph_wait()
            with _pool_scope(pool, allocate=True):
                began = _core.graph_capture_begin()
                if not began:
                    bwd.graph_refused = "device refused to start backward recording"
                    return False
                try:
                    _core._materialize_step_readouts(bwd.roots, False)
                finally:
                    handle = _core.graph_capture_end()
        except Exception as exc:
            bwd.graph_refused = "backward recording raised %s: %s" % (type(exc).__name__, exc)
            if handle:
                _core.graph_release(handle)
            return False
        finally:
            jt.flags.keep_graph = before
        if not handle:
            refusal = _core.graph_capture_refusal()
            if refusal.startswith("retryable: "):
                bwd.rerecords += 1
                if bwd.rerecords < _RERECORD_LIMIT:
                    bwd.record_at = bwd.replayed + 1
                    return False
                refusal = refusal[len("retryable: "):]
            bwd.graph_refused = refusal or "backward recording contained no device work"
            return False
        bwd.cuda_graph = handle
        bwd.ready_outputs = _ready_aliases(bwd.outputs)
        _core.graph_bind_leaves(handle, bwd.roots)
        return True

    def _assemble_grads(self, cap, bwd, sources):
        total = cap.n_inputs + len(cap.trainable_params)
        result = [None] * total
        snap = _core._try_snapshot_cuda_step_output
        with jt.no_grad():
            for slot, src in zip(bwd.target_slots, sources):
                out = snap(src)
                if out is None:
                    out = _empty_like(src)
                    out.stop_grad()
                    out._copy_into(src, False)
                result[slot] = out
        return tuple(result)

    def _execute_backward(self, ctx, cap, grad_outputs):
        if cap.invalidated:
            raise RuntimeError(
                "graphed_module: backward called after the captured graph was "
                "invalidated or reset"
            )
        if ctx.call_id != cap.call_seq:
            raise RuntimeError(
                "graphed_module: saved activations for an earlier forward were "
                "overwritten by a subsequent forward call before repeated backward"
            )
        is_repeat_backward = ctx.backward_ran
        ctx.backward_ran = True
        total_slots = cap.n_inputs + len(cap.trainable_params)
        grad_mask = tuple(g is not None for g in grad_outputs)
        if not any(grad_mask):
            return (None,) * total_slots

        # Materialize & dtype-align incoming grad_outputs OUTSIDE keep_graph
        # so outer loss backward ops finish normally and are never kept.
        prepared = []
        pending = []
        for g, fwd_out in zip(grad_outputs, cap.fwd_outputs):
            if g is None:
                continue
            target_dtype = str(_native_dtype(fwd_out))
            if str(_native_dtype(g)) != target_dtype:
                with jt.no_grad(), jt.flag_scope(keep_graph=0):
                    g = g.cast(target_dtype)
            if not g.is_finished:
                pending.append(g)
            prepared.append(g)
        if pending:
            with jt.flag_scope(keep_graph=0):
                jt.sync(pending, False, False)

        bwd = cap.bwd_caps.get(grad_mask)
        if bwd is None:
            bwd = self._build_backward(cap, grad_mask, prepared)
            bwd.replayed += 1
            self.stats["bwd_replayed"] += 1
            return self._assemble_grads(cap, bwd, bwd.ready_outputs)

        _stage_buffers(bwd.active_static_grad_outs, prepared)

        pool = self._ensure_pool()

        if bwd.cuda_graph and _core.graph_leaves_moved(bwd.cuda_graph):
            self._drop_cuda_graphs(cap, record_next=True)

        if bwd.cuda_graph:
            with _pool_scope(pool):
                _core.graph_launch(bwd.cuda_graph)
            bwd.replayed += 1
            self.stats["bwd_replayed"] += 1
            self.stats["bwd_graph"] += 1
            return self._assemble_grads(cap, bwd, bwd.ready_outputs)

        should_record = (
            not is_repeat_backward
            and bwd.graph_refused is None
            and self._record
            and cap.fwd_graph_refused is None
            and bwd.replayed >= bwd.record_at
            and (ctx.ran_fwd_executor or ctx.bwd_ready_for_record)
        )
        if should_record:
            if self._record_bwd_cuda_graph(cap, bwd):
                with _pool_scope(pool):
                    _core.graph_launch(bwd.cuda_graph)
                bwd.replayed += 1
                self.stats["bwd_replayed"] += 1
                self.stats["bwd_graph"] += 1
                return self._assemble_grads(cap, bwd, bwd.ready_outputs)
            # Recording failed: `bwd` nodes were marked `_kept=1` during the
            # failed capture without executing GPU kernels. Clear `_kept` and
            # run through executor to produce exact gradients for this call.
            if bwd.graph_refused is not None and cap.fwd_cuda_graph:
                self._drop_cuda_graphs(cap)
            before = jt.flags.keep_graph
            jt.flags.keep_graph = 1
            try:
                for root in bwd.roots:
                    if not root.is_finished:
                        root._release_kept()
                with _pool_scope(pool, allocate=True):
                    jt.sync(bwd.roots, False, False)
            finally:
                jt.flags.keep_graph = before
            bwd.ready_outputs = _ready_aliases(bwd.outputs)
            bwd.replayed += 1
            self.stats["bwd_replayed"] += 1
            return self._assemble_grads(cap, bwd, bwd.ready_outputs)

        # Executor replay path.
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 1
        try:
            with _pool_scope(pool, allocate=True):
                if not is_repeat_backward and (ctx.ran_fwd_executor or ctx.bwd_ready_for_record):
                    _core._materialize_step_readouts(bwd.roots, False)
                else:
                    # E.g. retain_graph=True repeated backward or dropped graph:
                    # run deterministically from the kept graph.
                    for root in bwd.roots:
                        if not root.is_finished:
                            root._release_kept()
                    jt.sync(bwd.roots, False, False)
        finally:
            jt.flags.keep_graph = before
        bwd.ready_outputs = _ready_aliases(bwd.outputs)
        bwd.replayed += 1
        self.stats["bwd_replayed"] += 1
        return self._assemble_grads(cap, bwd, bwd.ready_outputs)

    # -- main call entry -----------------------------------------------
    def __call__(self, *args, **kwargs):
        # Never nest inside another active capture (e.g. outer capture_step or
        # outer graphed_module).
        if (
            _core.graph_capture_active()
            or _step_capture.active()
            or _capturing_depth() > 0
        ):
            return self._run_eager(args, kwargs)

        # Without CUDA (or with explicit CPU inputs), run eagerly without
        # permanently refusing in case CUDA is enabled later.
        if not jt.flags.use_cuda:
            return self._run_eager(args, kwargs)

        input_vars = _input_vars((args, kwargs), [])
        if any(_is_cpu_only_var(v) for v in input_vars):
            return self._run_eager(args, kwargs)

        # Inference / no_grad path: delegate to forward-only GraphReplay.
        if jt.flags.no_grad:
            return self._run_eval(args, kwargs)

        cap = self._capture
        module_vars = cap.param_holders if cap is not None else _module_vars(self._wrapped_module)
        has_grad = any(bool(v.requires_grad) for v in input_vars) or any(
            isinstance(v, jt.Var) and bool(v.requires_grad) for v in module_vars
        )
        if not has_grad:
            return self._run_eval(args, kwargs)

        if self._refused is not None:
            return self._run_eager(args, kwargs)

        # Guard against overlapping forwards (e.g. weight-tied recurrent calls)
        # whose previous forward still owns the static saved activations.
        if cap is not None:
            active = cap.active_ctx_ref() if cap.active_ctx_ref is not None else None
            if active is not None and not active.backward_ran:
                return self._run_eager(args, kwargs)

        sig = self._signature_for(args, kwargs, input_vars, module_vars)

        if cap is not None:
            reason = self._stale(cap, sig)
            if reason is not None:
                object.__setattr__(self, "last_invalidated", reason)
                self.invalidate()
                cap = None
                recaptures = self._recaptures + 1
                object.__setattr__(self, "_recaptures", recaptures)
                if recaptures >= _GIVE_UP_AFTER:
                    object.__setattr__(
                        self, "_refused",
                        "inputs or parameters kept changing; falling back to eager",
                    )
                    return self._run_eager(args, kwargs)
                module_vars = _module_vars(self._wrapped_module)
                sig = self._signature_for(args, kwargs, input_vars, module_vars)

        if cap is None:
            if sig != self._seen_sig:
                object.__setattr__(self, "_seen_sig", sig)
                object.__setattr__(self, "_warmup_count", 0)
            if self._warmup_count < self._warmup:
                object.__setattr__(self, "_warmup_count", self._warmup_count + 1)
                return self._run_eager(args, kwargs, probe_barriers=True)
            cap, result_or_outs = self._capture_forward(
                args, kwargs, input_vars, module_vars, sig
            )
            if cap is None:
                return result_or_outs
            fn = _GraphedApply(self, cap, precomputed_outputs=result_or_outs)
            taped = fn(*(input_vars + cap.trainable_params))
            return _rebuild(cap.template, list(taped))

        fn = _GraphedApply(self, cap, call_inputs=input_vars)
        taped = fn(*(input_vars + cap.trainable_params))
        return _rebuild(cap.template, list(taped))


def graphed_module(module, *, warmup=2, pool=None, record=True):
    """Wrap an `nn.Module` (or sequence of modules) in differentiable CUDA Graphs.

    When passed a `list` or `tuple` of modules, all returned `GraphedModule`
    wrappers share one private `_GraphPool` so their temporary workspaces reuse
    the same device arena.
    """
    _bind_runtime()
    if isinstance(module, (list, tuple)) and not isinstance(module, jt.Module):
        shared_pool = pool
        if (
            shared_pool is None
            and record
            and jt.flags.use_cuda
            and _core.graph_capture_supported()
        ):
            shared_pool = _GraphPool()
        return type(module)(
            GraphedModule(m, warmup=warmup, pool=shared_pool, record=record)
            for m in module
        )
    return GraphedModule(module, warmup=warmup, pool=pool, record=record)


__all__ = ["GraphedModule", "graphed_module"]
