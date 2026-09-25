"""Re-run a whole step -- forward, backward, optimizer update -- from its graph.

`graph_replay` does this for an inference module: the module is a function of
its inputs and parameters, and nothing it does outlives the call. A training
step is not like that. It *updates state* -- parameters, optimizer moments,
norm statistics -- and it runs Python that keeps its own books: the
optimizer's step count, the learning rate a scheduler set. A replay has to
reproduce both, without knowing which modules and optimizers the step touched.

State is found by asking the holders. Everything a step updates, it updates by
rebinding a holder the caller keeps to the Var that computes the new value --
`p.update(new)`, an in-place op, a norm layer assigning its running mean. While
the step is captured the runtime records every such rebinding
(`_state_capture_begin`); the kept graph reads the state's old Var, so after
the capture each holder is pointed back at that old Var and every replay
writes the new value into it. When the update was already in place -- the
fused AdamW writes parameters and moments into their own storage -- there is
nothing to write back.

Python-side books are the components' own business, and they say so while the
step is captured:

  - `live_array(value, refresh)` is a small array the graph reads that is
    re-filled from `refresh()` before every replay -- the fused AdamW's step
    count and learning rate;
  - `on_replay(hook)` runs `hook` before every replay -- the optimizer
    advancing the step counts it reports;
  - `guard(read)` re-captures whenever `read()` stops returning what it
    returned during the capture -- a scalar some component baked into the
    graph, such as a learning rate that is not live;
  - `refuse(reason)` says the step cannot be replayed at all.

What cannot be seen is refused rather than replayed wrongly: a step that draws
random numbers, a step that reads a value back to the host (its Python path
then depends on tensor values), a result that is not Vars in plain containers.
Every refusal is decided before the captured graph runs, and the refused call
then runs as written, so the step happens exactly once either way; a read in
the middle of the step switches it to running as written on the spot.

A device recording re-issues the addresses it saw, so it is checked against
its leaves before every launch (`graph_leaves_moved`): reading a parameter
back to the host migrates it, and that step goes through the executor while a
later one records again.

Typical use::

    step = jt.capture_step(train_step)
    for batch in loader:
        loss = step(batch)

and, under the torch frontend, ``torch.compile(train_step, mode="reduce-overhead")``.
The first call of a given input signature runs as written (it creates lazily
built state, such as optimizer moments); the second is captured; later calls
replay. `stats` says what happened.
"""

import jittor as jt
import jittor_core as _core

from .graph_replay import (_RERECORD_LIMIT, _Unreplayable, _empty_like, _object_ids,
                           _graph_has_nondeterministic_op,
                           _input_vars, _map_inputs, _no_auto, _output_template,
                           _rebuild, _signature)


#: The capture in progress on this thread, or None. Captures do not nest.
_ACTIVE = None

#: Re-captures tolerated before a step is left to run as written. A capture
#: costs one eager step and a graph build; a guard that changes every call
#: would otherwise pay that forever.
_GIVE_UP_AFTER = 8


def active():
    """Whether a step is being captured right now."""
    return _ACTIVE is not None


def tracing():
    """Whether the code running now is being traced for a replay.

    What `torch.compiler.is_compiling()` answers under the torch frontend.
    Libraries ask it to leave out what a traced graph cannot hold -- a value
    checked on the host, as Transformers does to drop an all-ones attention
    mask -- and a capture refuses exactly that.
    """
    from .graph_replay import _TRACING
    return _ACTIVE is not None or _TRACING[0] > 0


def live_array(value, refresh, dtype="float32"):
    """`jt.array(value)`, re-filled from `refresh()` before every replay.

    Outside a capture this is just the array.
    """
    var = jt.array(value, dtype)
    cap = _ACTIVE
    if cap is None:
        return var
    # A leaf of the captured graph, so executed now: a replay re-runs pending
    # nodes, and an array's host staging is gone after its first run.
    before = jt.flags.keep_graph
    jt.flags.keep_graph = 0
    try:
        var.sync(False, False)
    finally:
        jt.flags.keep_graph = before
    cap.prologue.append(lambda: var._copy_into(jt.array(refresh(), dtype)))
    return var


def on_replay(hook):
    """Run `hook()` before every replay of the step being captured."""
    if _ACTIVE is not None:
        _ACTIVE.prologue.append(hook)


def guard(read):
    """Re-capture whenever `read()` differs from its value now."""
    if _ACTIVE is not None:
        _ACTIVE.guards.append((read, read()))


def live_fused_update(impl, lr, steps, indices, read_lr, frozen, found_inf=None,
                      owner=None):
    """What a fused optimizer update passes as its learning rate while captured.

    A replay does not run the optimizer, so what it would have done between
    steps is handed over instead. ``steps[i]`` for each of `indices` is the
    step count the optimizer keeps for an entry of this update, already
    advanced for this step; every replay advances them again. The kernel reads
    the step and ``read_lr()`` -- a scheduler may change the rate at any step
    -- on the device. `frozen` returns the hyperparameters baked into the
    launch, which are guarded. Returns ``[step, lr]`` as a live array, or `lr`
    itself -- and a refusal -- when `impl` cannot read one.

    `found_inf`, a gradient scaler's device flag, makes it ``[step, lr,
    found_inf]``: the kernel skips a flagged step. The replay has advanced the
    counts regardless, so the skipped steps are counted on the device, on
    `owner`, and taken off the step the kernel sees -- the bias correction of
    the step after a skip is what it would have been in eager mode.
    """
    if (not getattr(impl, "accepts_live_step", False) or not indices
            or len({int(steps[i]) for i in indices}) != 1):
        refuse("this optimizer update cannot read its step count on the device")
        return lr

    def advance():
        for i in indices:
            steps[i] = int(steps[i]) + 1
    on_replay(advance)
    guard(frozen)
    first = indices[0]
    hyper = live_array([float(steps[first]), float(read_lr())],
                       lambda: [float(steps[first]), float(read_lr())])
    if found_inf is None:
        return hyper
    skipped = getattr(owner, "_amp_skipped_steps", None)
    if skipped is None:
        skipped = jt.zeros((1,), "float32")
        _materialize(skipped)
        owner._amp_skipped_steps = skipped
    found = found_inf.float32().reshape((1,))
    hyper = jt.concat([hyper[:1] - skipped, hyper[1:2], found])
    skipped.update(skipped + found)
    return hyper


def _materialize(var):
    """Execute `var` now, as a finished leaf, even inside a capture."""
    before = jt.flags.keep_graph
    jt.flags.keep_graph = 0
    try:
        var.sync(False, False)
    finally:
        jt.flags.keep_graph = before


#: Philox draws for a step being captured, one kernel per draw. Each thread
#: owns one element: subsequence = element index, offset = the draw's slot in
#: this step's stretch of the stream, so no two draws -- in one step or across
#: steps -- read the same counter.
_PHILOX = r"""
#include <curand_kernel.h>
template <typename T, bool NORMAL>
__global__ void jt_step_capture_philox(T* out, long long n, const long long* state,
                                       long long draw) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStatePhilox4_32_10_t s;
    // state = [seed, first slot of this step]; four 32-bit values per slot.
    curand_init((unsigned long long)state[0], (unsigned long long)i,
                (unsigned long long)(state[1] + draw) * 4, &s);
    if (NORMAL) out[i] = (T)curand_normal_double(&s);
    else out[i] = (T)(1.0 - curand_uniform_double(&s));   // [0, 1), as jt.random
}
"""


class _StepRandom:
    """The random stream of one captured step: where it starts, how much it uses."""

    __slots__ = ("state", "draws", "base", "_seed")

    def __init__(self, seed):
        self.base = 0
        self.draws = 0
        self.state = None
        self._seed = seed

    def slot(self):
        if self.state is None:
            self.state = live_array([self._seed, 0], self._advance, "int64")
        draw = self.draws
        self.draws += 1
        return draw

    def _advance(self):
        # Every replay is the next stretch of the stream, as many slots long
        # as the capture used.
        self.base += self.draws
        return [self._seed, self.base]


def random_draw(shape, dtype, type):
    """A random Var for a step being captured on the device, or None.

    The runtime's generators are host-side: cuRAND's host API bakes the
    stream offset into the launch, so a recorded graph repeats one draw on
    every launch. A capture draws through Philox instead, reading its seed
    and position from a live array the replay advances -- as PyTorch's CUDA
    graphs do. The numbers are not the eager generator's: the same
    distribution, a different stream.
    """
    cap = _ACTIVE
    if cap is None or dtype not in ("float32", "float64") or type not in ("uniform", "normal"):
        return None
    if cap.random is None:
        seed = int(jt.get_seed()) * 1000003 + id(cap) % 1000003
        cap.random = _StepRandom(seed)
    draw = cap.random.slot()
    numel = 1
    for size in shape:
        numel *= int(size)
    ctype = "float" if dtype == "float32" else "double"
    normal = "true" if type == "normal" else "false"
    return jt.code(
        list(shape), dtype, [cap.random.state],
        cuda_header=_PHILOX,
        cuda_src=f"""
        long long n = {numel};
        if (n) jt_step_capture_philox<{ctype}, {normal}><<<(n + 255) / 256, 256>>>(
            out0_p, n, (const long long*)in0_p, {draw}LL);
        """).stop_grad()


def live_rate(impl, read_lr, frozen):
    """The learning rate an update passes while a step is captured.

    ``[lr]`` as a live array when `impl` reads its rate on the device
    (``accepts_live_lr``), so a scheduler's change reaches a replay; otherwise
    the number itself, guarded, so a change re-captures. `frozen` returns the
    hyperparameters baked in either way, which are guarded.
    """
    guard(frozen)
    if getattr(impl, "accepts_live_lr", False):
        return live_array([float(read_lr())], lambda: [float(read_lr())])
    guard(read_lr)
    return read_lr()


def refuse(reason):
    """The step being captured cannot be replayed; it keeps running as written."""
    if _ACTIVE is not None and _ACTIVE.refused is None:
        _ACTIVE.refused = reason


class _Capture:
    __slots__ = ("inputs", "host_inputs", "outputs", "template", "roots", "state",
                 "prologue", "guards", "signature", "refused", "random",
                 "frozen_random")

    def __init__(self):
        self.prologue = []
        self.guards = []
        self.refused = None
        self.random = None


def _finish_normally(roots):
    """Run a graph built under `keep_graph` once more, as an ordinary step, and finish it."""
    pending = [r for r in roots if not r.is_finished]
    if not pending:
        return
    before = jt.flags.keep_graph
    jt.flags.keep_graph = 0
    try:
        for r in pending:
            r._release_kept()
        jt.sync(pending, False, True)
    finally:
        jt.flags.keep_graph = before


def _unique(vars_):
    seen, result = set(), []
    for v in vars_:
        if v.var_ptr not in seen:
            seen.add(v.var_ptr)
            result.append(v)
    return result


class StepCapture:
    """A callable that replays `fn`'s captured step. See the module docstring."""

    def __init__(self, fn, *, record=True):
        self._fn = fn
        self._capture = None
        self._refused = None
        self._seen = None
        self._recaptures = 0
        self._cuda_graph = 0
        self._graph_out = None
        self._graph_refused = None if record else "recording was turned off"
        self._record_at = 2
        self._rerecords = 0
        #: Why the last capture was thrown away, for diagnosis.
        self.last_invalidated = None
        self.stats = {"eager": 0, "captured": 0, "replayed": 0, "graph": 0}

    def __del__(self):
        try:
            self.invalidate()
        except Exception:
            pass

    @property
    def refused(self):
        """Why the step is not replayed, or None."""
        return self._refused

    def _eager(self, args, kwargs):
        self.stats["eager"] += 1
        with _no_auto():
            return self._fn(*args, **kwargs)

    # -- capture -------------------------------------------------------
    def _capture_now(self, args, kwargs):
        """Run the step once, capturing it. Returns (capture or None, result)."""
        global _ACTIVE
        # Everything that exists now becomes a leaf of the graph: a replay
        # re-runs whatever is still pending, including a leaf's own producer.
        jt.sync_all()

        def private_copy(value):
            copy = _empty_like(value)
            copy._copy_into(value)
            return copy
        private_args, private_kwargs = _map_inputs((args, kwargs), private_copy)

        cap = _Capture()
        before = jt.flags.keep_graph
        _ACTIVE = cap
        _core._state_capture_begin()
        jt.flags.keep_graph = 2
        records = []
        readbacks = _core._host_readback_count()
        try:
            with _no_auto():
                result = self._fn(*private_args, **private_kwargs)
        finally:
            records = _core._state_capture_end()
            jt.flags.keep_graph = before
            _ACTIVE = None
        if _core._host_readback_count() != readbacks:
            cap.refused = cap.refused or (
                "the step read a value back to the host, so its path depends on "
                "tensor values and cannot be replayed")

        outputs = []
        try:
            template = _output_template(result, outputs, {}, _object_ids((args, kwargs)))
        except _Unreplayable as exc:
            template = None
            cap.refused = cap.refused or str(exc)
        roots = _unique(outputs + [new for _, _, new in records])
        # Everything that can refuse is decided before the graph runs: a
        # refused step then runs exactly once, as written. Deciding after --
        # running it kept, then putting the state back and running it again --
        # took a step twice wherever an update writes storage the capture
        # cannot see: the fused SGD writes its velocities into its own inputs.
        if cap.refused is None and not roots:
            cap.refused = "the step computes nothing"
        # A host-generator draw left in the graph re-runs, and so draws again,
        # on every executor replay; only a device recording would freeze it.
        cap.frozen_random = _graph_has_nondeterministic_op()
        if cap.refused is not None:
            self._refused = cap.refused
            _finish_normally(roots)
            return None, result

        # A step that synced its own results has run already; running the
        # kept graph again would take it twice.
        if any(not r._mem_ptr_now for r in roots):
            jt.flags.keep_graph = 2
            try:
                jt.sync(roots, False, False)
            finally:
                jt.flags.keep_graph = before

        # Point every rebound holder back at its old Var, which the graph
        # reads, and give it the new value. The step has run once, so this is
        # its update; every replay repeats it.
        state = []
        for holder, old, new in records:
            inplace = old._mem_ptr_now == new._mem_ptr_now
            if not inplace:
                old._copy_into(new, False)
            holder._update(old)
            state.append((holder, old, new, inplace))

        cap.inputs = _input_vars((private_args, private_kwargs), [])
        cap.host_inputs = any(int(v.device_id) < 0 for v in cap.inputs)
        cap.outputs = outputs
        cap.template = template
        cap.roots = roots
        cap.state = state
        cap.signature = _signature(args, kwargs)
        return cap, self._results(cap, outputs)

    def _results(self, cap, sources):
        outs = [_empty_like(o) for o in cap.outputs]
        for out, src in zip(outs, sources):
            out._copy_into(src, False)
        return _rebuild(cap.template, outs)

    # -- guards --------------------------------------------------------
    def _stale(self, cap, args, kwargs):
        for root in cap.roots:
            if root.is_finished:
                return "the captured graph was finished, most likely by a read"
        if _signature(args, kwargs) != cap.signature:
            return "the inputs changed shape or dtype"
        for holder, old, _, _ in cap.state:
            if holder.var_ptr != old.var_ptr:
                return "state the step updates was replaced from outside it"
        for read, value in cap.guards:
            if read() != value:
                return "a value the step baked in has changed"
        return None

    # -- replay --------------------------------------------------------
    def _write_back(self, cap):
        for _, old, new, inplace in cap.state:
            if not inplace:
                old._copy_into(new, False)

    def _record_cuda_graph(self, cap):
        """Record one replay -- the step and its write-backs -- as a device graph."""
        if not _core.graph_capture_supported():
            self._graph_refused = "this build cannot record device graphs"
            return
        host = _core.graph_host_work(cap.roots)
        if host:
            self._graph_refused = ("part of the step runs on the host (%s), which a "
                                   "recording would drop" % host)
            return
        if cap.frozen_random:
            self._graph_refused = ("the step draws from the host generator, which a "
                                   "recording would repeat")
            return
        outs = [_empty_like(o) for o in cap.outputs]
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 2
        handle = 0
        try:
            if not _core.graph_capture_begin():
                self._graph_refused = "the device refused to start recording"
                return
            try:
                jt.sync(cap.roots, False, False)
                self._write_back(cap)
                for out, src in zip(outs, cap.outputs):
                    out._copy_into(src, False)
            finally:
                handle = _core.graph_capture_end()
        except Exception as exc:            # a capture poisons its stream
            self._graph_refused = f"recording raised {type(exc).__name__}: {exc}"
            if handle:
                _core.graph_release(handle)
            return
        finally:
            jt.flags.keep_graph = before
        if not handle:
            self._graph_refused = "the recording contained no device work"
            return
        self._cuda_graph = handle
        self._graph_out = outs
        _core.graph_bind_leaves(handle, cap.roots)

    def _replay(self, cap, args, kwargs):
        for hook in cap.prologue:
            hook()
        if self._cuda_graph and cap.host_inputs:
            _core.graph_wait()
        for captured, given in zip(cap.inputs, _input_vars((args, kwargs), [])):
            captured._copy_into(given)
        if self._cuda_graph and not _core.graph_leaves_moved(self._cuda_graph):
            _core.graph_launch(self._cuda_graph)
            self.stats["graph"] += 1
            return self._results(cap, self._graph_out)
        if self._cuda_graph:
            # A leaf the recording reads has moved -- a parameter read back to
            # the host migrates -- so its addresses are stale; this step goes
            # through the executor and a later one records again.
            _core.graph_release(self._cuda_graph)
            self._cuda_graph = 0
            self._graph_out = None
            self._rerecords += 1
            if self._rerecords >= _RERECORD_LIMIT:
                self._graph_refused = "the leaves a recording reads kept moving"
            else:
                self._record_at = self.stats["replayed"] + 2
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 2
        try:
            jt.sync(cap.roots, False, False)
            self._write_back(cap)
        finally:
            jt.flags.keep_graph = before
        return self._results(cap, cap.outputs)

    def __call__(self, *args, **kwargs):
        if self._refused is not None:
            return self._eager(args, kwargs)
        cap = self._capture
        reason = self._stale(cap, args, kwargs) if cap is not None else None
        if reason is not None:
            self.last_invalidated = reason
            self.invalidate()
            cap = None
            self._recaptures += 1
            if self._recaptures >= _GIVE_UP_AFTER:
                self._refused = "the step kept changing, so it was re-captured too often"
                return self._eager(args, kwargs)
        if cap is None:
            signature = _signature(args, kwargs)
            if signature != self._seen:
                # The first call of a signature runs as written: it builds
                # whatever the step creates lazily, which must exist -- and
                # be executed -- before a capture can treat it as a leaf.
                self._seen = signature
                return self._eager(args, kwargs)
            cap, result = self._capture_now(args, kwargs)
            self.stats["captured"] += 1
            if cap is None:
                self.stats["eager"] += 1
                return result
            self._capture = cap
            return result
        if (not self._cuda_graph and self._graph_refused is None
                and self.stats["replayed"] >= self._record_at):
            self._record_before_replay(cap, args, kwargs)
        out = self._replay(cap, args, kwargs)
        self.stats["replayed"] += 1
        return out

    def _record_before_replay(self, cap, args, kwargs):
        """Record the device graph right before a replay, which then launches it.

        Recording executes nothing, and it needs every buffer and kernel in
        place -- which the executor replays before it have done. It does not
        need a run of its own: for a step, that run *is* a step, and it had to
        be undone again from a snapshot of all the state. Waiting for the
        device is enough to keep earlier work out of the recording.
        """
        _core.graph_wait()
        self._record_cuda_graph(cap)

    def invalidate(self):
        """Release the captured step; the next call captures again.

        Without running it: running a step's graph is taking the step. The
        marks come off, so the graph is an ordinary pending one, and with the
        capture's references gone nothing holds the part that updates state.
        """
        if self._cuda_graph:
            _core.graph_release(self._cuda_graph)
            self._cuda_graph = 0
        self._graph_out = None
        cap, self._capture = self._capture, None
        if cap is None:
            return
        for root in cap.roots:
            if not root.is_finished:
                root._release_kept()


def capture_step(fn, *, record=True):
    """Replay `fn`'s step instead of rebuilding it. See the module docstring."""
    return StepCapture(fn, record=record)
