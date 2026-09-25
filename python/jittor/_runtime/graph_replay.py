"""Re-run an inference graph instead of rebuilding it every step.

Building the graph is the dominant cost of a small step. For a decode step of
an 8-layer d512 transformer (batch 1, one token) the 2.6 ms divides into about
1.7 ms of python-side graph construction, 0.1 ms of planning and 0.8 ms of
launch loop. Cutting per-op frontend cost cannot close that -- even a free
frontend leaves more than the whole step costs in PyTorch -- but not rebuilding
the graph at all can: the same step re-run from a captured graph measured
1.31 ms against 2.70 ms rebuilding it, and 1.37 ms for the equivalent PyTorch
step.

This is opt-in, and deliberately so. A captured graph is a photograph of one
set of shapes, one set of parameters and one path through the module's python
code; it cannot notice that any of those changed. What it *can* notice is
checked on every call, and a capture that stops being valid is thrown away
rather than used:

  - the arguments' shapes, dtypes and count,
  - whether the module is in training mode,
  - whether any parameter has been rebound (an optimizer step, a load, a
    device move) -- the graph reads the parameter Vars it captured, and an
    update replaces them, so it would otherwise quietly answer with the old
    weights,
  - whether the captured graph is still unfinished. This is the one that
    matters most, because a finished graph does not fail: it keeps answering
    with the values it last computed. Reading a value out of a kept graph
    finishes it, and so does any batch that collects it outside the capture,
    so a long-lived capture does get retaken from time to time. `stats` says
    how often.

Two things make a graph unreplayable rather than merely stale, and capture
refuses both rather than answering wrongly: a graph that draws random numbers
(a replay would repeat the same draw), and a traced call that reads a value
back to the host, because then the python path taken depends on tensor values
and the next call's path may differ. A readback is detected by its effect --
it finishes the graph being captured.

This is inference, and only inference. The call runs under `no_grad` and the
result carries no gradient -- wrapping a module you are training does not
raise, it just hands back something you cannot backpropagate through. Checking
that by hand is not possible from here: a jittor float Var reports
`requires_grad` true by default, so the flag says nothing about intent, and
gating on whether the caller happens to be inside a `no_grad` scope would turn
the wrapper off for ordinary inference code that never opens one.

Typical use::

    replay = jt.graph_replay(model, example_input)
    for batch in stream:
        out = replay(batch)

Arguments may be positional or keyword, and plain tuples, lists and dicts of
Vars; anything else is matched by identity and treated as a constant of the
capture. The result may be a Var or any nesting of tuples, lists, named
tuples, dicts and dataclasses of them -- `ModelOutput` and diffusers' output
classes included -- and comes back as that same structure around fresh Vars
of the module's own tensor type. Under the torch frontend,
`torch.compile(model, mode="reduce-overhead")` wraps a module in this.

What it is worth, on the one shape that measured reproducibly here -- a
decode step, eager and replayed in separate processes, a different input every
call, every answer checked against eager:

    tf-d512-L8 decode b1s1   eager 2.54-2.67 ms   replayed 1.57-1.79 ms

About 1.5x, repeatable across runs. For reference the equivalent PyTorch step
is 1.37-1.43 ms, so this closes most of that gap without closing all of it.

The other shapes are not quoted because they did not measure reproducibly on
this machine: a b8s256 forward lands in one of two states about 2x apart
(14.06 ms or 6.49 ms) from run to run, on the eager side as much as here, and
the same bimodality shows up in prefill. Measure your own model rather than
believing a table.

Two ways to be fooled, both of which fooled the author:

  - Feeding the captured Var back in measures nothing. `_replay_once` skips
    the input copy when the argument *is* the captured Var, and with no input
    written the graph does not re-execute -- the call collapses to copying the
    output it already holds, and it still answers correctly, because the
    answer for that input has not changed. b8s256 "replayed" in 6.8 ms that
    way against 14.0 eager.
  - Timing both arms in one process measures the order. Whichever runs second
    starts warm: with replay refused, so that both arms ran the very same
    eager code, the second measured 1.57 ms against the first's 2.74.

Replay is not free money. A graph the runtime already overlaps well comes out
slower replayed -- an mlp forward measured 0.33 ms eager and 0.42 ms replayed,
a conv stack was level. **A/B your own model** before keeping this; the two
traps above say how to do it without fooling yourself.

`measure=True` will do that A/B once at capture and refuse for good if replay
loses, but it is off by default: the measurement perturbs the process it
measures, leaving every later replay executing 219 kernels instead of 132 and
the wrapper 30% slower than it is without it. That is more than the margin it
exists to protect.

`replay.stats` says what actually happened -- how many calls replayed, how
many fell back and why -- because a silent fallback that quietly costs the
speedup is worse than none.
"""

from contextlib import nullcontext as _nullcontext
import copy as _copy
import dataclasses as _dataclasses
import time
import weakref

import jittor as jt
import jittor_core as _core

from .. import flags


#: Ops whose output depends on more than their inputs. A captured graph
#: replays the values it recorded, so a graph containing one of these is not
#: re-runnable at all and capture refuses rather than repeating a draw.
_NONDETERMINISTIC_OPS = ("random", "curand_random")


class _no_auto:
    """Run a module call without the automatic policy looking at it.

    Everything in here calls the module directly, and those calls re-enter
    `Module.__call__` at depth 0 -- where the automatic policy engages. Left
    alone, an explicit `jt.graph_replay(model)` would end up with a second,
    automatic capture of the same module nested inside its own, and the two
    would take turns answering with each other's buffers.
    """

    __slots__ = ("_module_cls", "_depth")

    def __enter__(self):
        from jittor._core.module import Module
        self._module_cls = Module
        self._depth = Module._call_depth
        Module._call_depth = self._depth + 1
        return self

    def __exit__(self, *exc):
        self._module_cls._call_depth = self._depth
        return False


class _Capture:
    __slots__ = ("inputs", "host_inputs", "outputs", "template", "params", "training",
                 "signature")


class _Unreplayable(Exception):
    """A module result the capture cannot hand back; the message says why."""


def _spec(value):
    """What has to match for a captured graph to still answer for `value`."""
    if isinstance(value, jt.Var):
        return ("var", type(value), tuple(value.shape), str(value.dtype))
    if value is None or isinstance(value, (int, float, bool, str)):
        return ("scalar", type(value).__name__, value)
    if type(value) in (tuple, list):
        return (type(value).__name__,) + tuple(_spec(v) for v in value)
    if type(value) is dict:
        return ("dict",) + tuple((k, _spec(v)) for k, v in value.items())
    return ("other", type(value).__name__, id(value))


def _signature(args, kwargs=None):
    if not kwargs:
        return tuple(_spec(a) for a in args)
    return (tuple(_spec(a) for a in args),
            tuple((k, _spec(v)) for k, v in kwargs.items()))


def _input_vars(value, out):
    """Every Var in `value`, in the order `_map_inputs` visits them."""
    if isinstance(value, jt.Var):
        out.append(value)
    elif type(value) in (tuple, list):
        for v in value:
            _input_vars(v, out)
    elif type(value) is dict:
        for v in value.values():
            _input_vars(v, out)
    return out


def _map_inputs(value, fn):
    """`value` with each Var replaced by `fn(var)`.

    Only the plain containers are walked, the same ones `_spec` describes:
    anything else is matched by identity, so a Var hidden inside it is a
    constant of the capture, exactly as a closure variable would be.
    """
    if isinstance(value, jt.Var):
        return fn(value)
    if type(value) in (tuple, list):
        return type(value)(_map_inputs(v, fn) for v in value)
    if type(value) is dict:
        return {k: _map_inputs(v, fn) for k, v in value.items()}
    return value


def _output_template(value, leaves, index):
    """Describe `value` as a tree over `leaves`, the distinct Vars it holds.

    What a module returns is rarely a lone Var: `nn.LSTM` returns a tuple, a
    diffusers UNet an output dataclass, a transformers model a `ModelOutput`
    (a dict subclass that is also a dataclass). The graph only needs the Vars;
    the rest is rebuilt around fresh ones on every call.
    """
    if isinstance(value, jt.Var):
        i = index.get(id(value))
        if i is None:
            i = index[id(value)] = len(leaves)
            leaves.append(value)
        return ("var", i)
    if value is None or isinstance(value, (bool, int, float, str)):
        return ("const", value)
    if type(value) in (tuple, list):
        return ("seq", type(value), [_output_template(v, leaves, index) for v in value])
    if isinstance(value, tuple) and hasattr(type(value), "_fields"):
        return ("namedtuple", type(value),
                [_output_template(v, leaves, index) for v in value])
    if isinstance(value, dict):
        return ("dict", value,
                [(k, _output_template(v, leaves, index)) for k, v in value.items()])
    if _dataclasses.is_dataclass(value) and not isinstance(value, type):
        return ("dataclass", value,
                [(f.name, _output_template(getattr(value, f.name), leaves, index))
                 for f in _dataclasses.fields(value)])
    raise _Unreplayable(f"the module returned a {type(value).__name__}, which "
                        "replay cannot rebuild")


def _rebuild(template, values):
    kind = template[0]
    if kind == "var":
        return values[template[1]]
    if kind == "const":
        return template[1]
    if kind == "seq":
        return template[1](_rebuild(t, values) for t in template[2])
    if kind == "namedtuple":
        return template[1](*[_rebuild(t, values) for t in template[2]])
    if kind == "dict":
        if type(template[1]) is dict:
            return {k: _rebuild(t, values) for k, t in template[2]}
        # A dict subclass -- `ModelOutput`, diffusers' `BaseOutput` -- keeps
        # its fields both as items and as attributes, and assigning an item
        # updates both; a copy of the captured object keeps whatever else it
        # carries.
        result = _copy.copy(template[1])
        for k, t in template[2]:
            result[k] = _rebuild(t, values)
        return result
    result = _copy.copy(template[1])
    for name, t in template[2]:
        object.__setattr__(result, name, _rebuild(t, values))
    return result


def _native_dtype(var):
    """The native dtype, also for a frontend tensor that reports its own."""
    native = getattr(type(var), "_frontend_native_dtype", None)
    if native is not None:
        return str(native.__get__(var, type(var)))
    return var.dtype


def _empty_like(var):
    """A materialized, uninitialized Var shaped, typed and placed like `var`.

    Of `var`'s own Python type, not a plain Var: under the torch frontend the
    module's code calls tensor methods that only the frontend type has, and a
    caller expects the result type it would have got from the module. And on
    `var`'s placement: an explicitly placed tensor next to one the runtime
    placed is refused by the first op that sees both.
    """
    token = _core._set_tensor_frontend_type(type(var))
    placement = None
    try:
        backend = var.placement_backend
        if backend >= 0:
            placement = _core._set_tensor_placement(backend, max(int(var.device_id), 0))
        try:
            with _device_scope_like(var):
                result = jt.empty(var.shape, _native_dtype(var))
        finally:
            if placement is not None:
                _core._reset_tensor_placement(placement)
    finally:
        _core._reset_tensor_frontend_type(token)
    result.sync(False, False)
    return result


def _device_scope_like(var):
    """Allocate inside this block on ``var``'s device, not the ambient one."""
    device = int(var.device_id) if isinstance(var, jt.Var) else -1
    if device < 0:
        return _nullcontext()
    return jt.flag_scope(device_id=device)


def _training(module):
    is_training = getattr(module, "is_training", None)
    if callable(is_training):
        return bool(is_training())
    return bool(getattr(module, "training", False))


def _sync_result(output):
    """Finish whatever a module returned, whatever its structure.

    Calling ``.sync()`` on the return value directly assumed a single Var, and
    every multi-output module raised ``AttributeError: 'tuple' object has no
    attribute 'sync'`` from inside auto-replay instead: ``nn.RNN``,
    ``nn.LSTM`` and ``nn.GRU`` all return ``(output, hidden)``.
    """
    leaves = []
    try:
        _output_template(output, leaves, {})
    except _Unreplayable:
        pass
    if leaves:
        jt.sync(leaves)


def _allocated_bytes():
    """Every byte the pools have handed out so far, host and devices together."""
    total = _core.device_memory_allocated_total(-1)
    for device in range(_core.get_device_count()):
        total += _core.device_memory_allocated_total(device)
    return total


def _graph_has_nondeterministic_op():
    """True if anything still pending draws random numbers."""
    try:
        graphs = jt.dump_all_graphs()
    except Exception:
        # Introspection is a debug entry point; if it is unavailable, say so
        # by refusing the capture rather than replaying an unknown graph.
        return True
    for node in getattr(graphs, "nodes_info", ()):
        text = node if isinstance(node, str) else str(node)
        for name in _NONDETERMINISTIC_OPS:
            if name in text:
                return True
    return False


class GraphReplay:
    """A callable that re-runs `module`'s captured graph. See the module docstring."""

    def __init__(self, module, *example_inputs, measure=False, weak=False,
                 max_retained_bytes=None):
        """`measure=True` times replay against eager once and refuses if it loses.

        Off by default, because the measurement does not leave the process as
        it found it: after it runs, every replay executes 219 kernels instead
        of 132 (nsys), and the wrapper measures 1.53 ms a call instead of 1.18.
        Where those extra kernels come from was not isolated -- capturing only
        the requested graph, sweeping pending work, and `jt.gc()` afterwards
        all left it unchanged -- so rather than ship a safety check that costs
        30% of what it is protecting, it is opt-in and the caller is told to
        A/B their own model instead.
        """
        # The automatic policy keeps its state on the module, so holding the
        # module back would be a cycle -- and a captured graph is never
        # finished by itself, so until something releases it the whole graph
        # stays pending and later batches sweep it up. Measured: an inference
        # phase left 417 Vars alive after its model was dropped, and the
        # training phase that followed in the same process went 6.57 -> 7.35 ms.
        # A weak reference lets the capture die with the module it belongs to.
        if weak:
            self._module_ref = weakref.ref(module)
            self._module_held = None
        else:
            self._module_ref = None
            self._module_held = module
        self._capture = None
        self._refused = None
        self._worth_it = None if measure else True
        # What a device recording may keep allocated; None for no bound. The
        # automatic policy sets one (`auto_graph_replay_retain_bytes`): it
        # engages on its own, and a recording holds every intermediate.
        self._max_retained_bytes = max_retained_bytes
        self._graph_bytes = 0
        # The device-side recording of a replay, once there is one. Replaying
        # through the executor still costs about 4 us of host time per
        # operator -- the plan walk, the per-operator scopes, the allocation
        # check, the launch -- and a decode step has of the order of a hundred
        # operators. A recorded graph pays that once, at capture, and every
        # later call is a single launch: measured 2.24 us for a whole step.
        self._cuda_graph = 0
        self._graph_out = None
        self._graph_refused = None
        self.stats = {"captured": 0, "replayed": 0, "rebuilt": 0, "graph": 0}
        if example_inputs:
            self(*example_inputs)

    @property
    def _module(self):
        """The wrapped module, or None once a weakly-held one has gone."""
        if self._module_ref is not None:
            return self._module_ref()
        return self._module_held

    def __del__(self):
        # Without this a capture outlives everything: its nodes are marked so
        # the executor never finishes them, and nobody else knows to ask.
        try:
            self.invalidate()
        except Exception:
            pass

    # -- capture -------------------------------------------------------
    def _params(self):
        params = getattr(self._module, "parameters", None)
        return list(params()) if callable(params) else []

    def _capture_now(self, args, kwargs=None):
        kwargs = kwargs or {}
        params = self._params()
        given = _input_vars((args, kwargs), [])
        # Every leaf the graph reads must already be materialized: a re-run
        # re-executes whatever is still pending, including a leaf's own
        # producer, whose host staging is gone by then.
        for leaf in given + params:
            if isinstance(leaf, jt.Var):
                leaf.sync(True, False)

        # The graph is built on private copies of the inputs, never on the
        # caller's Vars. Otherwise an argument that *is* the Var the capture
        # was taken with skips the input copy -- and with no input written the
        # graph does not re-execute, so the call hands back whatever the
        # previous input produced. Correct-looking, silently wrong, and it
        # only shows when the same Var comes round again.
        #
        # On the argument's device, not the ambient one. `jt.empty` follows
        # the ambient `device_id`, so a module whose tensors live anywhere else
        # -- which is how a multi-device server drives one -- got a device-0
        # copy of a device-1 input, and the very first op inside the module
        # was handed a mix that `dispatch_context` refuses: "Expected all
        # inputs to be on the same device, but found 0 and 1".
        def private_copy(value):
            copy = _empty_like(value)
            copy._copy_into(value)
            return copy
        private_args, private_kwargs = _map_inputs((args, kwargs), private_copy)

        before = jt.flags.keep_graph
        # 2, not 1: the graph stays re-runnable, but an intermediate's memory
        # goes back once the run has no further use for it, so a capture holds
        # what a normal call peaks at rather than the sum of everything it
        # allocates.
        jt.flags.keep_graph = 2
        try:
            with _no_auto(), jt.no_grad():
                output = self._module(*private_args, **private_kwargs)
                outputs = []
                try:
                    template = _output_template(output, outputs, {})
                except _Unreplayable as exc:
                    self._refused = str(exc)
                    return None
                if not outputs:
                    self._refused = "the module returned no Var"
                    return None
                # These vars' own graph and nothing else. A plain `sync()` is a
                # weak sync: it also sweeps in whatever other holder vars happen
                # to be pending, and with `keep_graph` on those become part of
                # what the capture keeps alive and re-runs on every single
                # replay. Measured through nsys, a capture taken with unrelated
                # work pending executed 219 kernels a call instead of 132.
                jt.sync(outputs, False, False)
            if _graph_has_nondeterministic_op():
                self._refused = "the graph draws random numbers, so a replay would repeat them"
                return None
            if any(o.is_finished for o in outputs):
                # Nothing outside this method has touched the graph yet, so
                # what can have finished it is the traced call reading a value
                # back -- its python path then depends on tensor values and the
                # next call's path may differ -- or an output that was never
                # computed here at all (an input or a parameter handed back).
                self._refused = ("the traced call read a value back to the host, or "
                                 "returned a Var it did not compute, so it cannot be replayed")
                return None
        finally:
            jt.flags.keep_graph = before

        cap = _Capture()
        cap.inputs = _input_vars((private_args, private_kwargs), [])
        # A host input the graph copies to the device -- a diffusion
        # timestep, typically -- is read by a recorded graph when it *runs*.
        cap.host_inputs = any(int(v.device_id) < 0 for v in cap.inputs)
        cap.outputs = outputs
        cap.template = template
        # Identity, not value: an optimizer step or a load rebinds the holder
        # to a new Var, and the captured graph would keep reading the old one.
        cap.params = [(p, p.var_ptr) for p in params]
        cap.training = _training(self._module)
        cap.signature = _signature(args, kwargs)
        return cap

    # -- guards --------------------------------------------------------
    def _stale(self, cap, args, kwargs=None):
        # The decisive one, and the only one that would otherwise be silent:
        # a finished graph still *answers*, with whatever it last computed.
        # Reading a value out of the captured output finishes it (the fetch
        # runs a batch, and a batch outside `keep_graph` finishes what it
        # collects), so anyone who reaches past this wrapper and reads the
        # captured Var lands here rather than on a stale number.
        for output in cap.outputs:
            if output.is_finished:
                return "the captured graph was finished, most likely by a read"
        if _signature(args, kwargs) != cap.signature:
            return "the inputs changed shape or dtype"
        if _training(self._module) != cap.training:
            return "the module changed training mode"
        for holder, ptr in cap.params:
            if holder.var_ptr != ptr:
                return "a parameter was replaced since the graph was captured"
        return None

    # -- is it actually faster? ----------------------------------------
    def _time(self, run, keep, rounds=3, per=10):
        """Wall time per call, with the device waited on.

        `keep` says whether the runs being timed are replays. It has to differ
        between the two arms and this is not a detail: with `keep_graph` on for
        the eager arm as well, each of its forwards leaves a graph that is
        never finished, and every later sync collects and re-runs all of them.
        The measurement then made the wrapper permanently 0.35 ms a call slower
        than it is -- 1.53 ms against 1.18 -- which is more than the whole
        difference it was supposed to be measuring.
        """
        best = float("inf")
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 2 if keep else 0
        try:
            for _ in range(per):
                run()
            jt.sync_all(True)
            for _ in range(rounds):
                t0 = time.perf_counter()
                for _ in range(per):
                    run()
                jt.sync_all(True)
                best = min(best, (time.perf_counter() - t0) / per)
        finally:
            jt.flags.keep_graph = before
        return best

    def _measure(self, args, kwargs):
        """Time replay against eager once, and refuse if replay does not win.

        Both arms have to do the same work. That used to need a stand-in Var,
        because `_replay_once` skipped the input copy when handed the very Var
        the capture was taken with -- and with no input written the graph does
        not re-execute, so the call collapsed to copying the output it already
        held. Timed that way a b8s256 forward "replayed" in 6.8 ms against 14.0
        eager. The capture reads private buffers now, so the copy always
        happens and there is nothing left to fake.
        """
        def eager():
            with _no_auto(), jt.no_grad():
                _sync_result(self._module(*args, **kwargs))
        try:
            # No stand-in is needed: the capture reads private buffers, so
            # `_replay_once` copies the input on every call and the graph
            # re-executes whatever Var it is handed.
            # Alternate the two rather than running one and then the other:
            # whichever goes second starts warm, and that bias is worth more
            # than the difference being measured. With prefill refused -- both
            # arms running the very same eager code -- the second measured
            # 1.57 ms against the first's 2.74.
            replay_once = lambda: self._replay_once(self._capture, args, kwargs)
            replayed = rebuilt = float("inf")
            for _ in range(2):
                # The eager arm runs with keep_graph off, which means its final
                # sync_all finishes the capture too -- so take a fresh one
                # before each replay round rather than timing a dead graph.
                if self._capture is None or self._stale(self._capture, args, kwargs):
                    self._capture = self._capture_now(args, kwargs)
                    if self._capture is None:
                        self._worth_it = True
                        return
                replayed = min(replayed, self._time(replay_once, keep=True))
                rebuilt = min(rebuilt, self._time(eager, keep=False))
        except Exception:
            # Timing is an optimization, not a contract. If anything about the
            # measurement fails, keep the capture and let the guards do their
            # job rather than losing the feature to a measurement problem.
            self._worth_it = True
            self.invalidate()
            return
        # Refuse only a real loss, not a tie. A graph where the two are level
        # costs nothing to replay, and a threshold that demanded a win would
        # turn the wrapper off unpredictably on small graphs where the two
        # measurements sit inside each other's noise.
        self._worth_it = replayed < rebuilt * 1.05
        if not self._worth_it:
            self._refused = (f"replay was slower than rebuilding for this graph "
                             f"({replayed * 1e3:.3f} ms against {rebuilt * 1e3:.3f} ms), "
                             f"so every call goes to the module")
        # The measurement ran the graph many times and drained the device
        # around it; take a fresh capture rather than trusting that one.
        self.invalidate()
        # And sweep up. The replay arm drains the device with `keep_graph` on,
        # which leaves *everything* pending at that moment unfinished -- not
        # only the capture -- and an unfinished node is re-run by every later
        # sync. Left behind, that made every subsequent call 0.35 ms slower
        # (1.53 ms against 1.18) for the rest of the process.
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 0
        try:
            jt.sync_all(True)
            jt.gc()
        except Exception:
            pass
        finally:
            jt.flags.keep_graph = before

    # -- call ----------------------------------------------------------
    # -- device-side recording ------------------------------------------
    def _record_cuda_graph(self, cap):
        """Record one replay as a device graph, so later calls are one launch.

        Taken on a LATER call, never the first: a first execution allocates,
        compiles and asks the driver questions, and a recording allows none of
        those. By the time this runs the same graph has already executed at
        least twice, so every buffer is in place and every kernel is built.

        The results land in buffers of this wrapper's own (`_graph_out`),
        outside the captured graph, because a recording re-issues fixed
        pointers: the call still hands the caller fresh Vars, copied from
        those buffers without re-running anything.

        Refusal is not an error. Anything the recording cannot contain leaves
        `_graph_refused` set and every later call replays through the executor,
        which is what it did before.
        """
        if not _core.graph_capture_supported():
            self._graph_refused = "this build cannot record device graphs"
            return False
        outs = [_empty_like(o) for o in cap.outputs]
        before = jt.flags.keep_graph
        # 2, as for a replay: an intermediate's memory goes back after its
        # last use. While recording, the pools hold such a free for the
        # recording and hand the block to a later allocation of the same
        # recording, so the graph keeps what one call peaks at -- the way
        # PyTorch's private graph pool does -- rather than every buffer it
        # touches: an SD1.5 UNet step held 4.2 GB that way against 1.9 GB.
        jt.flags.keep_graph = 2
        handle = 0
        try:
            # Drain first: the recording must contain the step, not the
            # backlog in front of it.
            jt.sync(cap.outputs, True, False)
            if not _core.graph_capture_begin():
                self._graph_refused = "the device refused to start recording"
                return False
            try:
                jt.sync(cap.outputs, False, False)
                # Inside the recording, so a launch leaves the answer here.
                for out, src in zip(outs, cap.outputs):
                    out._copy_into(src, False)
            finally:
                handle = _core.graph_capture_end()
        except Exception as exc:            # a capture poisons its stream
            self._graph_refused = f"recording raised {type(exc).__name__}: {exc}"
            if handle:
                _core.graph_release(handle)
            return False
        finally:
            jt.flags.keep_graph = before
        if not handle:
            self._graph_refused = "the recording contained no device work"
            return False
        self._cuda_graph = handle
        self._graph_out = outs
        return True

    def _replay_once(self, cap, args, kwargs=None):
        # A recording copies a host input to the device when it runs, so the
        # previous launch has to be done with that buffer before it is
        # rewritten; the host is otherwise free to run ahead of the device.
        if self._cuda_graph and cap.host_inputs:
            _core.graph_wait()
        # Always, not "unless it is the same Var": the copy is what the graph
        # re-executes for, and the capture reads buffers no caller holds.
        for captured, given in zip(cap.inputs, _input_vars((args, kwargs or {}), [])):
            captured._copy_into(given)

        # Fresh Vars, so the answer is the caller's to keep. Allocating them
        # per call is free as long as they are not waited on: `_empty_like`
        # materializes without a device wait, whereas a wait here drains the
        # device every call -- that alone cost 1.76 ms a call. Rotating a
        # fixed pair of buffers instead measures exactly the same (1.183 vs
        # 1.182 ms) and would make the result alias after two calls, which is
        # not a contract worth accepting for nothing.
        #
        # Finished, too, with `keep_graph` still off. That matters for what
        # the *caller* then does: a var that is still pending makes the
        # caller's own `out.sync()` a weak sync, which sweeps in every other
        # pending holder -- the capture among them -- and finishes it.
        # `model(x).sync(False)`, which is how an inference loop is written,
        # then destroyed the capture on every single call: 13 captures for 14
        # replays, and the whole thing 3.5x slower than eager.
        outs = [_empty_like(o) for o in cap.outputs]

        # A recorded device graph re-issues the whole step with one call. The
        # input copies above are on the same stream, so they are ordered ahead
        # of it without a wait.
        if self._cuda_graph:
            _core.graph_launch(self._cuda_graph)
            # `sync_src=False`: the launch already produced the bytes, and
            # syncing the captured outputs would run the whole graph again
            # through the executor -- which is exactly what the recording is
            # there to avoid.
            for out, src in zip(outs, self._graph_out):
                out._copy_into(src, False)
            self.stats["graph"] += 1
            return _rebuild(cap.template, outs)
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 2
        try:
            # Syncing the captured outputs is what re-runs the graph, once
            # for all of them; the copies build no op, so it does not grow.
            jt.sync(cap.outputs, False, False)
            for out, src in zip(outs, cap.outputs):
                out._copy_into(src, False)
        finally:
            jt.flags.keep_graph = before
        return _rebuild(cap.template, outs)

    def _eager(self, args, kwargs):
        self.stats["rebuilt"] += 1
        with _no_auto(), jt.no_grad():
            return self._module(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self._refused is not None:
            return self._eager(args, kwargs)

        cap = self._capture
        if cap is not None:
            reason = self._stale(cap, args, kwargs)
            if reason is not None:
                self.invalidate()
                cap = None

        if cap is None:
            # One eager call first: it materializes the parameters and any
            # buffer the module builds lazily, so the capture that follows has
            # nothing pending underneath it.
            allocated = _allocated_bytes()
            with _no_auto(), jt.no_grad():
                result = self._module(*args, **kwargs)
                _sync_result(result)
            # What a device recording of this graph would hold: every buffer
            # the call allocates, since a recording re-issues fixed pointers.
            self._graph_bytes = _allocated_bytes() - allocated
            cap = self._capture = self._capture_now(args, kwargs)
            self.stats["captured"] += 1
            if cap is None:
                # The warm-up already computed this call's answer.
                self.stats["rebuilt"] += 1
                return result
            if self._worth_it is None:
                self._measure(args, kwargs)
                if self._refused is not None:
                    return self._eager(args, kwargs)
                # `_measure` dropped the capture it timed; take a fresh one.
                cap = self._capture = self._capture_now(args, kwargs)
                self.stats["captured"] += 1
                if cap is None:
                    return self._eager(args, kwargs)

        # The results are copies, not the captured Vars: reading a captured
        # output directly would finish the graph -- `clone()` here finished it
        # outright, after which nothing re-ran and the replay silently kept
        # answering with the first input's result, in 0.08 ms.
        out = self._replay_once(cap, args, kwargs)
        self.stats["replayed"] += 1
        # Try to record only once per capture, and only after the graph has
        # run a few times: the first executions are the ones that allocate and
        # compile, and a recording tolerates neither.
        if (not self._cuda_graph and self._graph_refused is None
                and self.stats["replayed"] == 3):
            limit = self._max_retained_bytes
            if limit and self._graph_bytes > limit:
                # Replays through the executor free as they go; a recording
                # would keep all of it, for as long as the capture lives.
                self._graph_refused = (
                    "a recording would keep %.1f MiB alive, over the %.1f MiB "
                    "allowed" % (self._graph_bytes / 2**20, limit / 2**20))
            else:
                self._record_cuda_graph(cap)
        return out

    def invalidate(self):
        """Release the captured graph; the next call captures again.

        Dropping the python reference is not enough. The graph's nodes were
        deliberately left unfinished, and an unfinished node stays pending
        forever -- every later sync collects it and runs it again. Two
        recaptures and the device was executing three copies of the model per
        call: nsys counted 321 kernels a step against eager's 130. So the
        graph is handed back to the executor once with `keep_graph` off, which
        collects it, finishes it, and lets it be reclaimed. That costs one
        extra execution per invalidation, which is the price of not leaving a
        zombie behind.
        """
        # The recording points into this graph's buffers, so it dies with it.
        if self._cuda_graph:
            _core.graph_release(self._cuda_graph)
            self._cuda_graph = 0
        self._graph_out = None
        self._graph_refused = None
        cap, self._capture = self._capture, None
        if cap is None:
            return
        pending = [o for o in cap.outputs if not o.is_finished]
        if not pending:
            return
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 0
        try:
            # Take the mark off first: a marked node is never finished, by
            # design, so the sync below would otherwise run the graph and
            # leave it exactly as it was.
            for output in pending:
                output._release_kept()
            jt.sync(pending, False, True)
        except Exception:
            # Releasing is best effort: a graph that cannot run any more (its
            # parameters went away, say) must not turn into an exception from
            # whatever call happened to notice the capture was stale.
            pass
        finally:
            jt.flags.keep_graph = before

    @property
    def refused(self):
        """Why capture was refused, or None. Falls back to eager when set."""
        return self._refused


def graph_replay(module, *example_inputs, measure=False):
    """Wrap `module` so repeated inference re-runs its graph. See the module docstring."""
    return GraphReplay(module, *example_inputs, measure=measure)


# ---------------------------------------------------------------------------
# The automatic policy
#
# `jt.graph_replay(...)` is the explicit form and says exactly what it does.
# This is the same machinery applied without being asked, and it is deliberately
# narrow: the speedup is in not rebuilding a graph, which only dominates when
# the step is small, and a capture holds its intermediates for as long as it
# lives. So it engages only for a call that is
#
#   - under `no_grad`, because a replay carries no gradient;
#   - the outermost module call, not a submodule of one already running;
#   - all-positional, all-Var, with inputs totalling less than
#     `auto_graph_replay_bytes` -- the regime where rebuilding is the cost;
#   - repeating: the same shapes twice in a row, so a one-off call is never
#     captured;
#
# A capture replays through the executor and frees intermediates as it goes,
# so it costs about what a normal call peaks at. Recording it as a device
# graph, which is what makes a small step one launch, keeps every buffer, so
# that is only done for a graph allocating at most
# `auto_graph_replay_retain_bytes`. Small inputs do not bound it: an SD1.5 VAE
# decode takes a 32 KB latent and allocates 6.2 GB of 512x512 feature maps.
#
# Everything the capture cannot serve (a graph that draws random numbers, a
# traced call that read a value back, a result that is not Vars in tuples,
# lists, dicts or dataclasses) falls back and is not tried again for that
# module. So does a module whose shapes keep changing, after enough
# re-captures to show it.


#: Re-captures tolerated for one module before the policy leaves it alone. A
#: capture costs an eager forward, so a caller whose shapes change every call
#: would otherwise pay for a capture it never uses.
_GIVE_UP_AFTER = 8


class _AutoState:
    __slots__ = ("signature", "seen", "replay", "recaptures", "give_up")

    def __init__(self):
        self.signature = None
        self.seen = 0
        self.replay = None
        self.recaptures = 0
        self.give_up = False


def _element_size(dtype):
    """Bytes per element, for a native dtype or a frontend one.

    The native NanoString spells this `dsize()`; a torch frontend tensor
    reports a `torch.dtype`, which spells it `itemsize` and has no `dsize` at
    all. Reading only the native spelling made every module call through the
    torch frontend die here with "'dtype' object has no attribute 'dsize'" --
    the replay policy runs for every outermost module call, so it took down
    plain `nn.LayerNorm(x)` under the shim.
    """
    size = getattr(dtype, "dsize", None)
    if size is not None:
        return size() if callable(size) else size
    size = getattr(dtype, "itemsize", None)
    if size is not None:
        return size() if callable(size) else size
    # Unknown spelling: report one byte rather than raising. The caller only
    # compares the total against a byte threshold, so under-reporting merely
    # lets a call through to the normal path.
    return 1


def _input_bytes(args):
    total = 0
    for a in args:
        total += a.numel() * _element_size(a.dtype)
    return total


def auto_replay_for(module, args, kw):
    """The GraphReplay to use for this call, or None to run normally."""
    if kw or not args:
        return None
    flags = jt.flags
    if not flags.auto_graph_replay or not flags.no_grad:
        return None
    for a in args:
        if not isinstance(a, jt.Var):
            return None
    state = module.__dict__.get("_auto_graph_replay")
    if state is None:
        # Written through __dict__: Module.__setattr__ classifies assignments
        # into parameters and buffers, and this is neither.
        state = module.__dict__["_auto_graph_replay"] = _AutoState()
    if state.give_up:
        return None
    if _input_bytes(args) > flags.auto_graph_replay_bytes:
        return None
    signature = _signature(args)
    if signature != state.signature:
        state.signature = signature
        state.seen = 1
        if state.replay is not None:
            state.replay.invalidate()
            state.replay = None
            state.recaptures += 1
            if state.recaptures >= _GIVE_UP_AFTER:
                state.give_up = True
        return None
    state.seen += 1
    if state.seen < 2:
        return None
    if state.replay is None:
        # `measure=False`: the timing check perturbs what it measures (see
        # `_measure`), and the eligibility rules above already restrict this to
        # the shape of step where replay wins.
        state.replay = GraphReplay(
            module, measure=False, weak=True,
            max_retained_bytes=flags.auto_graph_replay_retain_bytes)
    if state.replay.refused is not None:
        state.give_up = True
        state.replay = None
        return None
    return state.replay
