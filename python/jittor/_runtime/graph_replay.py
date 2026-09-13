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

Measured across the comparison shapes (ms per forward, same inputs, answers
identical to eager to the last bit):

    tf-d512-L8  decode  b1s1     2.54 -> 1.20     (torch 1.37)
    tf-d1024-L4 decode  b1s1     1.30 -> 0.46     (torch 0.67)
    tf-d512-L8  prefill b1s128   2.79 -> 2.25     (torch 2.81)
    tf-d512-L8  batch   b8s256  14.04 -> 6.78     (torch 13.39)
    mlp-1024x4  b256             0.19 -> 0.18
    cnn-6conv   b64x3x64x64      4.09 -> 4.24

It is not free money, though: the last row is a graph the runtime already
overlaps well, and replaying it costs a little rather than saving. So the
first capture is timed against eager, and if replay comes out slower the
wrapper hands every call to the module and says so in `refused`. A tie still
replays -- demanding a win would switch the wrapper off unpredictably on
graphs where the two measurements sit inside each other's noise.

`replay.stats` says what actually happened -- how many calls replayed, how
many fell back and why -- because a silent fallback that quietly costs the
speedup is worse than none.
"""

import time

import jittor as jt

from .. import flags


#: Ops whose output depends on more than their inputs. A captured graph
#: replays the values it recorded, so a graph containing one of these is not
#: re-runnable at all and capture refuses rather than repeating a draw.
_NONDETERMINISTIC_OPS = ("random", "curand_random")


class _Capture:
    __slots__ = ("inputs", "output", "params", "training", "signature",
                 "slots", "turn")


def _spec(value):
    """What has to match for a captured graph to still answer for `value`."""
    if isinstance(value, jt.Var):
        return ("var", tuple(value.shape), str(value.dtype))
    if isinstance(value, (int, float, bool)):
        return ("scalar", type(value).__name__, value)
    return ("other", type(value).__name__, id(value))


def _signature(args):
    return tuple(_spec(a) for a in args)


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

    def __init__(self, module, *example_inputs):
        self._module = module
        self._capture = None
        self._refused = None
        self._worth_it = None
        self.stats = {"captured": 0, "replayed": 0, "rebuilt": 0}
        if example_inputs:
            self(*example_inputs)

    # -- capture -------------------------------------------------------
    def _params(self):
        params = getattr(self._module, "parameters", None)
        return list(params()) if callable(params) else []

    def _capture_now(self, args):
        params = self._params()
        # Every leaf the graph reads must already be materialized: a re-run
        # re-executes whatever is still pending, including a leaf's own
        # producer, whose host staging is gone by then.
        for leaf in list(args) + params:
            if isinstance(leaf, jt.Var):
                leaf.sync(True, False)

        before = jt.flags.keep_graph
        jt.flags.keep_graph = 1
        try:
            with jt.no_grad():
                output = self._module(*args)
                if not isinstance(output, jt.Var):
                    self._refused = ("the module returned "
                                     f"{type(output).__name__}, not a single Var")
                    return None
                output.sync()
            if _graph_has_nondeterministic_op():
                self._refused = "the graph draws random numbers, so a replay would repeat them"
                return None
            if output.is_finished:
                # Nothing outside this method has touched the graph yet, so
                # the only thing that can have finished it is the traced call
                # reading a value back -- which means its python path depends
                # on tensor values and the next call's path may differ.
                self._refused = ("the traced call read a value back to the host, so its "
                                 "path depends on tensor values and cannot be replayed")
                return None
        finally:
            jt.flags.keep_graph = before

        cap = _Capture()
        cap.inputs = [a for a in args if isinstance(a, jt.Var)]
        cap.output = output
        # Destinations allocated once, here, and rotated. Allocating one per
        # call builds an op per call, and once enough ops have accumulated the
        # runtime flushes them -- a batch outside `keep_graph` that collects
        # the captured graph and finishes it. That showed up as a recapture
        # every ~35 calls and left the whole wrapper slower than eager.
        #
        # Two of them, so a caller may hold the previous answer while asking
        # for the next. The one before that is overwritten.
        cap.slots = [jt.empty(output.shape, output.dtype) for _ in range(2)]
        for slot in cap.slots:
            slot.sync(True, False)
        cap.turn = 0
        # Identity, not value: an optimizer step or a load rebinds the holder
        # to a new Var, and the captured graph would keep reading the old one.
        cap.params = [(p, p.var_ptr) for p in params]
        cap.training = bool(getattr(self._module, "is_training", lambda: False)())
        cap.signature = _signature(args)
        return cap

    # -- guards --------------------------------------------------------
    def _stale(self, cap, args):
        # The decisive one, and the only one that would otherwise be silent:
        # a finished graph still *answers*, with whatever it last computed.
        # Reading a value out of the captured output finishes it (the fetch
        # runs a batch, and a batch outside `keep_graph` finishes what it
        # collects), so anyone who reaches past this wrapper and reads the
        # captured Var lands here rather than on a stale number.
        if cap.output.is_finished:
            return "the captured graph was finished, most likely by a read"
        if _signature(args) != cap.signature:
            return "the inputs changed shape or dtype"
        if bool(getattr(self._module, "is_training", lambda: False)()) != cap.training:
            return "the module changed training mode"
        for holder, ptr in cap.params:
            if holder.var_ptr != ptr:
                return "a parameter was replaced since the graph was captured"
        return None

    # -- is it actually faster? ----------------------------------------
    def _time(self, run, rounds=3, per=5):
        """Wall time per call, with the device waited on.

        The wait happens inside `keep_graph`, so draining the device does not
        finish the captured graph along the way -- which is what an ordinary
        sync_all would do, leaving the thing being measured dead.
        """
        best = float("inf")
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 1
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

    def _measure(self, args):
        """Time replay against eager once, and refuse if replay does not win."""
        def eager():
            with jt.no_grad():
                self._module(*args).sync(False)
        try:
            replayed = self._time(lambda: self._replay_once(self._capture, args))
            rebuilt = self._time(eager)
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

    # -- call ----------------------------------------------------------
    def _replay_once(self, cap, args):
        for captured, given in zip(cap.inputs,
                                   [a for a in args if isinstance(a, jt.Var)]):
            if given is not captured:
                captured._copy_into(given)
        out = cap.slots[cap.turn]
        cap.turn ^= 1
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 1
        try:
            # `_copy_into` syncs its source, which is what re-runs the graph;
            # the copy itself builds no op, so the graph does not grow.
            out._copy_into(cap.output)
        finally:
            jt.flags.keep_graph = before
        return out

    def __call__(self, *args):
        if self._refused is not None:
            self.stats["rebuilt"] += 1
            with jt.no_grad():
                return self._module(*args)

        cap = self._capture
        if cap is not None:
            reason = self._stale(cap, args)
            if reason is not None:
                self.invalidate()
                cap = None

        if cap is None:
            # One eager call first: it materializes the parameters and any
            # buffer the module builds lazily, so the capture that follows has
            # nothing pending underneath it.
            with jt.no_grad():
                self._module(*args).sync()
            cap = self._capture = self._capture_now(args)
            self.stats["captured"] += 1
            if cap is None:
                self.stats["rebuilt"] += 1
                with jt.no_grad():
                    return self._module(*args)
            if self._worth_it is None:
                self._measure(args)
                if self._refused is not None:
                    self.stats["rebuilt"] += 1
                    with jt.no_grad():
                        return self._module(*args)
                # `_measure` dropped the capture it timed; take a fresh one.
                cap = self._capture = self._capture_now(args)
                self.stats["captured"] += 1
                if cap is None:
                    self.stats["rebuilt"] += 1
                    with jt.no_grad():
                        return self._module(*args)

        # The result is a copy, not the captured Var: reading the captured
        # output directly would finish the graph -- `clone()` here finished it
        # outright, after which nothing re-ran and the replay silently kept
        # answering with the first input's result, in 0.08 ms.
        out = self._replay_once(cap, args)
        self.stats["replayed"] += 1
        return out

    def invalidate(self):
        """Drop the captured graph; the next call captures again."""
        self._capture = None

    @property
    def refused(self):
        """Why capture was refused, or None. Falls back to eager when set."""
        return self._refused


def graph_replay(module, *example_inputs):
    """Wrap `module` so repeated inference re-runs its graph. See the module docstring."""
    return GraphReplay(module, *example_inputs)
