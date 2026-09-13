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

    def __init__(self, module, *example_inputs, measure=False):
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
        self._module = module
        self._capture = None
        self._refused = None
        self._worth_it = None if measure else True
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
                # This var's own graph and nothing else. A plain `sync()` is a
                # weak sync: it also sweeps in whatever other holder vars happen
                # to be pending, and with `keep_graph` on those become part of
                # what the capture keeps alive and re-runs on every single
                # replay. Measured through nsys, a capture taken with unrelated
                # work pending executed 219 kernels a call instead of 132.
                output.sync(False, False)
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
        jt.flags.keep_graph = 1 if keep else 0
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
        """Time replay against eager once, and refuse if replay does not win.

        The replay side has to be fed a *different* Var than the captured one,
        or it measures nothing: `_replay_once` skips the input copy when the
        argument is already the captured Var, and with no input written the
        graph does not re-execute -- the call collapses to copying the output
        it already holds. Timed that way a b8s256 forward "replayed" in 6.8 ms
        against 14.0 eager, and the wrapper happily concluded replay was worth
        it. Fed a changing input it is 15.7 ms, i.e. slower.

        The stand-in carries the captured input's own bytes, so feeding it
        writes the same values back and the caller's Var is not disturbed.
        """
        def eager():
            with jt.no_grad():
                self._module(*args).sync(False)
        try:
            cap = self._capture
            stand_in = []
            for captured in cap.inputs:
                copy = jt.empty(captured.shape, captured.dtype)
                copy.sync(True, False)
                copy._copy_into(captured)
                stand_in.append(copy)
            probe = list(args)
            for slot, (captured, copy) in enumerate(zip(cap.inputs, stand_in)):
                probe[probe.index(captured)] = copy
            # Alternate the two rather than running one and then the other:
            # whichever goes second starts warm, and that bias is worth more
            # than the difference being measured. With prefill refused -- both
            # arms running the very same eager code -- the second measured
            # 1.57 ms against the first's 2.74.
            replay_once = lambda: self._replay_once(self._capture, tuple(probe))
            replayed = rebuilt = float("inf")
            for _ in range(2):
                # The eager arm runs with keep_graph off, which means its final
                # sync_all finishes the capture too -- so take a fresh one
                # before each replay round rather than timing a dead graph.
                if self._capture is None or self._capture.output.is_finished:
                    self._capture = self._capture_now(args)
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
        cap, self._capture = self._capture, None
        if cap is None or cap.output.is_finished:
            return
        before = jt.flags.keep_graph
        jt.flags.keep_graph = 0
        try:
            cap.output.sync(False)
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
