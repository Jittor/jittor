# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Property tests for the graph and for liveness accounting (10.18).

Why properties and not more point cases
---------------------------------------
``tests/core`` already contains point cases for this area, and measuring them
(``tools/measure_core_test_balance.py``) is what motivated this file: of 81
files, **10** name the liveness/graph APIs at all and **7** name anything about
the executor, while 69 are about dtypes and numerics. The area with the least
coverage is the one where a bug is silent.

The three ``test_zmem_leak*`` cases in ``test_function.py`` are the shape of the
problem. All three assert ``liveness_info()["lived_vars"] == 0`` after their own
work, and all three fail with ``2 != 0`` in a whole-file run. What they actually
report is that *something earlier in the process* leaked two vars -- none of
them says which shape, which execution mode, or which counter. Three cases, one
undifferentiated number, and the number is only reachable when the file is run
in its entirety.

The property here is the same claim made independently of prior state:

    building a graph and dropping it returns the process to the lived-var count
    it started from.

Stated as a property it can be swept over graph shapes, and the sweep found
strictly more than the three point cases did -- see ``KNOWN_LEAKING_SHAPES``.

Stability
---------
Absolute lived-var counts are **not** a stable quantity in this tree: 3.01
measured the same source concurrently twice and got 9 lived vars once and 7 the
other time, and needed four serial rounds for an md5 match. Two consequences,
both deliberate here:

* Every assertion below is on a **delta** across one shape, never on an absolute
  count, so prior state cancels.
* The shape sweep runs in **one child process**, not in this one. That is not
  only for isolation of the measurement: the leaking shapes leave residue
  behind, so running them here would make this file a new source of exactly the
  cross-file flake it is testing for. One child covers the whole matrix, so the
  cost is one interpreter start rather than one per case.
"""

import json
import unittest

import pytest

import jittor as jt

from _helpers.child_process import run_child_script


# ---------------------------------------------------------------------------
# The child: one process, the whole matrix
# ---------------------------------------------------------------------------

#: Graph shapes, as source, keyed by name. Each builds a graph, executes it and
#: drops it; the harness measures the lived-var delta around the call.
#:
#: The interesting axis is *how many of a multi-output op's outputs are stopped*.
#: The two shapes that leak differ from the ones that do not by exactly that.
_SHAPES = r'''
import gc
import json

import jittor as jt
from jittor import Function


class TwoOutputs(Function):
    """A Function with two outputs, the shape ``Tapes`` exists to support."""

    def execute(self, x, y):
        self.x, self.y = x, y
        return x * y, x / y

    def grad(self, g0, g1):
        return (g0 * self.y if g0 is not None else None,
                g1 * self.x if g1 is not None else None)


class OneOutput(Function):
    def execute(self, x, y):
        self.x, self.y = x, y
        return x * y

    def grad(self, g):
        return g * self.y, g * self.x


def _pair():
    return jt.array(3.0), jt.array(4.0)


def two_outputs_none_stopped():
    a, b = _pair()
    c, d = TwoOutputs()(a, b)
    jt.sync([c, d])


def two_outputs_first_stopped():
    a, b = _pair()
    c, d = TwoOutputs()(a, b)
    c.stop_grad()
    jt.sync([c, d])


def two_outputs_second_stopped():
    a, b = _pair()
    c, d = TwoOutputs()(a, b)
    d.stop_grad()
    jt.sync([c, d])


def two_outputs_both_stopped():
    a, b = _pair()
    c, d = TwoOutputs()(a, b)
    c.stop_grad()
    d.stop_grad()
    jt.sync([c, d])


def two_outputs_second_stopped_only_first_synced():
    a, b = _pair()
    c, d = TwoOutputs()(a, b)
    d.stop_grad()
    jt.sync([c])


def two_outputs_second_stopped_never_executed():
    a, b = _pair()
    c, d = TwoOutputs()(a, b)
    d.stop_grad()


def two_outputs_second_stopped_then_backward():
    a, b = _pair()
    c, d = TwoOutputs()(a, b)
    d.stop_grad()
    da, db = jt.grad(c + d * 3, [a, b])
    assert da.data == 4 and db.data == 0


def one_output_stopped():
    a, b = _pair()
    c = OneOutput()(a, b)
    c.stop_grad()
    jt.sync([c])


def one_output_backward():
    a, b = _pair()
    c = OneOutput()(a, b)
    jt.grad(c, [a, b])


def ordinary_ops_two_results_one_stopped():
    """Not a Function: two single-output ops, one result stopped."""
    a, b = _pair()
    c, d = a * b, a / b
    d.stop_grad()
    jt.sync([c, d])


def plain_elementwise_backward():
    a = jt.array([1.0, 2.0])
    jt.grad((a * a + a).sum(), a)


SHAPES = {
    name: value for name, value in sorted(globals().items())
    if name[0] != "_" and callable(value) and getattr(value, "__module__", None) == "__main__"
    and name not in ("TwoOutputs", "OneOutput")
}
'''

_CHILD = _SHAPES + """

def measure():
    from contextlib import ExitStack as _TestPolicyStack
    with _TestPolicyStack() as _test_policy_stack:
        results = {}
        for lazy in (1, 0):
            _test_policy_stack.enter_context(jt.runtime.scope(lazy_execution=lazy))
            for name, shape in SHAPES.items():
                # Settle first, so the delta is about this shape only. gc runs
                # because a Python-side cycle (a traceback, a Function holding its
                # inputs on self) also holds Vars, and that is not a leak.
                jt.clean(); gc.collect(); jt.clean()
                before = dict(hold_vars=jt.introspection.counters.held_vars, lived_vars=jt.introspection.counters.live_vars, lived_ops=jt.introspection.counters.live_ops)["lived_vars"]
                shape()
                gc.collect(); jt.clean()
                after = dict(hold_vars=jt.introspection.counters.held_vars, lived_vars=jt.introspection.counters.live_vars, lived_ops=jt.introspection.counters.live_ops)["lived_vars"]
                results["%s/lazy=%d" % (name, lazy)] = after - before
        _test_policy_stack.enter_context(jt.runtime.scope(lazy_execution=1))
        return results


def sweep():
    from contextlib import ExitStack as _TestPolicyStack
    with _TestPolicyStack() as _test_policy_stack:
        \"""The dangling-node half of graph_check, with the registry actually on.

        ``check_graph``'s setter turns node tracking on (6.C21: in a release build
        the sweep used to walk an empty table and report success). So the number
        this returns is the evidence that the sweep ran, and it is only non-zero for
        nodes created *after* the flag went on.
        \"""
        report = {}
        jt.clean(); gc.collect(); jt.clean()
        # Before the flag: nothing is registered, so the sweep has nothing to walk.
        report["swept_before_flag"] = jt.graph_check()

        _test_policy_stack.enter_context(jt.runtime.scope(check_graph=1))
        held = [jt.array([1.0, 2.0]) for _ in range(4)]
        combined = (held[0] * held[1] + held[2]).sum()
        jt.sync([combined])
        report["swept_with_flag"] = jt.graph_check()
        # The liveness half walks forward from the holders, so this is how many
        # roots it had. It is reported separately from `swept` on purpose: the two
        # halves cover different node sets and one can be empty while the other is
        # not, which is the whole point of 6.C21.
        report["hold_vars"] = dict(hold_vars=jt.introspection.counters.held_vars, lived_vars=jt.introspection.counters.live_vars, lived_ops=jt.introspection.counters.live_ops)["hold_vars"]
        report["lived_vars"] = dict(hold_vars=jt.introspection.counters.held_vars, lived_vars=jt.introspection.counters.live_vars, lived_ops=jt.introspection.counters.live_ops)["lived_vars"]
        _test_policy_stack.enter_context(jt.runtime.scope(check_graph=0))
        del held, combined
        jt.clean(); gc.collect(); jt.clean()
        return report


out = {"deltas": measure(), "sweep": sweep()}
print("PROPERTY-JSON " + json.dumps(out), flush=True)
"""


def _run_matrix():
    child = run_child_script(_CHILD, merge_stderr=True, timeout=600)
    output = child.stdout.decode("utf8", "replace")
    marker = "PROPERTY-JSON "
    index = output.find(marker)
    if index < 0:
        raise AssertionError(
            "the liveness property child produced no result. It is one process "
            "covering the whole matrix, so this is a hard failure rather than a "
            "skip -- an empty matrix would let every assertion below pass.\n"
            + output[-4000:])
    payload = output[index + len(marker):].splitlines()[0]
    return json.loads(payload), output


class _Matrix:
    """One child run, shared by every case in this file."""

    data = None
    output = ""

    @classmethod
    def get(cls):
        if cls.data is None:
            cls.data, cls.output = _run_matrix()
        return cls.data


#: Shapes whose lived-var account does not balance, with the count they leak.
#:
#: **This is a real defect, found by this file, not a tolerated quirk.** The
#: mechanism is visible on stderr of any run that triggers it:
#:
#:     [f] node.h:263 Check failed: value_ > 0
#:         backward liveness release without a matching owner
#:
#: which is ``LivenessCounter<backward>::release()`` being called once more than
#: it was owned -- the ``NodeLiveness`` invariant 2.10 introduced these types to
#: enforce, violated. The throw aborts the release half-way and is then caught
#: and discarded by ``var_holder.cc``'s teardown path, so two Vars stay in the
#: node registry forever, each with ``f=0 b=1`` -- that is, each reporting
#: ``need_free() == true`` while still being alive.
#:
#: Those two Vars are exactly the ``2 != 0`` that ``test_zmem_leak``,
#: ``test_zmem_leak2`` and ``test_zmem_leak3`` report. Attribution belongs to
#: 2.10 (the counters) with 2.19 having established the "failed release" half;
#: this file's contribution is the trigger, minimised:
#:
#:   * a **multi-output** op created through ``Function`` (so ``Tapes``, in
#:     ``src/ops/tape_op.h``), and
#:   * **some but not all** of its outputs ``stop_grad()``-ed, and
#:   * the batch actually executed with every output synced.
#:
#: Each of those is necessary. ``two_outputs_both_stopped`` and
#: ``two_outputs_none_stopped`` are clean, so it is the asymmetry and not
#: ``stop_grad`` itself. ``one_output_stopped`` is clean, so it needs more than
#: one output -- and ``op.cc``'s vnbb guard is spelled ``_outputs.size()==1 &&
#: ... is_stop_grad()``, which is the asymmetry stated in the source; whether
#: that guard is the fix is for 2.10 to decide, this file only says the property
#: is violated. ``ordinary_ops_two_results_one_stopped`` is clean, so it is
#: specific to the taped path rather than to multi-output ops in general (a
#: two-output ``jt.code`` op was checked by hand and is also clean).
#: ``two_outputs_second_stopped_never_executed`` and
#: ``..._only_first_synced`` are clean, so the batch has to run.
#:
#: Independent of ``lazy_execution``: both modes leak on both shapes.
KNOWN_LEAKING_SHAPES = {
    "two_outputs_first_stopped/lazy=0": 2,
    "two_outputs_first_stopped/lazy=1": 2,
    "two_outputs_second_stopped/lazy=0": 2,
    "two_outputs_second_stopped/lazy=1": 2,
    "two_outputs_second_stopped_then_backward/lazy=0": 2,
}


class TestLivenessAccountingProperties(unittest.TestCase):
    """One property, swept over graph shapes and both execution modes."""

    def test_the_matrix_actually_ran(self):
        """An empty matrix would make every other case here vacuously true."""
        deltas = _Matrix.get()["deltas"]
        self.assertGreaterEqual(len(deltas), 20, deltas)
        # Both execution modes are covered: the leak is not mode-specific, and
        # a sweep that quietly lost one mode would hide half of it.
        for lazy in ("lazy=0", "lazy=1"):
            self.assertTrue(
                any(key.endswith(lazy) for key in deltas),
                "matrix lost %s: %r" % (lazy, sorted(deltas)))

    def test_dropping_a_graph_leaks_nothing_new(self):
        """The gate-protecting half: no shape may start leaking.

        Scoped to "nothing *new*" so that the known defect above does not paint
        the gate red -- a red that everyone learns to ignore protects nothing.
        A shape that begins to leak, or one that leaks more than it did, fails
        here.
        """
        deltas = _Matrix.get()["deltas"]
        offenders = {
            name: delta for name, delta in deltas.items()
            if delta != KNOWN_LEAKING_SHAPES.get(name, 0)
        }
        self.assertEqual(
            offenders, {},
            "lived-var accounting changed for these shapes. Each entry is "
            "(shape/mode: lived_vars delta over building and dropping one "
            "graph); expected 0, or the recorded count for a known leak. "
            "A new non-zero is a new leak; a zero where a leak was recorded "
            "means it was fixed -- update KNOWN_LEAKING_SHAPES and the 2.10 "
            "entry rather than widening this dict.\n"
            "known: %r" % (KNOWN_LEAKING_SHAPES,))

    @pytest.mark.xfail(
        strict=True,
        reason="2.10: backward liveness over-release leaves 2 Vars alive per "
               "occurrence. Trigger minimised in KNOWN_LEAKING_SHAPES above: a "
               "Function with several outputs, some but not all stop_grad()-ed, "
               "executed. This is the invariant as it should hold; it is strict "
               "so that fixing 2.10 turns this red and forces the bookkeeping "
               "and the three test_zmem_leak cases to be revisited.")
    def test_dropping_a_graph_leaks_nothing_at_all(self):
        """The property as it should hold, with no exceptions carved out."""
        deltas = _Matrix.get()["deltas"]
        self.assertEqual(
            {name: delta for name, delta in deltas.items() if delta}, {})

    def test_the_leak_is_two_vars_per_occurrence(self):
        """Not decoration: it pins the leak to one over-release each.

        ``NodeLiveness`` releases forward, backward and pending together. A
        single unmatched ``backward`` release strands the Var it was called on
        and the one output edge that had propagated to -- two. If this number
        moves, the mechanism is not the one described above and the attribution
        to 2.10 has to be re-derived rather than assumed.
        """
        deltas = _Matrix.get()["deltas"]
        leaked = {name: delta for name, delta in deltas.items() if delta}
        self.assertTrue(leaked, "nothing leaked; see the strict xfail above")
        self.assertEqual(sorted(set(leaked.values())), [2], leaked)


class TestGraphCheckProperties(unittest.TestCase):
    """``check_graph`` must be able to say that it checked nothing.

    6.C21 found that ``do_graph_check``'s dangling-node half swept a table that
    only ``NODE_MEMCHECK`` builds filled, so in every shipped build
    ``check_graph=1`` ran half the check and reported success. The fix made the
    registry follow the flag and made the function *return* how much it swept.

    A property test for "the check works" is worthless if it cannot tell a pass
    from a skip, so that is what these two cases separate.
    """

    def test_the_sweep_reports_how_much_it_swept(self):
        report = _Matrix.get()["sweep"]
        # Nodes are registered only while the flag is on, so a graph_check with
        # the flag off has nothing to walk -- and says so with 0 rather than
        # reporting a clean sweep. This distinction is the whole fix: before it,
        # both cases returned "fine".
        self.assertEqual(report["swept_before_flag"], 0, report)
        # With the flag on and nodes built afterwards, the sweep has real nodes.
        # This is the assertion that would have failed before 6.C21.
        self.assertGreater(report["swept_with_flag"], 0, report)

    def test_the_sweep_covers_only_what_it_could_have_registered(self):
        """``swept`` is not the live-node count, and must not be read as one.

        Nodes enter the registry only while ``check_graph`` is on, so a process
        that switched it on late has live vars the sweep can never see -- 10
        swept against 17 lived, measured. That gap is a documented limit of the
        6.C21 fix ("nodes made before the flag went on are not covered, which
        costs coverage but can never produce a false report"), and pinning it
        here keeps anyone from later treating a clean sweep as a whole-process
        guarantee.
        """
        report = _Matrix.get()["sweep"]
        self.assertGreater(report["lived_vars"], 0, report)
        self.assertLessEqual(
            report["swept_with_flag"], report["lived_vars"] + report["hold_vars"],
            report)

    def test_a_verified_graph_passes_the_liveness_recompute(self):
        """The other half: f/b/p recomputed from the edges must match.

        ``graph_check`` walks forward from the holders and recomputes all three
        counters from the edges, raising on any mismatch -- so reaching a number
        at all is the pass. What makes that meaningful rather than vacuous is
        that the walk had roots, which is what ``hold_vars`` reports here;
        paired with the sweep count above, both halves are shown to have run.
        """
        report = _Matrix.get()["sweep"]
        self.assertGreater(
            report["hold_vars"], 0,
            "the liveness half had no roots, so it verified nothing: %r"
            % (report,))


class TestDumpedGraphProperties(unittest.TestCase):
    """Structural invariants of the live graph, read through ``dump_all_graphs``.

    These run in-process and touch no flags, because they are true of whatever
    the process happens to be holding: prior state changes the graph but not the
    properties. That is what makes them safe here while the liveness sweep needs
    a child.
    """

    def _graph(self):
        held = [jt.array([1.0, 2.0, 3.0]) for _ in range(3)]
        left = held[0] * held[1]
        right = held[1] + held[2]
        joined = (left * right).sum()
        jt.sync([joined])
        graphs = jt.dump_all_graphs()
        return graphs, (held, left, right, joined)

    def test_edges_are_symmetric(self):
        """``j`` is an input of ``i`` exactly as often as ``i`` is an output of ``j``.

        The two tables are built in one pass over the same edges, so a mismatch
        means an edge was recorded in one direction only -- and every traversal
        in the core walks one direction or the other, so half of them would miss
        it.
        """
        graphs, _keep = self._graph()
        forward, backward = [], []
        for index, inputs in enumerate(graphs.inputs):
            for other in inputs:
                forward.append((other, index))
        for index, outputs in enumerate(graphs.outputs):
            for other in outputs:
                backward.append((index, other))
        self.assertEqual(sorted(forward), sorted(backward))

    def test_indices_are_in_range_and_tables_agree_in_length(self):
        graphs, _keep = self._graph()
        count = len(graphs.nodes_info)
        self.assertGreater(count, 0)
        self.assertEqual(len(graphs.inputs), count)
        self.assertEqual(len(graphs.outputs), count)
        for table in (graphs.inputs, graphs.outputs):
            for index, neighbours in enumerate(table):
                for other in neighbours:
                    self.assertGreaterEqual(other, 0)
                    self.assertLess(other, count)
                    # A node that is its own input is a one-node cycle, which
                    # the topological sorts below would spin on forever.
                    self.assertNotEqual(other, index)

    def test_the_graph_is_acyclic(self):
        """Every traversal in the core assumes this; nothing asserted it.

        ``count_fuse`` states the assumption -- it dequeues every op exactly
        once and fails if it cannot -- but only for one execution batch. This is
        the same claim for the whole live graph.
        """
        graphs, _keep = self._graph()
        count = len(graphs.nodes_info)
        remaining = [len(inputs) for inputs in graphs.inputs]
        queue = [i for i in range(count) if not remaining[i]]
        seen = 0
        while queue:
            node = queue.pop()
            seen += 1
            for other in graphs.outputs[node]:
                remaining[other] -= 1
                if not remaining[other]:
                    queue.append(other)
        self.assertEqual(
            seen, count,
            "%d of %d nodes are on a cycle" % (count - seen, count))

    def test_every_held_var_is_in_the_dump(self):
        """``hold_vars`` heads the walk, so losing one loses its whole subgraph."""
        graphs, _keep = self._graph()
        self.assertEqual(
            len(graphs.hold_vars), jt.introspection.counters.held_vars)
        self.assertLessEqual(len(graphs.hold_vars), len(graphs.nodes_info))


if __name__ == "__main__":
    unittest.main()
