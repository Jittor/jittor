# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The graph-build phase probes of task 3.21, and what they may not do.

The probes are compiled in only by ``JT_GRAPH_BUILD_PROFILE=1``, so most of
this file does not run in an ordinary build.  That is the point rather than a
gap: a probe under ``JitKey::reserve`` -- which is inlined into every JIT
kernel -- may not cost even a predictable branch in the builds the gates
measure.  ``src/utils/graph_build_profile.h`` says why.

To run the skipped half::

    JT_GRAPH_BUILD_PROFILE=1 JITTOR_HOME=... TMPDIR=... nvcc_path=... \\
        python -m pytest tests/compiler/test_graph_build_profile.py -q

What still runs in every build is the part that matters for reading a report:
that an unprobed core reports *nothing* rather than reporting zeros.  A phase
table of zeros and a phase table that was never filled are the same picture,
and a harness that cannot tell them apart will publish "the jit key costs 0".
"""
import unittest

import jittor as jt


PROBED = bool(jt.core.graph_build_profile_enabled())


class TestProbeAvailability(unittest.TestCase):
    """Runs in both builds."""

    def test_the_flag_and_the_phase_list_agree(self):
        """Neither may claim more than the other: an empty list from a build
        that says it is probed, or a filled one from a build that says it is
        not, is a reader that cannot be trusted either way."""
        phases = list(jt.core.graph_build_profile_phases())
        counts = list(jt.core.graph_build_profile_counts())
        nanoseconds = list(jt.core.graph_build_profile_nanoseconds())
        self.assertEqual(len(counts), len(phases))
        self.assertEqual(len(nanoseconds), len(phases))
        self.assertEqual(bool(phases), PROBED)

    def test_an_unprobed_build_reports_nothing_rather_than_zeros(self):
        if PROBED:
            raise unittest.SkipTest("this build has the probes")
        self.assertEqual(list(jt.core.graph_build_profile_phases()), [])
        self.assertEqual(list(jt.core.graph_build_profile_counts()), [])
        self.assertEqual(list(jt.core.graph_build_profile_nanoseconds()), [])
        # And resetting an absent table is not an error.
        jt.core.graph_build_profile_reset()

    def test_the_three_named_phases_of_the_task_exist_when_probed(self):
        if not PROBED:
            raise unittest.SkipTest("needs JT_GRAPH_BUILD_PROFILE=1")
        phases = set(jt.core.graph_build_profile_phases())
        self.assertLessEqual({"pyjt_entry", "edge_table", "jit_key"}, phases)


@unittest.skipUnless(PROBED, "needs a core built with JT_GRAPH_BUILD_PROFILE=1")
class TestPhaseAttribution(unittest.TestCase):
    """Does the split put a cost where the cost actually is?

    Each test moves exactly one thing and checks that the phase that owns it
    moves with it while the others hold still.  A probe that merely produced
    plausible-looking numbers would pass none of these.
    """

    def setUp(self):
        self.previous_flush = jt.flags.auto_flush_ops
        # Nothing may be submitted while a graph is being built, or the window
        # contains execution and the phases stop meaning build cost.
        jt.flags.auto_flush_ops = 0
        jt.sync_all(True)

    def tearDown(self):
        jt.sync_all(True)
        jt.flags.auto_flush_ops = self.previous_flush

    def _measure(self, build):
        build()          # let any first-time compile happen outside the window
        jt.sync_all(True)
        jt.core.graph_build_profile_reset()
        out = build()
        nanoseconds = list(jt.core.graph_build_profile_nanoseconds())
        counts = list(jt.core.graph_build_profile_counts())
        del out
        jt.sync_all(True)
        phases = list(jt.core.graph_build_profile_phases())
        return (dict(zip(phases, counts)), dict(zip(phases, nanoseconds)))

    def test_building_a_graph_assembles_no_jit_key(self):
        """The plan for 3.21 names jit-key concatenation as one of three
        per-operator build costs. It is not one: a key is assembled by the
        executor, at run time, once per fused segment."""
        counts, _ = self._measure(lambda: [jt.zeros((4,)).abs() for _ in range(20)])
        self.assertEqual(counts["jit_key"], 0)
        self.assertEqual(counts["jit_key_write"], 0)
        self.assertGreater(counts["op_init"], 20)

    def test_executing_a_graph_does_assemble_jit_keys(self):
        x = jt.zeros((4,))

        def run():
            (x.abs() + 1.0).sync()

        counts, _ = self._measure(run)
        self.assertGreater(counts["jit_key"], 0)
        self.assertGreater(counts["jit_key_write"], 0)
        self.assertGreater(counts["jit_key_bytes"], 0)

    def test_a_longer_fused_segment_lands_on_the_jit_key_phase(self):
        """Same operator kind, same number of executor runs, longer key."""
        x = jt.zeros((4,))

        def chain(length):
            def run():
                value = x
                for _ in range(length):
                    value = value.abs() + 1.0
                value.sync()
            return run

        short_counts, short_ns = self._measure(chain(4))
        long_counts, long_ns = self._measure(chain(64))
        self.assertGreater(long_counts["jit_key_bytes"],
                           short_counts["jit_key_bytes"] * 4)
        self.assertGreater(long_ns["jit_key"], short_ns["jit_key"])

    def test_more_input_edges_land_on_the_edge_table_phase(self):
        """Operator count held fixed, edge count multiplied by 32.

        The control that says the edge phase is measuring edges: if it were
        really measuring "per operator" work its count would not move, and if
        the harness were charging the whole build to it, the operator-count
        phases would move too.
        """
        def fanin(width):
            sources = [jt.zeros((4,)) for _ in range(width)]
            jt.sync_all(True)

            def run():
                return [jt.code((4,), "float32", sources,
                                cpu_src="@out0(0) = 0;") for _ in range(40)]
            return run

        narrow_counts, narrow_ns = self._measure(fanin(2))
        wide_counts, wide_ns = self._measure(fanin(64))

        # The number of operators, and so of Op::init and output Vars, is the
        # same on both sides.
        self.assertEqual(wide_counts["op_init"], narrow_counts["op_init"])
        self.assertEqual(wide_counts["var_create"], narrow_counts["var_create"])
        # ...but the edge phase is charged for the extra edges.
        self.assertGreater(wide_ns["edge_table"], narrow_ns["edge_table"] * 4)
        # ...and the phases that own operator-count work are not.
        self.assertLess(wide_ns["op_init"], narrow_ns["op_init"] * 4)

    def test_more_operators_land_on_the_per_operator_phases(self):
        def many(count):
            def run():
                return [jt.zeros((4,)).abs() for _ in range(count)]
            return run

        few_counts, _ = self._measure(many(10))
        lots_counts, _ = self._measure(many(100))
        for phase in ("pyjt_entry", "op_init", "var_create", "edge_table"):
            self.assertGreater(lots_counts[phase], few_counts[phase] * 5, phase)

    def test_reset_clears_every_phase(self):
        self._measure(lambda: [jt.zeros((4,)).abs() for _ in range(10)])
        jt.core.graph_build_profile_reset()
        self.assertEqual(set(jt.core.graph_build_profile_counts()), {0})
        self.assertEqual(set(jt.core.graph_build_profile_nanoseconds()), {0})

    def test_the_phases_are_exclusive_so_they_can_be_added_up(self):
        """`edge_table` and the rest are entered from inside `pyjt_entry`. If
        the times were inclusive, adding the rows would count that work twice
        and a report could reach more than 100% of the wall time."""
        import time

        build = lambda: [jt.zeros((4,)).abs() for _ in range(200)]
        build()
        jt.sync_all(True)
        jt.core.graph_build_profile_reset()
        started = time.perf_counter()
        out = build()
        wall_ns = (time.perf_counter() - started) * 1e9
        total_ns = sum(jt.core.graph_build_profile_nanoseconds())
        del out
        jt.sync_all(True)
        self.assertGreater(total_ns, 0)
        self.assertLess(total_ns, wall_ns)


if __name__ == "__main__":
    unittest.main()
