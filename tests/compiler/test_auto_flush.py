# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``auto_flush_ops`` pipelines graph construction with execution.

Once that many operators have been built since the last execution, the
pending graph is launched without waiting for the device. The tests pin
down what must not change: results, gradients through a graph that was
partly executed while it was still being built, dead-code elimination for
values nobody kept, and errors that still reach the caller.
"""
import unittest

import numpy as np

import jittor as jt


def _chain(x, n):
    for _ in range(n):
        x = x * 1.001 + 0.5
    return x


@unittest.skipIf(not jt.has_cuda, "No cuda found")
class TestAutoFlush(unittest.TestCase):
    """The pipeline only acts on CUDA: CPU kernels run synchronously on the
    calling thread, so launching early there could only cost fusion."""

    def setUp(self):
        self._saved = (jt.flags.auto_flush_ops, jt.flags.use_cuda)
        jt.flags.use_cuda = 1
        # A failed flush suspends the pipeline until a sync succeeds; start
        # every case from a clean, unsuspended executor.
        jt.sync_all()

    def tearDown(self):
        jt.flags.auto_flush_ops, jt.flags.use_cuda = self._saved

    def test_results_match_lazy_execution(self):
        a = np.random.RandomState(0).randn(64, 64).astype("float32")
        jt.flags.auto_flush_ops = 0
        reference = _chain(jt.array(a), 300).numpy()
        jt.flags.auto_flush_ops = 16
        pipelined = _chain(jt.array(a), 300).numpy()
        np.testing.assert_allclose(pipelined, reference, rtol=1e-6, atol=0)

    def test_pending_graph_is_launched_while_building(self):
        seen = []

        def forward(np, data):
            seen.append(1)
            np.copyto(data["outputs"][0], data["inputs"][0] + 1)

        x = jt.array(np.ones(4, "float32"))
        jt.flags.auto_flush_ops = 0
        lazy = [jt.numpy_code([4], "float32", [x], forward) for _ in range(20)]
        self.assertEqual(seen, [])
        jt.flags.auto_flush_ops = 8
        pipelined = [jt.numpy_code([4], "float32", [x], forward) for _ in range(20)]
        # Some of the graph ran before anyone asked for a value.
        self.assertGreater(len(seen), 0)
        jt.sync_all()
        self.assertEqual(len(seen), 40)
        np.testing.assert_array_equal(pipelined[-1].numpy(), np.full(4, 2.0))
        np.testing.assert_array_equal(lazy[0].numpy(), np.full(4, 2.0))

    def test_gradient_through_partly_executed_graph(self):
        jt.flags.auto_flush_ops = 8
        x = jt.array(np.random.RandomState(1).randn(16).astype("float32"))
        y = _chain(x, 50).sum()
        grad = jt.grad(y, x).numpy()
        np.testing.assert_allclose(grad, np.full(16, 1.001 ** 50), rtol=1e-5)

    def test_value_nobody_kept_is_not_computed(self):
        seen = []

        def forward(np, data):
            seen.append(1)
            np.copyto(data["outputs"][0], data["inputs"][0])

        jt.flags.auto_flush_ops = 4
        x = jt.array(np.ones(4, "float32"))
        dropped = jt.numpy_code([4], "float32", [x], forward)
        del dropped
        kept = [x + i for i in range(20)]
        jt.sync_all()
        self.assertEqual(seen, [])
        self.assertEqual(kept[-1].numpy()[0], 20.0)

    def test_execution_error_still_raised(self):
        jt.flags.auto_flush_ops = 4
        x = jt.array(np.ones(4, "float32"))
        bad = jt.code([4], "float32", [x], cpu_src="throw std::runtime_error(\"auto flush error\");")
        with self.assertRaises(Exception):
            kept = [bad + i for i in range(20)]
            jt.sync_all()

    def test_function_tape_survives_flush(self):
        # Flush on every operator, including between the tape ops that
        # jt.Function wires together after the forward has been built.
        class Scale(jt.Function):
            def execute(self, x, y):
                self.y = y
                return x * y, x + y

            def grad(self, g0, g1):
                return g0 * self.y + g1, None

        jt.flags.auto_flush_ops = 1
        x = jt.array(np.arange(8, dtype="float32"))
        y = jt.array(np.full(8, 3.0, "float32"))
        a, b = Scale.apply(x, y)
        loss = (a * 2 + b).sum()
        grad = jt.grad(loss, x).numpy()
        np.testing.assert_allclose(grad, np.full(8, 7.0))

    def test_larger_results_match_lazy_execution(self):
        a = np.random.RandomState(2).randn(256, 256).astype("float32")
        jt.flags.auto_flush_ops = 0
        reference = _chain(jt.array(a), 300).numpy()
        jt.flags.auto_flush_ops = 16
        pipelined = _chain(jt.array(a), 300).numpy()
        np.testing.assert_allclose(pipelined, reference, rtol=1e-6, atol=0)


@unittest.skipIf(not jt.compiler.has_acl, "No ACL found")
class TestAutoFlushBackendBoundary(unittest.TestCase):

    def test_acl_does_not_enable_cuda_pipeline(self):
        seen = []

        def forward(np, data):
            seen.append(1)
            np.copyto(data["outputs"][0], data["inputs"][0] + 1)

        with jt.flag_scope(use_acl=1, use_cuda=1, auto_flush_ops=1):
            x = jt.ones(4, dtype="float32")
            x.sync()
            self.assertEqual(x.location(), "device")

            # numpy_code is an intentional host-side observer: if the CUDA-only
            # pipeline leaks into ACL, these callbacks run before explicit sync.
            outputs = [
                jt.numpy_code([4], "float32", [x], forward)
                for _ in range(20)
            ]
            self.assertEqual(seen, [])
            jt.sync_all()
            self.assertEqual(len(seen), 20)
            np.testing.assert_array_equal(
                outputs[-1].numpy(), np.full(4, 2.0, dtype="float32"))


if __name__ == "__main__":
    unittest.main()
