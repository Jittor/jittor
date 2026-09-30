"""An elementwise kernel writes its output into an input it reads for the last time.

`reuse_dying_inputs`: the output takes over the block of an input whose memory
would go right after the kernel, so the peak does not hold both, and the
kernel writes cache lines it has just read. What must not change is every
value -- above all when the input is *not* dying: a holder that recomputes an
intermediate from it later, a reader in the kernel at another index, a kept
graph that runs again. Flag value 2 checks, after every segment that reused
an input, that the input's memory really did go. Outputs under 256 KB are
left alone, so every tensor here is larger.
"""

import unittest

import numpy as np

import jittor as jt
from _helpers import capability as _test_capability
from _helpers.common import JittorTestCase
from _helpers.device_types import instantiate_device_type_tests

_HAS_CUDA = _test_capability.check_accelerator('cuda', backend=jt).enabled


def _data(seed, *shape):
    return np.random.RandomState(seed).randn(*shape).astype("float32")


class TestReuseDyingInputs(JittorTestCase):

    def test_values_with_and_without_reuse(self, device):
        a, b = _data(0, 512, 48), _data(1, 48, 256)

        def run(flag):
            with jt.flag_scope(reuse_dying_inputs=flag):
                x, w = jt.array(a), jt.array(b)
                m = jt.matmul(x, w)           # dies in the elementwise kernel
                y = (m * 3.0 + 1.0).exp() * 0.01
                return y.numpy()
        ref = run(0)
        for flag in (1, 2):
            np.testing.assert_allclose(run(flag), ref, rtol=1e-5, atol=1e-6)

    def test_an_intermediate_a_holder_recomputes_later(self, device):
        # `h` is computed inside the kernel that writes `out` and is not
        # stored; its holder makes a later batch recompute it from `m`, so the
        # kernel must not write over `m` although this batch reads it last.
        a, b = _data(2, 512, 32), _data(3, 32, 256)
        with jt.flag_scope(reuse_dying_inputs=2):
            x, w = jt.array(a), jt.array(b)
            m = jt.matmul(x, w)
            h = m + 1.0
            out = h * 2.0
            del m
            out.sync()
            expect = a @ b + 1.0
            np.testing.assert_allclose(h.numpy(), expect, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(out.numpy(), expect * 2, rtol=1e-4, atol=1e-4)

    def test_an_input_also_read_at_another_index(self, device):
        a = _data(4, 512, 512) * 0.1
        with jt.flag_scope(reuse_dying_inputs=2):
            m = jt.matmul(jt.array(a), jt.array(a))
            y = m * 2.0 + m.transpose()
            del m
            ref = a @ a
            np.testing.assert_allclose(y.numpy(), ref * 2 + ref.T, rtol=1e-4, atol=1e-3)

    def test_a_kept_graph_replayed(self, device):
        a, b = _data(5, 512, 16), _data(6, 16, 512)

        def step(x, w):
            m = jt.matmul(x, w)
            return [(m * 0.5 - 1.0).sigmoid()]
        with jt.flag_scope(reuse_dying_inputs=2):
            capture = jt.capture_step(step)
            w = jt.array(b)
            for i in range(4):
                x = jt.array(a + i)
                got = capture(x, w)[0].numpy()
                ref = 1 / (1 + np.exp(-((a + i) @ b * 0.5 - 1.0)))
                np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


instantiate_device_type_tests(TestReuseDyingInputs, globals())


@unittest.skipUnless(_HAS_CUDA, "the peak and the launch order are read off the device")
class TestReuseDyingInputsOnDevice(JittorTestCase):

    def setUp(self):
        super().setUp()
        from contextlib import ExitStack
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(jt.flag_scope(use_cuda=1))

    def _peak(self, flag):
        a = _data(7, 512, 512)
        # Gradient-free, so the product is not kept for a backward either.
        with jt.flag_scope(reuse_dying_inputs=flag), jt.no_grad():
            x = jt.array(a)
            x.sync()
            jt.sync_all(True)
            jt.gc()
            base = jt.core.device_memory_used(0)
            jt.core.reset_device_memory_peak(0)
            m = jt.matmul(x, x)
            y = m * 2.0 + 1.0
            del m
            y.sync()
            jt.sync_all(True)
            peak = jt.core.device_memory_peak(0) - base
            np.testing.assert_allclose(y.numpy(), (a @ a) * 2 + 1, rtol=1e-4, atol=1e-2)
            return peak

    def test_the_output_takes_over_the_dying_input(self):
        size = 512 * 512 * 4
        without, with_reuse = self._peak(0), self._peak(1)
        self.assertGreaterEqual(without - with_reuse, size)

    def test_the_last_reader_of_a_fresh_value_runs_next(self):
        # `g` is made first and read only by the elementwise kernel; `w` is an
        # independent GEMM created before that kernel. Creation order would
        # run the second GEMM between `g` and its reader.
        a = _data(8, 256, 256)
        x, y = jt.array(a), jt.array(a + 1)
        jt.sync([x, y])
        jt.sync_all(True)

        def build():
            g = jt.matmul(x, y)
            w = jt.matmul(y, x)
            e = g * 2.0 + 1.0
            return [e, w]
        jt.sync(build())
        jt.sync_all(True)
        with jt.profile() as p:
            e, w = build()
            jt.sync([e, w])
            jt.sync_all(True)
        names = [dict(k)["name"] if not isinstance(k, dict) else k["name"]
                 for k in p.result.kernel_records]
        elementwise = [i for i, n in enumerate(names) if "func_" in n]
        gemms = [i for i, n in enumerate(names) if "gemm" in n and "splitK" not in n]
        self.assertTrue(len(elementwise) == 1 and len(gemms) == 2, names)
        self.assertTrue(gemms[0] < elementwise[0] < gemms[1], names)
        np.testing.assert_allclose(e.numpy(), (a @ (a + 1)) * 2 + 1, rtol=1e-4, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
