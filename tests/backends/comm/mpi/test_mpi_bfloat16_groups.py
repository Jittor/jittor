"""Real two-rank CPU MPI transport, arithmetic and communicator gradients.

Launch under mpirun with a private, already warmed cache per rank. The BF16
bit-transport cases deliberately include NaN payloads; arithmetic is checked
separately, including round-to-even ties and subnormal sums.
"""
import unittest

import numpy as np

import jittor as jt
from jittor._runtime import forbid_backend_fallbacks
from jittor.distributed import process_group as dist


def _from_bits(bits):
    raw = np.asarray(bits, dtype="uint16").view("int16")
    storage = jt.array(raw, dtype="int16").stop_grad()
    return jt.reinterpret_view(storage, tuple(raw.shape), "bfloat16").stop_grad()


def _bits(value):
    return jt.reinterpret_view(value, tuple(value.shape), "int16").numpy().view("uint16")


@unittest.skipUnless(jt.in_mpi and jt.world_size == 2 and not jt.flags.use_cuda,
                     "no inside mpirun with two CPU ranks")
class TestMpiBfloat16Groups(unittest.TestCase):
    def setUp(self):
        self.fallbacks = jt.core.backend_fallback_count()
        self.policy = forbid_backend_fallbacks()
        self.policy.__enter__()
        self.addCleanup(self.policy.__exit__, None, None, None)
        self.group = dist.new_group([1, 0])

    def tearDown(self):
        self.assertEqual(jt.core.backend_fallback_count(), self.fallbacks)

    def _cpu(self, value, dtype):
        value.sync()
        self.assertEqual(value.location(), "cpu")
        self.assertEqual(str(value.dtype), dtype)

    def test_broadcast_and_gather_preserve_raw_bfloat16_bits(self):
        patterns = np.asarray([
            0x0000, 0x8000, 0x0001, 0x007f, 0x0080, 0x3f80,
            0x7f7f, 0x7f80, 0xff80, 0x7fc1, 0xffc2, 0x7fa1], dtype="uint16")
        local = np.roll(patterns, jt.rank)
        value = _from_bits(local)
        for root in (0, 1):
            output = self.group.broadcast(value, src=root)
            self._cpu(output, "bfloat16")
            np.testing.assert_array_equal(_bits(output), np.roll(patterns, root))
            np.testing.assert_array_equal(_bits(value), local)
        output = self.group.all_gather(value)
        self._cpu(output, "bfloat16")
        np.testing.assert_array_equal(_bits(output), np.concatenate([
            np.roll(patterns, 1), patterns]))

    def test_bfloat16_sum_rounds_ties_and_preserves_subnormals(self):
        operands = ([0x3f80, 0x3f81, 0x0001, 0x7f80, 0xff80],
                    [0x3b80, 0x3b80, 0x0002, 0xff80, 0xff80])
        value = _from_bits(operands[jt.rank])
        output = self.group.all_reduce(value, "sum")
        self._cpu(output, "bfloat16")
        # 1+half-ulp rounds down; (1+ulp)+half-ulp rounds up to even.
        np.testing.assert_array_equal(_bits(output)[:3], [0x3f80, 0x3f82, 0x0003])
        observed = output.float32().numpy()
        self.assertTrue(np.isnan(observed[3]))
        self.assertTrue(np.isneginf(observed[4]))
        finite = jt.array([1.5 + jt.rank, -2.0 - jt.rank], dtype="bfloat16").stop_grad()
        mean = self.group.all_reduce(finite, "mean")
        self._cpu(mean, "bfloat16")
        np.testing.assert_array_equal(mean.float32().numpy(), [2., -2.5])

    def test_broadcast_and_gather_backward_use_the_group_rank_order(self):
        for operation in ("broadcast", "gather", "sum"):
            with self.subTest(operation=operation):
                value = jt.array([.5, 1.25], dtype="bfloat16")
                if operation == "broadcast":
                    output = self.group.broadcast(value, src=1)
                    expected = 3. if jt.rank == 1 else 0.
                elif operation == "gather":
                    output = self.group.all_gather(value)
                    expected = 3.
                else:
                    output = self.group.all_reduce(value, "sum")
                    expected = 3.
                gradient = jt.grad(output.float32().sum() * (jt.rank + 1), value)
                self._cpu(gradient, "bfloat16")
                np.testing.assert_array_equal(gradient.float32().numpy(), [expected, expected])

    def test_bfloat16_reduce_nonzero_group_root_and_backward(self):
        # This group is ordered [1, 0]: local root 1 is global rank 0.
        for operation, expected, grad in (("add", [2., 3.5], 1.),
                                          ("mean", [1., 1.75], .5)):
            with self.subTest(operation=operation):
                value = jt.array([.5 + jt.rank, 1.25 + jt.rank], dtype="bfloat16")
                output = jt.compile_extern.mpi_ops.mpi_reduce(
                    value, operation, 1, self.group._backend_handle)
                self._cpu(output, "bfloat16")
                np.testing.assert_array_equal(
                    output.float32().numpy(), expected if jt.rank == 0 else [0., 0.])
                gradient = jt.grad(output.float32().sum() * (jt.rank + 1), value)
                self._cpu(gradient, "bfloat16")
                np.testing.assert_array_equal(gradient.float32().numpy(), [grad, grad])

    def test_non_sum_composites_keep_dtype_and_reject_autograd(self):
        operands = (([200, 250, 3], [3, 2, 200]),
                    ([-2., 1.5, -.25], [-.5, -3., 2.]))
        for dtype in ("uint8", "float32", "bfloat16"):
            pair = operands[0] if dtype == "uint8" else operands[1]
            numpy_dtype = "uint8" if dtype == "uint8" else "float32"
            left, right = (np.asarray(v, dtype=numpy_dtype) for v in pair)
            value = jt.array(pair[jt.rank], dtype=dtype).stop_grad()
            for operation, expected in (("max", np.maximum(left, right)),
                                        ("min", np.minimum(left, right)),
                                        ("product", left * right)):
                output = self.group.all_reduce(value, operation)
                self._cpu(output, dtype)
                np.testing.assert_array_equal(output.float32().numpy(), expected)
        value = jt.array([1., 2.], dtype="float32")
        for operation in ("max", "min", "product"):
            with self.assertRaisesRegex(NotImplementedError, "autograd"):
                self.group.all_reduce(value, operation)

    def test_distinct_clones_and_singleton_membership(self):
        other = dist.new_group([1, 0])
        self.assertNotEqual(self.group._backend_handle, other._backend_handle)
        self.assertNotEqual(other._backend_handle, 0)
        value = jt.array([1. + jt.rank], dtype="float32").stop_grad()
        np.testing.assert_array_equal(other.all_reduce(value, "sum").numpy(), [3.])
        singleton = dist.new_group([1])
        self.assertEqual(singleton.rank(), 0 if jt.rank == 1 else -1)
        self.assertIs(singleton.all_reduce(value, "sum"), value)
        with self.assertRaisesRegex(ValueError, "outside the process group"):
            self.group.broadcast(value, src=3)


if __name__ == "__main__":
    unittest.main()
