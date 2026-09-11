"""Basic Torch API numerical precision and device-memory contracts.

Two properties that a passing API smoke does not cover:

*Precision.* An operation that runs and returns a plausibly shaped tensor can
still be computed at the wrong width. Every case here is checked against an
independent NumPy computation rather than against another Jittor result, and
the float64 cases are chosen so that silently computing them in float32 fails.

*Device memory.* An operation may be correct and still allocate on the wrong
side. ``Tensor.cpu()`` exists to release device memory; if its destination
buffer is allocated on the device, peak device usage doubles and the call
fails for any tensor larger than half of free VRAM -- the operation whose
purpose is to free device memory becomes the one that cannot run. That is not
visible in the numbers, only in the accounting, which is why it needs its own
contract here.
"""

import functools
import unittest

import numpy as np
import torch


def _cuda_available():
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def requires_cuda(test):
    """Probe CUDA when the test runs, after collection is complete."""
    if isinstance(test, type):
        original = getattr(test, "setUp", None)

        def setUp(self):
            if not _cuda_available():
                self.skipTest("cuda is required for device memory contracts")
            if original is not None:
                original(self)

        test.setUp = setUp
        return test

    @functools.wraps(test)
    def wrapped(self, *args, **kwargs):
        if not _cuda_available():
            self.skipTest("cuda is required for device memory contracts")
        return test(self, *args, **kwargs)

    return wrapped


class TestBasicPrecision(unittest.TestCase):
    """Basic operations agree with an independent NumPy reference."""

    def _ref(self, tensor):
        return np.asarray(tensor.detach().cpu().numpy())

    def test_float32_arithmetic_matches_numpy(self):
        rng = np.random.RandomState(0)
        a = rng.rand(64, 32).astype("float32")
        b = rng.rand(64, 32).astype("float32")
        ta, tb = torch.tensor(a), torch.tensor(b)
        for name, got, expected in (
            ("add", ta + tb, a + b),
            ("sub", ta - tb, a - b),
            ("mul", ta * tb, a * b),
            ("div", ta / (tb + 1.0), a / (b + 1.0)),
            ("pow", ta ** 2, a ** 2),
        ):
            with self.subTest(op=name):
                np.testing.assert_allclose(
                    self._ref(got), expected, rtol=1e-6, atol=1e-6)

    def test_matmul_and_bmm_match_numpy(self):
        rng = np.random.RandomState(1)
        a = rng.rand(32, 48).astype("float32")
        b = rng.rand(48, 16).astype("float32")
        np.testing.assert_allclose(
            self._ref(torch.tensor(a) @ torch.tensor(b)), a @ b,
            rtol=1e-5, atol=1e-5)
        x = rng.rand(4, 8, 12).astype("float32")
        y = rng.rand(4, 12, 6).astype("float32")
        np.testing.assert_allclose(
            self._ref(torch.bmm(torch.tensor(x), torch.tensor(y))),
            np.matmul(x, y), rtol=1e-5, atol=1e-5)

    def test_reductions_match_numpy_over_each_axis(self):
        rng = np.random.RandomState(2)
        a = rng.rand(16, 24, 8).astype("float32")
        t = torch.tensor(a)
        for axis in (0, 1, 2):
            with self.subTest(axis=axis):
                np.testing.assert_allclose(
                    self._ref(t.sum(dim=axis)), a.sum(axis=axis),
                    rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(
                    self._ref(t.mean(dim=axis)), a.mean(axis=axis),
                    rtol=1e-5, atol=1e-5)
                np.testing.assert_array_equal(
                    self._ref(t.argmax(dim=axis)), a.argmax(axis=axis))

    def test_float64_is_not_silently_computed_in_float32(self):
        """A float32 evaluation of these cases is wrong by ~1e-8, not ~1e-16."""
        t = torch.tensor(1.0, dtype=torch.float64) / torch.tensor(
            3.0, dtype=torch.float64)
        self.assertEqual(t.dtype, torch.float64)
        self.assertLess(abs(float(t.item()) - (1.0 / 3.0)), 1e-15)

        # 1 + 2^-40 is representable in float64 and lost entirely in float32.
        small = torch.tensor(2.0, dtype=torch.float64) ** -40
        one_plus = torch.tensor(1.0, dtype=torch.float64) + small
        self.assertGreater(float(one_plus.item()) - 1.0, 0.0)

        rng = np.random.RandomState(3)
        a = rng.rand(128, 128)
        got = self._ref(torch.tensor(a, dtype=torch.float64) @ torch.tensor(
            a, dtype=torch.float64))
        self.assertEqual(got.dtype, np.float64)
        np.testing.assert_allclose(got, a @ a, rtol=1e-12, atol=1e-12)

    def test_dtype_and_shape_operations_preserve_values(self):
        rng = np.random.RandomState(4)
        a = rng.rand(6, 10).astype("float32")
        t = torch.tensor(a)
        np.testing.assert_allclose(self._ref(t.reshape(10, 6)), a.reshape(10, 6))
        np.testing.assert_allclose(self._ref(t.transpose(0, 1)), a.T)
        np.testing.assert_allclose(self._ref(t[2:5, 3:7]), a[2:5, 3:7])
        np.testing.assert_allclose(
            self._ref(torch.cat([t, t], dim=0)), np.concatenate([a, a], 0))
        self.assertEqual(t.to(torch.float64).dtype, torch.float64)
        self.assertEqual(t.to(torch.int64).dtype, torch.int64)
        np.testing.assert_array_equal(
            self._ref(t.to(torch.int64)), a.astype("int64"))

    def test_autograd_gradients_match_the_analytic_value(self):
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = (x * x * 3.0).sum()
        y.backward()
        # d/dx 3x^2 = 6x
        np.testing.assert_allclose(
            self._ref(x.grad), np.array([6.0, 12.0, 18.0]),
            rtol=1e-6, atol=1e-6)


@requires_cuda
class TestDeviceMemory(unittest.TestCase):
    """Device-memory accounting contracts for the basic transfer APIs."""

    #: Large enough that a duplicated allocation is unambiguous against
    #: allocator noise, small enough to be safe on a shared card.
    ELEMENTS = 64 * 1024 * 1024          # 256 MiB of float32
    NBYTES = ELEMENTS * 4

    def _allocated(self):
        torch.cuda.synchronize()
        return int(torch.cuda.memory_allocated())

    def _materialize(self, tensor):
        """Force the lazy graph to actually allocate ``tensor`` on the device.

        Jittor factories are lazy and ``torch.cuda.synchronize()`` only flushes
        work that something has demanded, so a freshly created tensor is not
        allocated yet and ``memory_allocated()`` does not count it. Reading a
        full reduction is the cheapest demand that touches every element.
        """
        float(tensor.sum().item())
        torch.cuda.synchronize()
        return tensor

    def test_allocation_is_reported(self):
        torch.cuda.empty_cache()
        before = self._allocated()
        t = self._materialize(torch.ones(self.ELEMENTS, device="cuda"))
        held = self._allocated()
        self.assertGreaterEqual(
            held - before, self.NBYTES * 0.9,
            "allocating %d bytes was not reported by memory_allocated()"
            % self.NBYTES)
        del t

    def test_explicit_release_returns_device_memory(self):
        """Release is available; ``empty_cache()`` is deliberately not it.

        ``torch.cuda.empty_cache()`` defaults to a hint that does nothing,
        because TRELLIS calls it inside its inference path where a collection
        costs seconds; the opt-in is ``JITTOR_TORCH_CUDA_EMPTY_CACHE=gc``.
        This asserts that the underlying release actually works, so a future
        change to that default is a decision about cost rather than a guess
        about whether anything is reclaimable.
        """
        import gc as _gc
        import jittor as jt

        # Start from a collected state: another test's freed tensor would
        # otherwise still be counted, and this allocation would reuse it.
        _gc.collect()
        jt.gc()
        baseline = self._allocated()
        t = self._materialize(torch.ones(self.ELEMENTS, device="cuda"))
        self.assertGreaterEqual(self._allocated() - baseline, self.NBYTES * 0.9)
        del t
        _gc.collect()
        jt.gc()
        self.assertLess(
            self._allocated() - baseline, self.NBYTES * 0.5,
            "device memory was not reclaimed by an explicit collection")

    def test_default_empty_cache_is_a_hint_that_does_not_grow_memory(self):
        t = self._materialize(torch.ones(1024 * 1024, device="cuda"))
        before = self._allocated()
        torch.cuda.empty_cache()
        self.assertLessEqual(self._allocated(), before)
        del t

    def test_moving_to_cpu_does_not_allocate_another_device_buffer(self):
        """``.cpu()`` must not need a second device-side buffer.

        A device-to-host copy whose destination is allocated on the device
        doubles peak device memory, which makes any tensor larger than half of
        free VRAM impossible to move off the card.
        """
        torch.cuda.empty_cache()
        # Materialized first: otherwise the growth measured below would be
        # the source tensor being allocated, not a duplicated destination.
        t = self._materialize(torch.ones(self.ELEMENTS, device="cuda"))
        before = self._allocated()
        host = t.cpu()
        torch.cuda.synchronize()
        after = self._allocated()
        self.assertEqual(host.device.type, "cpu")
        self.assertLess(
            after - before, self.NBYTES * 0.5,
            "moving %d bytes to the host grew device memory by %d bytes; the "
            "copy destination is being allocated on the device"
            % (self.NBYTES, after - before))

    def test_repeated_transfers_do_not_grow_device_memory(self):
        torch.cuda.empty_cache()
        t = self._materialize(torch.ones(1024 * 1024, device="cuda"))
        baseline = self._allocated()
        for _ in range(12):
            t.cpu()
        torch.cuda.empty_cache()
        grown = self._allocated() - baseline
        self.assertLess(
            grown, 8 * 1024 * 1024,
            "twelve host transfers leaked %d bytes of device memory" % grown)


if __name__ == "__main__":
    unittest.main()
