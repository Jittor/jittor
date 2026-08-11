"""Torch-grade error-message clarity (Task #2: C++/CUDA-layer error clarity).

Locks the contract that common user mistakes raise CLEAR, actionable messages instead of
empty AssertionErrors, raw "not enough values to unpack", or a wall of g++ template errors.
Each test asserts on key substrings (not exact text) so wording tweaks don't break it.

These checks are raised at graph-build / codegen time and are backend-independent, so they
are exercised on CPU (the message is identical on CUDA).

Run:  python -m pytest tests/compat/torch/test_torch_compat_errors.py
"""
import unittest
import numpy as np
import jittor as jt


def _msg(fn):
    """Run fn, return (exc_type_name, full_message) — fails the test if it does NOT raise."""
    try:
        fn()
    except Exception as e:  # noqa
        return type(e).__name__, str(e)
    return None, ""


class TestTorchCompatErrors(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 0

    # ---- unsupported op x dtype (op_compiler expand_op_search) ----
    def test_unsupported_op_on_complex(self):
        a = np.array([1 + 2j, 3 - 4j], dtype="complex64")
        for op in ("exp", "log", "sin"):
            name, m = _msg(lambda op=op: getattr(jt, op)(jt.array(a)).numpy())
            self.assertIsNotNone(name, f"{op}(complex) should raise")
            self.assertIn("not supported for dtype", m, f"{op}: {m[:120]}")
            self.assertIn("complex64", m, f"{op}: {m[:120]}")

    # ---- Conv2d: channel mismatch (was an empty AssertionError) ----
    def test_conv2d_channel_mismatch(self):
        name, m = _msg(lambda: jt.nn.Conv2d(8, 16, 3)(jt.rand(1, 3, 32, 32)))
        self.assertIsNotNone(name, "channel mismatch should raise")
        self.assertIn("channels", m)
        self.assertIn("in_channels", m)
        # the actual numbers must be present so the user can see what went wrong
        self.assertIn("8", m)
        self.assertIn("3", m)

    # ---- Conv2d: wrong ndim (was "not enough values to unpack") ----
    def test_conv2d_wrong_ndim(self):
        name, m = _msg(lambda: jt.nn.Conv2d(3, 16, 3)(jt.rand(3, 32)))
        self.assertIsNotNone(name, "wrong-ndim should raise")
        self.assertIn("4-D", m)
        self.assertNotIn("unpack", m, "must not surface the raw unpack error")

    # ---- Conv2d: output collapses to <=0 (was an empty AssertionError) ----
    def test_conv2d_output_too_small(self):
        name, m = _msg(lambda: jt.nn.Conv2d(3, 16, 9)(jt.rand(1, 3, 4, 4)))
        self.assertIsNotNone(name, "too-small input should raise")
        self.assertIn("output size", m)

    # ---- bitwise/shift on float (was a g++ compile wall) ----
    def test_bitwise_on_float(self):
        for fn in (lambda: (jt.rand(3) & jt.rand(3)).numpy(),
                   lambda: (jt.rand(3).int32() << jt.rand(3)).numpy()):
            name, m = _msg(fn)
            self.assertIsNotNone(name, "bitwise/shift on float should raise")
            self.assertIn("integer or boolean", m, m[:160])
            self.assertNotIn("operator&", m, "must not surface raw g++ operator error")

    def test_bitwise_on_int_still_works(self):
        r = (jt.array([6, 3, 5]) & jt.array([4, 1, 1])).numpy()
        np.testing.assert_array_equal(np.asarray(r), [4, 1, 1])

    # ---- reduce over an out-of-range dim (now names the valid range) ----
    def test_reduce_dim_out_of_range(self):
        name, m = _msg(lambda: jt.array(np.ones((2, 3), "float32")).sum(dim=5).numpy())
        self.assertIsNotNone(name, "out-of-range reduce dim should raise")
        self.assertIn("out of range", m)
        self.assertIn("valid dims", m)

    # ---- already-clear messages we must not regress ----
    def test_binary_shape_mismatch_clear(self):
        name, m = _msg(lambda: (jt.array(np.ones((3, 4), "float32"))
                                + jt.array(np.ones((5, 6), "float32"))).numpy())
        self.assertIsNotNone(name)
        self.assertIn("Shape not match", m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
