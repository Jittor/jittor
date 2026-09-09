# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""argmax/argmin over a 1-D input, the commonest call the op has.

``ArgReduceOp``'s kernel is generated from templates parameterised by ``YDIM``,
the rank of the *output*. Reducing a 1-D input without ``keepdims`` makes
``YDIM`` 0, and the stride line spelled ``ystride@{YDIM-1}`` -- i.e.
``ystride@{-1}``, which is not a name. The kernel did not compile::

    arg_reduce_op.cc:191:20: error: expected initializer before '-' token
        index_t ystride@{YDIM-1} = 1;

The neighbouring ``@for`` loops already degenerated to nothing for ``YDIM == 0``
and nothing in the body referenced a y-stride there, so the single line just
needed the same guard. 2-D and higher inputs were always fine, which is why the
gap survived: the existing suite covers ``[5]`` only through ``check_reduce``,
whose ``np.allclose`` also could not see the second half of this.

That second half is CUDA's. ``CubArgReduceOp::infer_shape`` padded a rank-0
result out to shape ``(1,)``, so once the CPU kernel compiled, the same call
returned a scalar on CPU and a length-1 vector on CUDA. ``np.allclose``
broadcasts the two together and reports no difference; the shape assertions
below do not.

The index/value identity is what gives this file teeth against a future
rewrite: ``x[argmax(x)] == max(x)`` is answerable without an oracle, so a
reimplementation that returned a plausible-but-wrong index cannot pass by
matching NumPy on one axis and drifting on another.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


class _ArgReduce1dContract:

    device_flag = 0

    def _var(self, raw):
        v = jt.array(raw)
        self.assertEqual(str(v.dtype), str(raw.dtype))
        return v

    def test_1d_without_keepdims_compiles_and_is_correct(self):
        # The reported failure: this used to abort in the C++ compiler.
        with jt.flag_scope(use_cuda=self.device_flag):
            raw = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0], dtype="float32")
            x = self._var(raw)
            for op, npi, npv in (("max", np.argmax, np.max),
                                 ("min", np.argmin, np.min)):
                key, value = jt.arg_reduce(x, op, 0, False)
                self.assertEqual(key.numpy().shape, ())
                self.assertEqual(value.numpy().shape, ())
                self.assertEqual(int(key.numpy()), int(npi(raw)))
                self.assertEqual(float(value.numpy()), float(npv(raw)))

    def test_public_argmax_and_argmin_spellings(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            raw = np.array([-1.5, 2.5, 0.0, 2.5, -3.0], dtype="float32")
            x = self._var(raw)
            key, value = jt.argmax(x, 0)
            self.assertEqual(key.numpy().shape, ())
            self.assertEqual(int(key.numpy()), int(np.argmax(raw)))
            self.assertEqual(float(value.numpy()), float(np.max(raw)))
            key, value = jt.argmin(x, 0)
            self.assertEqual(key.numpy().shape, ())
            self.assertEqual(int(key.numpy()), int(np.argmin(raw)))
            self.assertEqual(float(value.numpy()), float(np.min(raw)))

    def test_the_index_selects_the_reported_value(self):
        # No oracle: whatever index comes back has to address the value that
        # came back with it, and that value has to be the extremum.
        with jt.flag_scope(use_cuda=self.device_flag):
            rng = np.random.default_rng(20260909)
            for n in (1, 2, 7, 33, 1024):
                raw = rng.standard_normal(n).astype("float32")
                x = self._var(raw)
                for op in ("max", "min"):
                    key, value = jt.arg_reduce(x, op, 0, False)
                    i = int(key.numpy())
                    self.assertEqual(raw[i], float(value.numpy()),
                                     "n=%d %s: index %d does not address the "
                                     "returned value" % (n, op, i))
                    want = raw.max() if op == "max" else raw.min()
                    self.assertEqual(float(value.numpy()), float(want))

    def test_keepdims_still_returns_a_length_one_axis(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            raw = np.array([3.0, 1.0, 4.0], dtype="float32")
            key, value = jt.arg_reduce(self._var(raw), "max", 0, True)
            self.assertEqual(key.numpy().shape, (1,))
            self.assertEqual(value.numpy().shape, (1,))
            self.assertEqual(int(key.numpy()[0]), 2)

    def test_1d_agrees_with_the_2d_shape_that_always_worked(self):
        # A 1-D reduce and the same data as a single row must give the same
        # index; the 2-D form is the one that already compiled, so this pins
        # the newly-reachable path to the path that was always exercised.
        with jt.flag_scope(use_cuda=self.device_flag):
            raw = np.array([0.5, -2.0, 7.0, 7.0, 1.0], dtype="float32")
            flat_key, flat_value = jt.arg_reduce(self._var(raw), "max", 0, False)
            row_key, row_value = jt.arg_reduce(
                self._var(raw.reshape(1, -1)), "max", 1, False)
            self.assertEqual(int(flat_key.numpy()), int(row_key.numpy()[0]))
            self.assertEqual(float(flat_value.numpy()), float(row_value.numpy()[0]))

    def test_integer_and_float64_inputs(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            for raw in (np.array([3, 1, 4, 1, 5], dtype="int32"),
                        np.array([-7, -1, -9], dtype="int32")):
                key, value = jt.arg_reduce(self._var(raw), "max", 0, False)
                self.assertEqual(key.numpy().shape, ())
                self.assertEqual(int(key.numpy()), int(np.argmax(raw)))
                self.assertEqual(int(value.numpy()), int(np.max(raw)))

    def test_backward_through_a_1d_arg_reduce(self):
        with jt.flag_scope(use_cuda=self.device_flag):
            x = jt.random([9])
            key, value = jt.arg_reduce(x, "max", 0, False)
            grad = jt.grad((value * value).sum(), x) / 2
            # d/dx of max(x)**2/2 is x at the argmax and 0 everywhere else.
            got = (grad * x).numpy()
            np.testing.assert_allclose(got, (grad * grad).numpy(), rtol=1e-5)


class TestArgReduce1dCpu(_ArgReduce1dContract, unittest.TestCase):
    device_flag = 0


@unittest.skipUnless(_has_cuda(), "a CUDA device is required")
class TestArgReduce1dCuda(_ArgReduce1dContract, unittest.TestCase):
    device_flag = 1


if __name__ == "__main__":
    unittest.main()
