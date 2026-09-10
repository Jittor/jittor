# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""An index that arrives in a Var must be checked against the dimension.

``x[99]`` on a length-5 tensor has always raised: a Python ``int`` index is
normalised and checked while the op is being built. ``x[jt.array([99])]`` took
a different path and was checked nowhere. It read past the buffer and returned
whatever was there -- ``0.0`` for a small overshoot, a segfault at ``1e8``
(KI-OPS-010). ``setitem`` shared the hole and was worse: it *wrote* past the
buffer, which is heap corruption that surfaces somewhere else entirely.

Both spellings reach the same two kernels, so the gap covered every operator
built on them -- ``take``, ``gather``, ``index_select``, and the embedding
lookup that is the hottest use of all of them. A wrong vocabulary id is the
single most common bug in real training code, and it silently trained on row
zero.

Why the checks live where they do
---------------------------------
The loop cannot raise from inside itself: on CPU it is an OpenMP region and on
CUDA it is a kernel. So the CPU kernel clamps the offending index, records it,
and the host raises once the loop is over -- the clamp is what keeps the read
inside the buffer until then. The device kernel has nobody to report to, so it
prints the index and traps, which is the bargain PyTorch makes for its own
device-side asserts.

That difference is why the CUDA half of this file runs in a subprocess. A trap
takes the CUDA context with it, so a second out-of-range case in the same
process would fail for a reason that has nothing to do with the code under
test. Running the legal indices in-process and the out-of-range one outside it
is not a workaround -- it is the only arrangement in which both assertions mean
what they say.
"""

from _helpers import capability as _test_capability

import os
import subprocess
import sys
import textwrap
import unittest

import numpy as np

import jittor as jt


def _has_cuda():
    return bool(_test_capability.check_accelerator('cuda', backend=jt).enabled)


#: A length-5 first dimension, indexed with values on either side of it.
OUT_OF_RANGE = (5, 99, -6, -99, 100000000)


class TestVarIndexBoundsCpu(unittest.TestCase):
    """Every operator that funnels into ``getitem`` with a Var index."""

    device_flag = 0

    def _x(self):
        return jt.array(np.arange(5, dtype="float32"))

    def _e(self):
        return jt.array(np.arange(20, dtype="float32").reshape(5, 4))

    def _raises(self, fn, what):
        with jt.flag_scope(use_cuda=self.device_flag):
            with self.assertRaises(RuntimeError, msg=(
                    "%s accepted an index outside the dimension. Before "
                    "KI-OPS-010 was fixed this returned 0.0 for a small "
                    "overshoot and segfaulted for a large one." % what)) as caught:
                fn()
        # The message has to name the offending index, not merely fail. A check
        # that says only "something was wrong" leaves the caller to bisect their
        # own data, which is the situation this replaces.
        self.assertIn("out of bounds", str(caught.exception),
                      "%s raised, but not about the index" % what)

    def test_getitem_rejects_every_out_of_range_var_index(self):
        for bad in OUT_OF_RANGE:
            with self.subTest(index=bad):
                self._raises(lambda: self._x()[jt.array([bad])].sync(),
                             "getitem(Var(%d))" % bad)

    def test_gather_rejects_out_of_range(self):
        self._raises(lambda: jt.gather(self._x(), 0, jt.array([9])).sync(),
                     "gather")

    def test_index_select_rejects_out_of_range(self):
        self._raises(lambda: self._x().index_select(0, jt.array([9])).sync(),
                     "index_select")

    def test_row_index_rejects_out_of_range(self):
        self._raises(lambda: self._e()[jt.array([9]), :].sync(), "2-D row index")

    def test_setitem_rejects_out_of_range(self):
        """The write side. This one was memory corruption, not a wrong read."""
        def write():
            y = self._x()
            y[jt.array([99])] = 1.0
            y.sync()
        self._raises(write, "setitem")

    def test_getitem_backward_rejects_out_of_range(self):
        """The gradient scatters through ``setitem``, so it needs its own case.

        A forward that raises and a backward that quietly writes out of bounds
        is the shape this file exists to prevent: the check would look present
        and the corruption would still happen, one pass later.
        """
        def backward():
            w = self._e()
            jt.grad(w[jt.array([9]), :].sum(), w).sync()
        self._raises(backward, "getitem backward")

    def test_legal_indices_are_unchanged(self):
        """The check must not cost correctness on the indices that are fine.

        Negative indices count from the end and have to keep doing so; that
        normalisation now happens inside the bounds check rather than after it,
        which is exactly the kind of move that silently drops a convention.
        """
        with jt.flag_scope(use_cuda=self.device_flag):
            got = self._x()[jt.array([-1, 0, 4, -5])].numpy()
            np.testing.assert_array_equal(got, [4.0, 0.0, 4.0, 0.0])

            rows = self._e()[jt.array([-1, 2]), :].numpy()
            np.testing.assert_array_equal(
                rows, [[16.0, 17.0, 18.0, 19.0], [8.0, 9.0, 10.0, 11.0]])

            y = self._x()
            y[jt.array([-1, 0])] = 7.0
            np.testing.assert_array_equal(y.numpy(), [7.0, 1.0, 2.0, 3.0, 7.0])

    def test_python_int_and_slice_bounds_are_still_what_they_were(self):
        """The two paths that were already right must not have moved.

        A Python ``int`` index raises, and a slice clamps the way NumPy does.
        Making the Var path strict is only correct if it did not also make
        ``x[2:99]`` an error.
        """
        with jt.flag_scope(use_cuda=self.device_flag):
            with self.assertRaises(RuntimeError):
                self._x()[99].sync()
            np.testing.assert_array_equal(self._x()[2:99].numpy(), [2.0, 3.0, 4.0])


#: Run in a subprocess: a device-side trap ends the CUDA context, so this
#: cannot share a process with anything that runs after it.
_CUDA_CHILD = textwrap.dedent("""
    import numpy as np, jittor as jt
    jt.flags.use_cuda = 1
    x = jt.array(np.arange(5, dtype="float32"))
    try:
        value = x[jt.array([99])].numpy()
    except Exception:
        raise SystemExit(0)          # refused, which is the point
    print("RETURNED", value)
    raise SystemExit(3)              # produced a number for an index that has none
""")


@unittest.skipIf(not _has_cuda(), "no CUDA device")
class TestVarIndexBoundsCuda(unittest.TestCase):

    def test_legal_indices_are_unchanged_on_device(self):
        with jt.flag_scope(use_cuda=1):
            x = jt.array(np.arange(5, dtype="float32"))
            np.testing.assert_array_equal(
                x[jt.array([-1, 0, 4, -5])].numpy(), [4.0, 0.0, 4.0, 0.0])

    def test_out_of_range_does_not_produce_a_value_on_device(self):
        env = dict(os.environ)
        env["JITTOR_TORCH_SHIM"] = "0"
        result = subprocess.run([sys.executable, "-c", _CUDA_CHILD],
                                capture_output=True, text=True, env=env,
                                timeout=2400)
        # Anything but a clean exit-0 means the child never reached the read or
        # never got a value back. Exit 3 is the one outcome this rejects: a
        # number returned for an index that does not exist.
        self.assertNotEqual(
            result.returncode, 3,
            "CUDA returned a value for an out-of-range index: %s"
            % result.stdout.strip())
        self.assertNotIn("RETURNED", result.stdout)


if __name__ == "__main__":
    unittest.main()
