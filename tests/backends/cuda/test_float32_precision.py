# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""One precision policy for float32 matmul and convolution.

Four knobs used to answer "how is this product accumulated", in four
encodings, and they disagreed with each other:

* ``cublas_matmul`` and ``cublas_batched_matmul`` accumulated float16 in
  float32 by default and in float16 once tensor cores were switched on;
  ``cublas_acc_matmul`` accumulated float16 in float16 always. Which of the
  three ran was a property of how the graph was built, not of any setting, so
  the accumulate precision of one float16 product was not something the caller
  could read off anywhere.
* ``cudnn_conv`` and the three ``cudnn_conv3d`` ops accumulated float16 in
  float32 and asked for tensor-op math; ``cudnn_conv_backward_x`` and
  ``cudnn_conv_backward_w`` accumulated float16 *in float16* and left the math
  type at ``CUDNN_DEFAULT_MATH``. One float16 convolution therefore had one
  accumulate precision going forward and a different one coming back -- and
  its own backend-API fast path, which always accumulates in float32,
  disagreed with its legacy fallback.

Now ``jt.flags.float32_matmul_precision`` names the policy on torch's scale
and both backends read it. ``use_tensorcore``, ``cuda_allow_tf32`` and
``cuda_allow_cudnn_tf32`` remain as deprecated overrides that can only raise
the tier for the domain they name.

The selection is read back from the log line each op emits just before it
calls the library, not inferred from numerics: on CUDA >= 11 the algorithm
hint is advisory and several of these choices are invisible in the output
values.
"""
import unittest

import numpy as np

import jittor as jt

from _helpers.logs import find_log_with_re


#: ``<op> algo select: precision=<tier> computeType=<...> algo=<...>``
_GEMM_RE = r"algo select: precision=(\S+) computeType=(\S+) algo=(\S+)"
#: ``<op> precision select: precision=<tier> computeType=<...> mathType=<...>``
_CONV_RE = r"precision select: precision=(\S+) computeType=(\S+) mathType=(\S+)"

DEFAULT = "CUBLAS_GEMM_DEFAULT"
TENSOR_OP = "CUBLAS_GEMM_DEFAULT_TENSOR_OP"
FMA = "CUDNN_FMA_MATH"
ALLOW_CONVERSION = "CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION"

TIERS = ("highest", "high", "medium")


def _capture(prefix, pattern, build):
    """Run ``build`` and return the selection its log line reports.

    ``build`` has to hand back the vars it wants executed and they have to be
    synced here: a var nobody holds is never computed, so dropping the result
    on the floor and calling ``jt.sync_all()`` captures an empty log and looks
    exactly like an op that stopped emitting the line.
    """
    with jt.log_capture_scope(log_silent=1, log_v=0,
                              log_vprefix="%s=100" % prefix) as raw_log:
        out = build()
        if not isinstance(out, (list, tuple)):
            out = [out]
        jt.fetch_sync(list(out))
    found = find_log_with_re(raw_log, pattern)
    assert found, "no %s selection log captured" % prefix
    return found[-1]


class TestFloat32PrecisionFlag(unittest.TestCase):
    """The flag itself. Core, so this half needs no GPU."""

    def setUp(self):
        self._saved = jt.flags.float32_matmul_precision

    def tearDown(self):
        jt.flags.float32_matmul_precision = self._saved

    def test_default_is_highest(self):
        """The default has to stay the exact numerics 1.3.x shipped.

        `highest` is CUBLAS_COMPUTE_32F and CUDNN_FMA_MATH, which is what
        ``use_tensorcore=0, cuda_allow_tf32=0, cuda_allow_cudnn_tf32=0``
        selected before this flag existed.
        """
        self.assertEqual(jt.flags.float32_matmul_precision, "highest")

    def test_each_tier_round_trips(self):
        for tier in TIERS:
            jt.flags.float32_matmul_precision = tier
            self.assertEqual(jt.flags.float32_matmul_precision, tier)

    def test_a_bad_tier_is_rejected_and_changes_nothing(self):
        """A rejected setter must not leave a half-applied policy behind.

        The flag is assigned before its setter runs, so a setter that throws
        without the rollback would leave ``float32_matmul_precision`` reading
        "fastest" while the ops still ran at whatever tier was parsed last.
        """
        jt.flags.float32_matmul_precision = "high"
        with self.assertRaises(Exception):
            jt.flags.float32_matmul_precision = "fastest"
        self.assertEqual(jt.flags.float32_matmul_precision, "high")


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCublasPrecision(unittest.TestCase):
    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.flags.use_tensorcore,
                       jt.flags.cuda_allow_tf32,
                       jt.flags.float32_matmul_precision)
        jt.flags.use_cuda = 1
        jt.flags.use_tensorcore = 0
        jt.flags.cuda_allow_tf32 = 0
        jt.flags.float32_matmul_precision = "highest"

    def tearDown(self):
        jt.sync_all()
        (jt.flags.use_cuda, jt.flags.use_tensorcore,
         jt.flags.cuda_allow_tf32,
         jt.flags.float32_matmul_precision) = self._saved

    # ---- the three ops, built so each one is the one that runs ----------
    def _matmul(self, dtype):
        a = jt.random((32, 48)).cast(dtype)
        b = jt.random((48, 64)).cast(dtype)
        return lambda: jt.matmul(a, b)

    def _batched(self, dtype):
        a = jt.random((3, 32, 48)).cast(dtype)
        b = jt.random((3, 48, 64)).cast(dtype)
        return lambda: jt.matmul(a, b)

    def _acc(self, dtype):
        a = jt.random((32, 48)).cast(dtype)
        b = jt.random((48, 64)).cast(dtype)
        return lambda: jt.compile_extern.cublas_ops.cublas_acc_matmul(
            a, b, 0, 0, -1, -1, 0, 0)

    OPS = (("cublas_matmul", "_matmul"),
           ("cublas_batched_matmul", "_batched"),
           ("cublas_acc_matmul", "_acc"))

    #: (dtype, tier) -> (computeType, algo), one table for all three ops.
    #
    #  float16 and bfloat16 do not appear per tier because they do not depend
    #  on it: reduced-precision operands always accumulate in float32, which
    #  is torch's rule and was already the default for two of the three ops.
    TABLE = {
        ("float32", "highest"): ("CUBLAS_COMPUTE_32F", DEFAULT),
        ("float32", "high"): ("CUBLAS_COMPUTE_32F_FAST_TF32", TENSOR_OP),
        ("float32", "medium"): ("CUBLAS_COMPUTE_32F_FAST_16BF", TENSOR_OP),
        ("float16", "highest"): ("CUBLAS_COMPUTE_32F", DEFAULT),
        ("float16", "high"): ("CUBLAS_COMPUTE_32F", DEFAULT),
        ("float16", "medium"): ("CUBLAS_COMPUTE_32F", DEFAULT),
        ("bfloat16", "highest"): ("CUBLAS_COMPUTE_32F", DEFAULT),
        ("bfloat16", "high"): ("CUBLAS_COMPUTE_32F", DEFAULT),
        ("bfloat16", "medium"): ("CUBLAS_COMPUTE_32F", DEFAULT),
        ("float64", "highest"): ("CUBLAS_COMPUTE_64F", DEFAULT),
        ("float64", "high"): ("CUBLAS_COMPUTE_64F", DEFAULT),
        ("float64", "medium"): ("CUBLAS_COMPUTE_64F", DEFAULT),
    }

    def test_every_op_follows_the_same_table(self):
        for dtype in ("float32", "float16", "bfloat16", "float64"):
            for tier in TIERS:
                jt.flags.float32_matmul_precision = tier
                want_compute, want_algo = self.TABLE[(dtype, tier)]
                for prefix, maker in self.OPS:
                    got_tier, got_compute, got_algo = _capture(
                        prefix, _GEMM_RE, getattr(self, maker)(dtype))
                    where = "%s %s %s" % (prefix, dtype, tier)
                    self.assertEqual(got_tier, tier, where)
                    self.assertEqual(got_compute, want_compute, where)
                    self.assertEqual(got_algo, want_algo, where)

    def test_float16_accumulates_in_float32_on_every_path(self):
        """The defect this table exists to pin down.

        ``cublas_acc_matmul`` used to pass CUBLAS_COMPUTE_16F unconditionally
        while the other two passed CUBLAS_COMPUTE_32F, so the accumulate
        precision of one float16 product depended on which op the graph
        picked. Stated on its own because the table above would still pass if
        all three agreed on the *wrong* value.
        """
        for tier in TIERS:
            jt.flags.float32_matmul_precision = tier
            for prefix, maker in self.OPS:
                _, compute, _ = _capture(prefix, _GEMM_RE,
                                         getattr(self, maker)("float16"))
                self.assertEqual(compute, "CUBLAS_COMPUTE_32F",
                                 "%s at %s" % (prefix, tier))

    # ---- the deprecated knobs still mean what they meant -----------------
    def test_use_tensorcore_maps_onto_the_tiers(self):
        for value, tier in ((0, "highest"), (1, "high"), (2, "medium"),
                            (3, "medium")):
            jt.flags.use_tensorcore = value
            got_tier, got_compute, _ = _capture(
                "cublas_matmul", _GEMM_RE, self._matmul("float32"))
            self.assertEqual(got_tier, tier, "use_tensorcore=%d" % value)
            self.assertEqual(got_compute, self.TABLE[("float32", tier)][0])
        jt.flags.use_tensorcore = 0

    def test_cuda_allow_tf32_raises_the_tier_but_cannot_lower_it(self):
        """An override raises; it never overrules a stricter policy downwards.

        ``cuda_allow_tf32=1`` on top of ``medium`` must not drop the matmul
        back to tf32 -- the flag says "tf32 is acceptable", not "use tf32".
        """
        jt.flags.cuda_allow_tf32 = 1
        got_tier, got_compute, _ = _capture(
            "cublas_matmul", _GEMM_RE, self._matmul("float32"))
        self.assertEqual(got_tier, "high")
        self.assertEqual(got_compute, "CUBLAS_COMPUTE_32F_FAST_TF32")

        jt.flags.float32_matmul_precision = "medium"
        got_tier, got_compute, _ = _capture(
            "cublas_matmul", _GEMM_RE, self._matmul("float32"))
        self.assertEqual(got_tier, "medium")
        self.assertEqual(got_compute, "CUBLAS_COMPUTE_32F_FAST_16BF")

    def test_float16_accumulate_is_visible_in_the_result(self):
        """The same defect, read off the numbers instead of the log.

        Reduce long enough and float16 accumulation shows: against a float64
        reference over k=8192, ``cublas_acc_matmul`` was off by 0.63 where
        ``cublas_matmul`` on the identical inputs was off by 0.14. Both now
        report 0.14.

        The inputs have to be irregular for this to bite. All-ones is exactly
        representable at every partial sum, so a float16 accumulator returns
        the exact answer and the defect hides completely -- which is how it
        survived this long.
        """
        rng = np.random.RandomState(0)
        k = 8192
        a_np = rng.randn(64, k).astype("float16")
        b_np = rng.randn(k, 64).astype("float16")
        want = a_np.astype("float64") @ b_np.astype("float64")
        a = jt.array(a_np)
        b = jt.array(b_np)
        acc = jt.compile_extern.cublas_ops.cublas_acc_matmul(
            a, b, 0, 0, -1, -1, 0, 0)
        plain = jt.matmul(a, b)
        got_acc, got_plain = jt.fetch_sync([acc, plain])
        err_acc = float(np.abs(got_acc.astype("float64") - want).max())
        err_plain = float(np.abs(got_plain.astype("float64") - want).max())
        # Halfway between the two measurements above, so the assertion fails
        # on float16 accumulation and passes on float32 without pinning an
        # exact value that a cuBLAS update could move.
        self.assertLess(err_acc, 0.3, "acc_matmul error %.4f" % err_acc)
        self.assertLess(err_plain, 0.3, "matmul error %.4f" % err_plain)

    def test_values_stay_correct_at_every_tier(self):
        a = np.random.RandomState(0).randn(32, 48).astype("float32")
        b = np.random.RandomState(1).randn(48, 64).astype("float32")
        want = a @ b
        for tier in TIERS:
            jt.flags.float32_matmul_precision = tier
            got = jt.matmul(jt.array(a), jt.array(b)).numpy()
            # high and medium deliberately spend accuracy, so the tolerance
            # has to admit tf32 and bfloat16 compute.
            tol = 1e-4 if tier == "highest" else 3e-1
            np.testing.assert_allclose(got, want, atol=tol, rtol=tol)


@unittest.skipIf(not jt.has_cuda, "No CUDA found")
class TestCudnnConvPrecision(unittest.TestCase):
    def setUp(self):
        self._saved = (jt.flags.use_cuda, jt.flags.use_tensorcore,
                       jt.flags.cuda_allow_tf32,
                       jt.flags.cuda_allow_cudnn_tf32,
                       jt.flags.float32_matmul_precision)
        jt.flags.use_cuda = 1
        jt.flags.use_tensorcore = 0
        jt.flags.cuda_allow_tf32 = 0
        jt.flags.cuda_allow_cudnn_tf32 = 0
        jt.flags.float32_matmul_precision = "highest"

    def tearDown(self):
        jt.sync_all()
        (jt.flags.use_cuda, jt.flags.use_tensorcore,
         jt.flags.cuda_allow_tf32, jt.flags.cuda_allow_cudnn_tf32,
         jt.flags.float32_matmul_precision) = self._saved

    def _conv(self, dtype, backward=False):
        rng = np.random.RandomState(0)
        x = jt.array(rng.randn(2, 8, 16, 16).astype("float32")).cast(dtype)
        w = jt.array(rng.randn(8, 8, 3, 3).astype("float32")).cast(dtype)

        def build():
            y = jt.nn.conv2d(x, w, None, 1, 1)
            if not backward:
                return y
            gx, gw = jt.grad(y.float32().sum(), [x, w])
            return [y, gx, gw]
        return build

    def _selection(self, op, dtype, backward=False):
        return _capture(op, _CONV_RE, self._conv(dtype, backward))

    def test_float32_conv_follows_the_policy(self):
        for tier, math in (("highest", FMA),
                           ("high", ALLOW_CONVERSION),
                           ("medium", ALLOW_CONVERSION)):
            jt.flags.float32_matmul_precision = tier
            got_tier, compute, got_math = self._selection("cudnn_conv", "float32")
            self.assertEqual(got_tier, tier)
            self.assertEqual(compute, "CUDNN_DATA_FLOAT")
            self.assertEqual(got_math, math, "float32 conv at %s" % tier)

    def test_float16_conv_accumulates_in_float32_forward_and_backward(self):
        """The convolution half of the same defect.

        ``cudnn_conv_backward_x``/``_w`` passed ``getDataType<Ty>()`` as the
        convolution's compute type, so a float16 convolution accumulated in
        float16 on the way back and in float32 on the way out -- and in
        float32 again if the backend-API fast path took the call, since that
        path hard-codes CUDNN_DATA_FLOAT.
        """
        for op in ("cudnn_conv", "cudnn_conv_backward_x",
                   "cudnn_conv_backward_w"):
            _, compute, math = self._selection(op, "float16", backward=True)
            self.assertEqual(compute, "CUDNN_DATA_FLOAT", op)
            self.assertEqual(math, ALLOW_CONVERSION, op)

    def test_cudnn_tf32_override_does_not_move_the_matmul(self):
        """torch's two allow_tf32 knobs stay independent.

        ``torch.backends.cudnn.allow_tf32`` defaults to True in torch while
        ``torch.backends.cuda.matmul.allow_tf32`` defaults to False, so the
        shim needs a convolution-only override; merging the two would silently
        drop every downstream matmul to tf32 the moment a framework touched
        the cuDNN switch.
        """
        jt.flags.cuda_allow_cudnn_tf32 = 1
        conv_tier, _, conv_math = self._selection("cudnn_conv", "float32")
        self.assertEqual(conv_tier, "high")
        self.assertEqual(conv_math, ALLOW_CONVERSION)

        a = jt.random((32, 48))
        b = jt.random((48, 64))
        gemm_tier, gemm_compute, _ = _capture(
            "cublas_matmul", _GEMM_RE, lambda: jt.matmul(a, b))
        self.assertEqual(gemm_tier, "highest")
        self.assertEqual(gemm_compute, "CUBLAS_COMPUTE_32F")


if __name__ == "__main__":
    unittest.main()
