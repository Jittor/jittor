# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The cuDNN convolution backward is defined once, in C++.

``nn/backends/cudnn.py`` used to wrap ``cudnn_conv`` and
``cudnn_conv_backward_x`` in ``jt.Function`` subclasses that supplied the
backward by hand, because autodiff through the raw op returned a wrongly
shaped gradient. The op was fixed (its ``grad`` reads the spatial sizes
through the layout strings instead of comparing against a layout name it
never receives), but the bypass stayed -- so the gradient of a cuDNN
convolution had two definitions, and the one that ran was the Python copy.
Two truths that drift is the failure mode this whole task family is about:
fixing the C++ side did not take effect.

These pin both halves: the values are checked against the CPU reindex path,
which is an independent implementation, and a rule check keeps a hand-written
backward from coming back into the Python layer.
"""
from pathlib import Path
import unittest

import numpy as np

import jittor as jt


CUDNN_BACKEND = (Path(__file__).resolve().parents[2]
                 / "python/jittor/nn/backends/cudnn.py")


def _reference_conv2d(x, w, stride, padding, dilation, groups):
    """Forward and both gradients from the CPU reindex path."""
    with jt.flag_scope(use_cuda=0):
        xv = jt.array(x)
        wv = jt.array(w)
        y = jt.nn.conv2d(xv, wv, None, stride, padding, dilation, groups)
        gx, gw = jt.grad(y.sum(), [xv, wv])
        return jt.fetch_sync([y, gx, gw])


def _reference_conv_transpose2d(x, w, stride, padding, output_padding,
                                dilation, groups):
    with jt.flag_scope(use_cuda=0):
        xv = jt.array(x)
        wv = jt.array(w)
        y = jt.nn.conv_transpose2d(xv, wv, None, stride, padding,
                                   output_padding, groups, dilation)
        gx, gw = jt.grad(y.sum(), [xv, wv])
        return jt.fetch_sync([y, gx, gw])


class TestCudnnConvBackwardHasOneDefinition(unittest.TestCase):
    """A rule, and it needs no GPU: the Python layer defines no backward."""

    def test_the_python_layer_does_not_redefine_the_backward(self):
        text = CUDNN_BACKEND.read_text(encoding="utf-8")
        offenders = [
            "%d: %s" % (n, line.strip())
            for n, line in enumerate(text.splitlines(), 1)
            if line.strip().startswith("def grad(")
            or "jt.Function" in line and not line.strip().startswith("#")
        ]
        self.assertEqual(
            offenders, [],
            "the cuDNN convolution backward belongs to the C++ op "
            "(CudnnConvOp::grad and friends); a jt.Function here is a second "
            "definition that silently wins:\n" + "\n".join(offenders))


@unittest.skipIf(not jt.has_cuda, "no CUDA")
class TestCudnnConvMatchesTheCpuReference(unittest.TestCase):
    """Values, against an independent implementation.

    The rule above only says the second definition is gone. It would pass
    just as well if removing it had broken the gradient, so the numbers have
    to be checked too -- and against the reindex path rather than against the
    removed Python backward, which would only prove the two agreed.
    """

    #: (in_ch, out_ch, groups, stride, padding, dilation)
    CASES = (
        (8, 6, 1, 1, 1, 1),
        (8, 6, 2, 1, 1, 1),     # grouped
        (8, 8, 8, 1, 1, 1),     # depthwise geometry
        (8, 6, 1, 2, 1, 1),     # strided
        (8, 6, 1, 1, 2, 2),     # dilated
        (8, 6, 1, 1, 0, 1),     # unpadded
    )

    def _check(self, in_ch, out_ch, groups, stride, padding, dilation):
        rng = np.random.RandomState(0)
        x = rng.randn(2, in_ch, 12, 13).astype("float32")
        w = rng.randn(out_ch, in_ch // groups, 3, 3).astype("float32")
        want = _reference_conv2d(x, w, stride, padding, dilation, groups)
        with jt.flag_scope(use_cuda=1):
            xv = jt.array(x)
            wv = jt.array(w)
            got = jt.nn._try_cudnn_conv2d(xv, wv, None, stride, padding,
                                          dilation, groups)
            self.assertIsNotNone(
                got, "the cuDNN path declined a plain float32 convolution")
            gx, gw = jt.grad(got.sum(), [xv, wv])
            got = jt.fetch_sync([got, gx, gw])
        for name, a, b in zip(("y", "dx", "dw"), got, want):
            scale = max(1.0, float(np.abs(b).max()))
            np.testing.assert_allclose(a / scale, b / scale,
                                       rtol=2e-5, atol=2e-5,
                                       err_msg="%s mismatch" % name)

    def test_forward_and_both_gradients_match(self):
        for case in self.CASES:
            with self.subTest(case=case):
                self._check(*case)

    def test_conv_transpose_forward_and_both_gradients_match(self):
        rng = np.random.RandomState(1)
        for groups, stride, padding, output_padding in (
                (1, 1, 0, 0), (1, 2, 1, 1), (2, 2, 1, 0)):
            with self.subTest(groups=groups, stride=stride):
                x = rng.randn(2, 6, 7, 8).astype("float32")
                w = rng.randn(6, 4, 3, 3).astype("float32")
                want = _reference_conv_transpose2d(
                    x, w, stride, padding, output_padding, 1, groups)
                with jt.flag_scope(use_cuda=1):
                    xv = jt.array(x)
                    wv = jt.array(w)
                    got = jt.nn._try_cudnn_conv_transpose2d(
                        xv, wv, None, stride, padding, output_padding,
                        1, groups)
                    self.assertIsNotNone(got)
                    gx, gw = jt.grad(got.sum(), [xv, wv])
                    got = jt.fetch_sync([got, gx, gw])
                for name, a, b in zip(("y", "dx", "dw"), got, want):
                    scale = max(1.0, float(np.abs(b).max()))
                    np.testing.assert_allclose(a / scale, b / scale,
                                               rtol=2e-5, atol=2e-5,
                                               err_msg="%s mismatch" % name)


if __name__ == "__main__":
    unittest.main()
