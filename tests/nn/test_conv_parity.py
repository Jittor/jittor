# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``nn.Conv2d`` and ``nn.functional.conv2d`` are one implementation.

They used to be two independent transcriptions of the same reindex, and they
had already drifted:

* the module set ``compile_options = {"G": G, "C": C}`` on *both* operands of
  the grouped path, the functional set ``{"G": G}`` on one of them -- so the
  same layer produced two different fused-op keys depending on how it was
  called;
* the module raised for a non-4-D input, a wrong channel count and a
  non-positive output size; the functional validated nothing and failed later
  with an empty ``AssertionError`` or "not enough values to unpack";
* the module took a CUDA depthwise kernel that the functional did not.

And ``Conv._conv_forward`` -- torch's documented hook, which mmdet's
``NormedConv2d`` calls -- went to the *functional*, so calling a layer and
calling its own ``_conv_forward`` did different things.

``execute`` now goes through ``_conv_forward`` into ``jt.nn.conv2d``.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt
from _helpers.torch_runtime import import_torch_modules, modules_available


_MISSING_REAL_TORCH = not modules_available("torch")
torch = None


def setUpModule():
    global torch
    if not _MISSING_REAL_TORCH:
        (torch,) = import_torch_modules("torch")


class _ConvParity:
    """Plain mixin; the CPU and CUDA classes below pick ``use_cuda``."""

    use_cuda = 0

    def setUp(self):
        self.rng = np.random.default_rng(20260903)

    def _layer_and_input(self, in_channels, out_channels, groups, **kwargs):
        with jt.flag_scope(use_cuda=self.use_cuda):
            layer = jt.nn.Conv2d(in_channels, out_channels, 3, groups=groups,
                                 **kwargs)
        x = self.rng.standard_normal((2, in_channels, 9, 10)).astype("float32")
        return layer, x

    def _assert_module_matches_functional(self, layer, x):
        with jt.flag_scope(use_cuda=self.use_cuda):
            seed = jt.array(self.rng.standard_normal(
                tuple(layer(jt.array(x)).shape)).astype("float32"))

            xa = jt.array(x)
            ya = layer(xa)
            ga = jt.grad((ya * seed).sum(), [xa, layer.weight])

            xb = jt.array(x)
            yb = jt.nn.conv2d(xb, layer.weight, layer.bias, layer.stride,
                              layer.padding, layer.dilation, layer.groups)
            gb = jt.grad((yb * seed).sum(), [xb, layer.weight])

            xc = jt.array(x)
            yc = layer._conv_forward(xc, layer.weight, layer.bias)

            values = [v.numpy() for v in
                      (ya, yb, yc, ga[0], ga[1], gb[0], gb[1])]
        ya, yb, yc, gxa, gwa, gxb, gwb = values
        # Same graph -> bit-equal forward. A second transcription with
        # different compile options would not be, even where a tolerance would
        # let it through.
        np.testing.assert_array_equal(ya, yb)
        np.testing.assert_array_equal(ya, yc)
        if self.use_cuda:
            # CUDA gradients are not bit-reproducible call to call: cuDNN
            # picks its backward algorithm at runtime and the depthwise
            # backward accumulates with atomicAdd. Repeating the *identical*
            # call already differs in the last bit
            # (TestCudaGradientReproducibility below pins that), so
            # bit-equality here would be flaky regardless of whether the two
            # spellings share an implementation.
            np.testing.assert_allclose(gxa, gxb, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(gwa, gwb, rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_array_equal(gxa, gxb)
            np.testing.assert_array_equal(gwa, gwb)

    def test_plain_convolution(self):
        layer, x = self._layer_and_input(4, 6, 1, padding=1)
        self._assert_module_matches_functional(layer, x)

    def test_grouped_convolution(self):
        """The path whose compile options differed between the two copies."""
        layer, x = self._layer_and_input(6, 9, 3, padding=1, stride=2)
        self._assert_module_matches_functional(layer, x)

    def test_depthwise_convolution(self):
        layer, x = self._layer_and_input(4, 4, 4, padding=1)
        self._assert_module_matches_functional(layer, x)

    def test_dilated_and_strided(self):
        layer, x = self._layer_and_input(3, 5, 1, padding=2, stride=2,
                                         dilation=2, bias=False)
        self._assert_module_matches_functional(layer, x)

    def test_functional_validates_what_the_module_used_to_validate_alone(self):
        weight = jt.array(self.rng.standard_normal((6, 4, 3, 3)).astype("float32"))
        with jt.flag_scope(use_cuda=self.use_cuda):
            with self.assertRaises(ValueError) as ctx:
                jt.nn.conv2d(jt.array(self.rng.standard_normal(
                    (4, 9, 10)).astype("float32")), weight)
            self.assertIn("4-D input", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx:
                jt.nn.conv2d(jt.array(self.rng.standard_normal(
                    (2, 5, 9, 10)).astype("float32")), weight)
            self.assertIn("4 channels", str(ctx.exception))

            with self.assertRaises(ValueError) as ctx:
                jt.nn.conv2d(jt.array(self.rng.standard_normal(
                    (2, 4, 2, 2)).astype("float32")), weight)
            self.assertIn("non-positive", str(ctx.exception))

    def test_module_raises_the_same_way_as_the_functional(self):
        layer, _ = self._layer_and_input(4, 6, 1)
        with jt.flag_scope(use_cuda=self.use_cuda):
            for bad, needle in (
                ((4, 9, 10), "4-D input"),
                ((2, 5, 9, 10), "4 channels"),
                ((2, 4, 2, 2), "non-positive"),
            ):
                with self.subTest(shape=bad):
                    with self.assertRaises(ValueError) as ctx:
                        layer(jt.array(
                            self.rng.standard_normal(bad).astype("float32")))
                    self.assertIn(needle, str(ctx.exception))


class TestConvParityCPU(_ConvParity, unittest.TestCase):
    use_cuda = 0

    def test_list_spatial_arguments_are_normalized(self):
        """Torch model configs may spell 2-D convolution sizes as lists."""
        with jt.flag_scope(use_cuda=0):
            layer = jt.nn.Conv2d(
                3, 4, kernel_size=[3, 3], stride=[1, 1],
                padding=[1, 1], dilation=[1, 1])
            output = layer(jt.ones((1, 3, 5, 5)))
        self.assertEqual(tuple(output.shape), (1, 4, 5, 5))

    def test_conv1d_torch_optional_arguments_are_accepted(self):
        with jt.flag_scope(use_cuda=0):
            layer = jt.nn.Conv1d(3, 4, [3], padding=1, padding_mode='zeros')
            output = layer(jt.ones((1, 3, 7)))
        self.assertEqual(tuple(output.shape), (1, 4, 7))

    @unittest.skipIf(_MISSING_REAL_TORCH, "an independent PyTorch oracle is not installed")
    def test_nonzero_padding_modes_match_torch(self):
        self._assert_nonzero_padding_modes_match_torch(use_cuda=0)

    def _assert_nonzero_padding_modes_match_torch(self, use_cuda):
        rng = np.random.default_rng(20260913)
        cases = (
            ("conv1d", (2, 3, 9), (4, 3, 3), 2),
            ("conv2d", (2, 3, 7, 8), (4, 3, 3, 3), (1, 2)),
        )
        torch_device = "cuda" if use_cuda else "cpu"
        for kind, input_shape, weight_shape, padding in cases:
            for mode in ("reflect", "replicate", "circular"):
                with self.subTest(kind=kind, mode=mode, device=torch_device):
                    x = rng.standard_normal(input_shape).astype("float32")
                    weight = rng.standard_normal(weight_shape).astype("float32")
                    bias = rng.standard_normal((weight_shape[0],)).astype("float32")
                    if kind == "conv1d":
                        jt_layer = jt.nn.Conv1d(
                            3, 4, 3, padding=padding, padding_mode=mode)
                        torch_layer = torch.nn.Conv1d(
                            3, 4, 3, padding=padding, padding_mode=mode)
                    else:
                        jt_layer = jt.nn.Conv2d(
                            3, 4, 3, padding=padding, padding_mode=mode)
                        torch_layer = torch.nn.Conv2d(
                            3, 4, 3, padding=padding, padding_mode=mode)
                    with jt.flag_scope(use_cuda=use_cuda):
                        jt_layer.weight.assign(weight)
                        jt_layer.bias.assign(bias)
                        actual = jt_layer(jt.array(x)).numpy()
                    with torch.no_grad():
                        torch_layer.weight.copy_(torch.from_numpy(weight))
                        torch_layer.bias.copy_(torch.from_numpy(bias))
                        torch_layer.to(torch_device)
                        expected = torch_layer(
                            torch.from_numpy(x).to(torch_device)).cpu().numpy()
                    np.testing.assert_allclose(
                        actual, expected, rtol=1e-5, atol=1e-5)


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "no CUDA")
class TestConvParityCUDA(_ConvParity, unittest.TestCase):
    use_cuda = 1

    @unittest.skipIf(_MISSING_REAL_TORCH, "an independent PyTorch oracle is not installed")
    def test_nonzero_padding_modes_match_torch(self):
        TestConvParityCPU._assert_nonzero_padding_modes_match_torch(
            self, use_cuda=1)


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "no CUDA")
class TestCudaGradientReproducibility(unittest.TestCase):
    """Why the parity test above compares CUDA gradients with a tolerance.

    The depthwise CUDA backward accumulates the weight gradient with atomicAdd,
    so repeating the *identical* call does not give bit-identical gradients.
    ``nn.Conv2d`` has always taken that kernel; what changed is that
    ``nn.functional.conv2d`` takes it too, which is the point of the merge --
    one implementation means one set of properties, including this one. torch
    classifies its own depthwise conv backward as non-deterministic in exactly
    the same way.

    The plain and grouped paths do reproduce bit for bit, so this is specific
    and measured rather than assumed. If the depthwise kernel is ever made
    deterministic, this test fails and the parity assertions above should
    tighten back to ``assert_array_equal``.
    """

    def _repeat_grads(self, in_ch, out_ch, groups, **kwargs):
        rng = np.random.default_rng(11)
        x = rng.standard_normal((2, in_ch, 9, 10)).astype("float32")
        w = rng.standard_normal((out_ch, in_ch // groups, 3, 3)).astype("float32")
        runs = []
        with jt.flag_scope(use_cuda=1):
            seed = None
            for _ in range(5):
                xv, wv = jt.array(x), jt.array(w)
                y = jt.nn.conv2d(xv, wv, None, 1, 1, 1, groups, **kwargs)
                if seed is None:
                    seed = jt.array(
                        rng.standard_normal(tuple(y.shape)).astype("float32"))
                runs.append([g.numpy()
                             for g in jt.grad((y * seed).sum(), [xv, wv])])
        return runs

    @staticmethod
    def _bitwise_stable(runs):
        return all(np.array_equal(first, other)
                   for run in runs[1:]
                   for first, other in zip(runs[0], run))

    def test_depthwise_backward_is_not_bit_reproducible(self):
        runs = self._repeat_grads(4, 4, 4)
        self.assertFalse(
            self._bitwise_stable(runs),
            "the depthwise CUDA backward became bit-reproducible; tighten "
            "_assert_module_matches_functional back to assert_array_equal")
        for run in runs[1:]:
            for first, other in zip(runs[0], run):
                np.testing.assert_allclose(first, other, rtol=1e-5, atol=1e-5)

    def test_the_generic_paths_are_numerically_reproducible(self):
        """Generic CUDA backward stays within the normal floating-point tolerance.

        cuDNN may select a different accumulation order between launches even
        for the same generic convolution.  The public contract is numerical
        parity, not bitwise identity across independent CUDA launches.
        """
        for in_ch, out_ch, groups in ((4, 6, 1), (6, 9, 3)):
            with self.subTest(groups=groups):
                runs = self._repeat_grads(in_ch, out_ch, groups)
                for run in runs[1:]:
                    for first, other in zip(runs[0], run):
                        np.testing.assert_allclose(first, other,
                                                    rtol=1e-5, atol=1e-5)
        # the same depthwise geometry, with the fast path off
        runs = self._repeat_grads(4, 4, 4, _depthwise_fast_path=False)
        for run in runs[1:]:
            for first, other in zip(runs[0], run):
                np.testing.assert_allclose(first, other, rtol=1e-5, atol=1e-5)

    def test_depthwise_kernel_agrees_with_the_generic_path(self):
        fast = self._repeat_grads(4, 4, 4)[0]
        generic = self._repeat_grads(4, 4, 4, _depthwise_fast_path=False)[0]
        for a, b in zip(fast, generic):
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "no CUDA")
class TestDepthwiseSelectionIsPerCall(unittest.TestCase):
    """The fast path is chosen when the layer runs, not when it is built.

    ``Conv.__init__`` used to build a ``DepthwiseConv`` only if
    ``jt.flags.use_cuda`` was already on, so a model constructed before the
    flag was set silently never took it -- and the two orders gave different
    (both correct, differently rounded) numbers.
    """

    def test_layer_built_on_cpu_still_takes_the_cuda_kernel(self):
        rng = np.random.default_rng(7)
        with jt.flag_scope(use_cuda=0):
            layer = jt.nn.Conv2d(4, 4, 3, groups=4, padding=1)
        x = rng.standard_normal((2, 4, 9, 10)).astype("float32")
        with jt.flag_scope(use_cuda=1):
            fast = layer(jt.array(x)).numpy()
            generic = jt.nn.conv2d(
                jt.array(x), layer.weight, layer.bias, layer.stride,
                layer.padding, layer.dilation, layer.groups,
                _depthwise_fast_path=False).numpy()
        np.testing.assert_allclose(fast, generic, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
