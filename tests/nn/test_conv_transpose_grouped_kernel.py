# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""A grouped ``conv_transpose`` asks for an accelerated kernel too.

``conv_transpose``'s ``groups == 1`` branch has consulted ``select_kernel``
since it was written; the grouped branch never did, so a depthwise transposed
convolution always took the eight-dimensional
``reindex * reindex -> reindex_reduce`` lowering -- even though cuDNN's adapter
takes a ``groups`` argument and passes it straight to
``cudnn_conv_backward_x``. The capability was there and nothing asked for it.

What that cost, measured on the MiniMax-H3 audio VAE, whose alias-free
resampler upsamples with ``conv_transpose1d(..., groups=C)``: one fused op was
79.8% of the whole decode, moving 344 MB/s on a card that does about 3 TB/s.
Routing the grouped call through the same lookup took that decode from
2.962 s to 1.404 s.

The speed is not the contract, so it is not what these assert. A faster path
that answers differently is not a speedup: what is asserted is that both paths
return the same numbers, which they do bit for bit.
"""
import unittest

import numpy as np

import jittor as jt
import jittor.nn.functional.convolution_transpose as convolution_transpose


#: (channels, length, kernel, stride, groups, padding, output_padding, dilation)
_CASES = (
    (8, 4096, 12, 2, 8, 0, 0, 1),      # depthwise, the audio resampler's shape
    (16, 2048, 12, 2, 16, 2, 0, 1),    # padded
    (32, 1024, 8, 4, 32, 1, 1, 1),     # output_padding
    (12, 777, 5, 3, 4, 1, 0, 1),       # groups < channels
    (8, 333, 7, 2, 8, 3, 1, 2),        # dilated
)


class TestGroupedConvTransposeKernel(unittest.TestCase):
    def _both_paths(self, channels, length, kernel, stride, groups,
                    padding, output_padding, dilation):
        rs = np.random.RandomState(0)
        x = jt.array(rs.randn(1, channels, length).astype("float32"))
        w = jt.array(rs.randn(channels, channels // groups, kernel).astype("float32"))

        def run():
            return jt.nn.conv_transpose1d(
                x, w, stride=stride, padding=padding,
                output_padding=output_padding, groups=groups,
                dilation=dilation).numpy()

        real = convolution_transpose.select_kernel
        try:
            convolution_transpose.select_kernel = lambda *a, **k: None
            lowered = run()
        finally:
            convolution_transpose.select_kernel = real
        return lowered, run()

    def test_the_accelerated_path_returns_what_the_lowering_returns(self):
        for case in _CASES:
            with self.subTest(case=case):
                lowered, accelerated = self._both_paths(*case)
                self.assertEqual(lowered.shape, accelerated.shape)
                np.testing.assert_array_equal(accelerated, lowered)

    def test_the_grouped_branch_consults_the_kernel_lookup(self):
        # The regression this guards is not a wrong number, it is a question
        # never asked -- so ask whether it was asked.
        asked = []
        real = convolution_transpose.select_kernel

        def watching(op, *args, **kwargs):
            asked.append(op)
            return real(op, *args, **kwargs)

        try:
            convolution_transpose.select_kernel = watching
            x = jt.random((1, 8, 64))
            w = jt.random((8, 1, 4))
            jt.nn.conv_transpose1d(x, w, stride=2, groups=8).sync()
        finally:
            convolution_transpose.select_kernel = real
        self.assertIn("conv_transpose2d", asked)


if __name__ == "__main__":
    unittest.main()
