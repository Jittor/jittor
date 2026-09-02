# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The model zoo's shared building blocks come from one place.

``ConvNormActivation``, ``SqueezeExcitation``, ``StochasticDepth`` and
``make_divisible`` used to be copied into each model file. The copies drifted --
one of them replaced ``activation_layer=None`` with a default before testing it,
which is what gave every EfficientNet projection an extra SiLU. These tests pin
both the single source and the semantics the copies disagreed about.
"""
import unittest

import numpy as np

import jittor as jt
from jittor import nn
import jittor.models as jtmodels
from jittor.models import _utils
from jittor.models import convnext, efficientnet, maxvit, mobilenet, mobilenet_v3, regnet


class TestSharedBlocksAreShared(unittest.TestCase):

    def test_models_use_the_shared_classes(self):
        self.assertIs(efficientnet.SqueezeExcitation, _utils.SqueezeExcitation)
        self.assertIs(regnet.SqueezeExcitation, _utils.SqueezeExcitation)
        self.assertIs(maxvit.SqueezeExcitation, _utils.SqueezeExcitation)
        self.assertIs(efficientnet.StochasticDepth, _utils.StochasticDepth)
        self.assertIs(convnext.StochasticDepth, _utils.StochasticDepth)
        self.assertIs(maxvit.StochasticDepth, _utils.StochasticDepth)
        self.assertIs(regnet.ConvNormActivation, _utils.ConvNormActivation)
        self.assertIs(maxvit.ConvNormActivation, _utils.ConvNormActivation)
        self.assertTrue(issubclass(efficientnet.ConvBNActivation,
                                   _utils.ConvNormActivation))
        # MobileNetV3's SE keeps its own signature but not its own body
        self.assertTrue(issubclass(mobilenet_v3.SqueezeExcitation,
                                   _utils.SqueezeExcitation))
        for module in (efficientnet, mobilenet_v3, regnet):
            self.assertIs(module.make_divisible, _utils.make_divisible)
        self.assertIs(mobilenet._make_divisible, _utils.make_divisible)


class TestConvNormActivationSemantics(unittest.TestCase):
    """``None`` means "skip", never "use the default"."""

    def test_none_activation_builds_no_activation(self):
        block = _utils.ConvNormActivation(3, 4, activation_layer=None)
        self.assertEqual([type(m).__name__ for m in block],
                         ["Conv", "BatchNorm"])

    def test_none_norm_builds_no_norm_and_enables_bias(self):
        block = _utils.ConvNormActivation(3, 4, norm_layer=None,
                                          activation_layer=None)
        self.assertEqual([type(m).__name__ for m in block], ["Conv"])
        self.assertIsNotNone(block[0].bias)

    def test_defaults_are_conv_norm_relu_without_bias(self):
        block = _utils.ConvNormActivation(3, 4)
        self.assertEqual([type(m).__name__ for m in block],
                         ["Conv", "BatchNorm", "ReLU"])
        self.assertIsNone(block[0].bias)

    def test_efficientnet_alias_still_defaults_to_silu(self):
        block = efficientnet.ConvBNActivation(3, 4)
        self.assertEqual([type(m).__name__ for m in block],
                         ["Conv", "BatchNorm", "SiLU"])
        block = efficientnet.ConvBNActivation(3, 4, activation_layer=None)
        self.assertEqual([type(m).__name__ for m in block],
                         ["Conv", "BatchNorm"])

    def test_padding_defaults_to_same(self):
        block = _utils.ConvNormActivation(3, 4, kernel_size=5, dilation=2)
        self.assertEqual(block[0].padding, (4, 4))
        block = _utils.ConvNormActivation(3, 4, kernel_size=5, padding=0)
        self.assertEqual(block[0].padding, (0, 0))

    def test_out_channels_is_exposed(self):
        self.assertEqual(_utils.ConvNormActivation(3, 7).out_channels, 7)


class TestMakeDivisible(unittest.TestCase):

    def test_matches_the_torchvision_rule(self):
        # (value, divisor) -> expected, from torchvision's _make_divisible
        cases = {
            (32, 8): 32, (33, 8): 32, (36, 8): 40, (4, 8): 8,
            (1, 8): 8, (17, 8): 16, (20, 8): 24, (144, 8): 144,
            (10, 4): 12, (7, 4): 8,
        }
        for (value, divisor), expected in cases.items():
            self.assertEqual(_utils.make_divisible(value, divisor), expected,
                             (value, divisor))

    def test_never_drops_more_than_ten_percent(self):
        for value in range(1, 400):
            out = _utils.make_divisible(value, 8)
            assert out >= 0.9 * value, (value, out)


class TestStochasticDepth(unittest.TestCase):

    def test_identity_in_eval(self):
        layer = _utils.StochasticDepth(0.5, "row")
        layer.eval()
        x = jt.random((4, 3))
        np.testing.assert_array_equal(layer(x).numpy(), x.numpy())

    def test_zero_probability_is_identity_in_training(self):
        layer = _utils.StochasticDepth(0.0, "row")
        layer.train()
        x = jt.random((4, 3))
        np.testing.assert_array_equal(layer(x).numpy(), x.numpy())

    def test_row_mode_drops_whole_samples(self):
        layer = _utils.StochasticDepth(0.5, "row")
        layer.train()
        out = layer(jt.ones((64, 3, 2))).numpy()
        # every kept sample is scaled by 1/(1-p) == 2, dropped ones are 0
        for row in out:
            assert np.allclose(row, 0.0) or np.allclose(row, 2.0), row

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            _utils.StochasticDepth(1.5)
        with self.assertRaises(ValueError):
            _utils.StochasticDepth(0.5, "column")


class TestSqueezeExcitation(unittest.TestCase):

    def test_gate_shape_and_range(self):
        block = _utils.SqueezeExcitation(8, 2)
        out = block(jt.random((2, 8, 4, 4)))
        self.assertEqual(tuple(out.shape), (2, 8, 4, 4))

    def test_mobilenet_v3_variant_uses_hardsigmoid(self):
        block = mobilenet_v3.SqueezeExcitation(16)
        self.assertEqual(type(block.scale_activation).__name__, "Hardsigmoid")
        self.assertEqual(type(block.activation).__name__, "ReLU")
        # squeeze_factor=4 with the divisible-by-8 rounding
        self.assertEqual(block.fc1.out_channels, _utils.make_divisible(4, 8))


class TestAffectedModelsStillBuild(unittest.TestCase):
    """Constructing and running each rewired model, so the refactor cannot
    quietly break one of them."""

    CASES = (
        ("efficientnet_b0", 64),
        ("regnet_x_400mf", 64),
        ("mobilenet_v3_small", 64),
        ("mobilenet_v2", 64),
        ("convnext_tiny", 64),
    )

    def test_forward(self):
        for name, size in self.CASES:
            if not hasattr(jtmodels, name):
                continue
            with self.subTest(model=name):
                model = getattr(jtmodels, name)(pretrained=False)
                model.eval()
                out = model(jt.random((1, 3, size, size)))
                self.assertEqual(tuple(out.shape), (1, 1000))


if __name__ == "__main__":
    unittest.main()
