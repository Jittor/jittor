# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""EfficientNet architecture parity with torchvision.

The MBConv projection is a 1x1 conv + norm with **no** activation. Jittor's
``ConvBNActivation`` replaced ``activation_layer=None`` with ``nn.SiLU`` before
testing it against ``None``, so the test never failed and every projection got
a SiLU: one extra non-linearity per MBConv block, in b0 through b7.

The expected counts come from torchvision 0.27.1 (``torchvision.models``
``efficientnet_b0`` .. ``efficientnet_b7``, ``weights=None``), collected by
counting ``type(m).__name__`` over ``model.modules()``. torchvision is not
importable in the jittor test environment (different Python minor version), so
the numbers are pinned here rather than computed.
"""
import collections
import unittest

import jittor as jt
import jittor.models as jtmodels


# torchvision 0.27.1: (Conv2d, BatchNorm2d, SiLU, Sigmoid, SqueezeExcitation,
#                      MBConv, StochasticDepth, Linear)
TORCHVISION_MODULE_COUNTS = {
    "efficientnet_b0": (81, 49, 49, 16, 16, 16, 16, 1),
    "efficientnet_b1": (115, 69, 69, 23, 23, 23, 23, 1),
    "efficientnet_b2": (115, 69, 69, 23, 23, 23, 23, 1),
    "efficientnet_b3": (130, 78, 78, 26, 26, 26, 26, 1),
    "efficientnet_b4": (160, 96, 96, 32, 32, 32, 32, 1),
    "efficientnet_b5": (194, 116, 116, 39, 39, 39, 39, 1),
    "efficientnet_b6": (224, 134, 134, 45, 45, 45, 45, 1),
    "efficientnet_b7": (273, 163, 163, 55, 55, 55, 55, 1),
}

# jittor spells the same layers with its own class names
JITTOR_NAMES = ("Conv", "BatchNorm", "SiLU", "Sigmoid", "SqueezeExcitation",
                "MBConv", "StochasticDepth", "Linear")


def _module_counts(model):
    counter = collections.Counter(type(m).__name__ for m in model.modules())
    return tuple(counter[name] for name in JITTOR_NAMES)


class TestEfficientNetTorchvisionParity(unittest.TestCase):

    def test_module_counts_match_torchvision(self):
        for arch, expected in TORCHVISION_MODULE_COUNTS.items():
            with self.subTest(arch=arch):
                model = getattr(jtmodels, arch)(pretrained=False)
                got = _module_counts(model)
                self.assertEqual(
                    got, expected,
                    msg="%s: %s\nexpected (torchvision) %s\ngot %s" % (
                        arch, JITTOR_NAMES, expected, got))

    def test_projection_has_no_activation(self):
        # every MBConv ends with [Conv, BatchNorm] -- conv + norm, nothing else
        model = jtmodels.efficientnet_b0(pretrained=False)
        blocks = [m for m in model.modules()
                  if type(m).__name__ == "MBConv"]
        self.assertEqual(len(blocks), 16)
        for index, block in enumerate(blocks):
            projection = list(block.block)[-1]
            layers = [type(m).__name__ for m in projection]
            self.assertEqual(
                layers, ["Conv", "BatchNorm"],
                msg="MBConv[%d] projection is %s; torchvision has "
                    "[Conv2d, BatchNorm2d] and no activation" % (index, layers))

    def test_expand_and_depthwise_keep_their_activation(self):
        # the fix must not remove the activations that torchvision does have
        model = jtmodels.efficientnet_b0(pretrained=False)
        blocks = [m for m in model.modules() if type(m).__name__ == "MBConv"]
        # block 1 of b0 has expand_ratio > 1, so expand + depthwise + project
        with_expand = [b for b in blocks if len(list(b.block)) == 4]
        assert with_expand, "expected at least one MBConv with an expand conv"
        for block in with_expand:
            children = list(block.block)
            for sub in children[:2]:
                self.assertEqual(
                    [type(m).__name__ for m in sub],
                    ["Conv", "BatchNorm", "SiLU"])

    def test_forward_still_runs(self):
        model = jtmodels.efficientnet_b0(pretrained=False)
        model.eval()
        out = model(jt.random((1, 3, 64, 64)))
        self.assertEqual(tuple(out.shape), (1, 1000))


if __name__ == "__main__":
    unittest.main()
