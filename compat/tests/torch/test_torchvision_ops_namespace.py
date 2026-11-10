"""torchvision.ops is a package so submodule imports work (timm needs ops.misc)."""

from __future__ import print_function

import unittest


class TestTorchvisionOpsNamespace(unittest.TestCase):
    def test_ops_submodule_import(self):
        import torchvision.ops.misc
        from torchvision.ops.misc import FrozenBatchNorm2d

        self.assertIsNotNone(FrozenBatchNorm2d)

    def test_ops_public_symbols_survive(self):
        import torchvision

        self.assertTrue(callable(torchvision.ops.box_convert))


if __name__ == '__main__':
    unittest.main()
