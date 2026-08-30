"""OpenMMLab import regressions for the Jittor Torch compatibility layer.

``mmcv-lite`` and ``mmengine`` keep their model definitions in pure Python,
but their import path still reads the CUDA dtype constructor names exposed by
PyTorch (for example ``torch.cuda.BoolTensor``).  The test deliberately avoids
``mmcv.ops``: that optional module is a separately compiled PyTorch extension
and cannot be made ABI-compatible by a Python shim alone.
"""

import importlib.util
import sys
import unittest

import numpy as np

import jittor as torch


_HAS_MMCV = importlib.util.find_spec("mmcv") is not None
_HAS_MMENGINE = importlib.util.find_spec("mmengine") is not None


class TestCudaTypedTensorCompat(unittest.TestCase):
    def test_cuda_dtype_constructors_preserve_dtype_and_device_checks(self):
        self.assertIs(sys.modules.get("torch"), torch)
        for name in (
            "FloatTensor", "DoubleTensor", "HalfTensor", "BFloat16Tensor",
            "LongTensor", "IntTensor", "ShortTensor", "CharTensor",
            "ByteTensor", "BoolTensor",
        ):
            with self.subTest(name=name):
                cuda_type = getattr(torch.cuda, name)
                self.assertIsNot(cuda_type, getattr(torch, name))
                self.assertEqual(cuda_type.__module__, "torch.cuda")

        cpu_bool = torch.tensor([True], dtype=torch.bool, device="cpu")
        cpu_long = torch.tensor([1], dtype=torch.int64, device="cpu")
        self.assertIsInstance(cpu_bool, torch.BoolTensor)
        self.assertIsInstance(cpu_long, torch.LongTensor)
        self.assertNotIsInstance(cpu_bool, torch.cuda.BoolTensor)
        self.assertNotIsInstance(cpu_long, torch.cuda.LongTensor)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA or an accelerator backend")
    def test_cuda_dtype_constructor_executes_on_device(self):
        previous = torch.flags.use_cuda
        try:
            value = torch.cuda.FloatTensor([1.0, 2.0])
            self.assertTrue(value.is_cuda)
            self.assertIsInstance(value, torch.cuda.FloatTensor)

            result = value * 3.0
            self.assertTrue(result.is_cuda)
            np.testing.assert_allclose(result.numpy(), np.array([3.0, 6.0], dtype=np.float32))
        finally:
            torch.sync_all()
            torch.flags.use_cuda = previous


@unittest.skipUnless(_HAS_MMCV and _HAS_MMENGINE, "needs mmcv-lite and mmengine")
class TestMmcvCompat(unittest.TestCase):
    def test_mmcv_and_mmengine_model_imports(self):
        # These are real downstream imports, rather than a hand-written module
        # stub.  Importing mmcv.ops is intentionally outside this contract (see
        # the module docstring above).
        self.assertIs(torch.utils, sys.modules["torch.utils"])
        self.assertIs(
            torch.utils.checkpoint,
            sys.modules["torch.utils.checkpoint"],
        )
        import mmcv.cnn
        import mmengine.model

        self.assertTrue(hasattr(mmcv.cnn, "build_norm_layer"))
        self.assertTrue(hasattr(mmengine.model, "BaseModel"))

    def test_mmcv_conv_module_constructs_with_torch_conv_attributes(self):
        from mmcv.cnn import ConvModule

        module = ConvModule(
            3,
            4,
            3,
            padding=1,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU"),
        )
        self.assertFalse(module.transposed)
        self.assertEqual(module.output_padding, (0, 0))


if __name__ == "__main__":
    unittest.main()
