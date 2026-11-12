"""Scalar readback preserves CUDA storage and all live aliases."""
import ctypes
import unittest

import torch


def _pointer_kind(tensor):
    driver = ctypes.CDLL('libcuda.so.1')
    kind = ctypes.c_int(-1)
    pointer = int(tensor.data_ptr())
    status = driver.cuPointerGetAttribute(ctypes.byref(kind), 2,
                                         ctypes.c_void_p(pointer))
    if status:
        raise RuntimeError('CUDA could not classify the tensor pointer: ' + str(status))
    return pointer, kind.value


@unittest.skipUnless(torch.cuda.is_available(), 'requires real CUDA')
class TestTensorItemResidency(unittest.TestCase):
    def test_scalar_and_views_keep_their_actual_device_storage(self):
        for dtype in (torch.float16, torch.float32):
            for geometry in ('scalar', 'one_element', 'view', 'view_scalar'):
                with self.subTest(dtype=dtype, geometry=geometry):
                    base = None
                    if geometry == 'scalar':
                        tensor = torch.tensor(1.25, dtype=dtype, device='cuda')
                    elif geometry == 'one_element':
                        tensor = torch.tensor([1.25], dtype=dtype, device='cuda')
                    else:
                        base = torch.tensor([9., 1.25, 3.], dtype=dtype, device='cuda')
                        tensor = base[1:2] if geometry == 'view' else base[1]
                    torch.cuda.synchronize()
                    before = _pointer_kind(tensor)
                    base_before = _pointer_kind(base) if base is not None else None
                    self.assertEqual(before[1], 2)
                    self.assertEqual(tensor.item(), 1.25)
                    self.assertEqual(_pointer_kind(tensor), before)
                    if base is not None:
                        self.assertEqual(_pointer_kind(base), base_before)
                    # A fresh allocation must not recycle the still-live source.
                    other = torch.tensor([9.], dtype=dtype, device='cuda')
                    torch.cuda.synchronize()
                    self.assertNotEqual(_pointer_kind(other)[0], before[0])
                    self.assertEqual(_pointer_kind(tensor), before)
                    self.assertEqual(tensor.device.type, 'cuda')

    def test_python_scalar_types(self):
        for dtype, value, kind in ((torch.int64, 7, int),
                                  (torch.bool, True, bool),
                                  (torch.float64, 1.25, float)):
            with self.subTest(dtype=dtype):
                tensor = torch.tensor(value, dtype=dtype, device='cuda')
                torch.cuda.synchronize()
                before = _pointer_kind(tensor)
                actual = tensor.item()
                self.assertIs(type(actual), kind)
                self.assertEqual(actual, value)
                self.assertEqual(_pointer_kind(tensor), before)
                self.assertEqual(before[1], 2)

    def test_non_scalar_error_keeps_cuda_source(self):
        tensor = torch.tensor([1., 2.], device='cuda')
        torch.cuda.synchronize()
        before = _pointer_kind(tensor)
        with self.assertRaises(RuntimeError):
            tensor.item()
        self.assertEqual(_pointer_kind(tensor), before)
        self.assertEqual(before[1], 2)


if __name__ == '__main__':
    unittest.main()
