"""`.to()` has to accept a torch dtype object, not only a string.

torch spells a dtype as an object. Code written against torch hands it straight
to `.to()`, and `transformers/image_processing_backends.py:327` does exactly
that for every image it preprocesses::

    images = self.normalize(images.to(dtype=torch.float32), image_mean, image_std)

`_dtype_spec` accepted only `NanoString`, callables and strings, so both the
positional and the keyword form raised, and the failure surfaced from inside
transformers as `TypeError: to() expected dtype to be a dtype spelling`. Found
by running MiniMax-H3 reference-to-video, whose Qwen2-VL image processor takes
that path for each reference image.
"""
import unittest

import jittor as jt
import numpy as np
import pytest
import torch

from _rng_init_support import require_rng_device


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
@pytest.mark.parametrize("shape", [(), (0,), (2, 2)])
def test_dtype_only_noop_preserves_identity_layout_and_device(device, dtype, shape):
    require_rng_device(device)
    with jt.flag_scope(auto_graph_replay=0):
        source = torch.ones(shape, dtype=dtype, device=device)
        if shape == (2, 2):
            source = source.t()
        assert source.to(dtype) is source
        source.sync()
        assert source.location() == ("cpu" if device == "cpu" else "device")
        assert source.to(dtype) is source
        assert source.dtype == dtype and source.device.type == device
        if shape == (2, 2):
            assert not source.is_contiguous()
        copied = source.to(dtype, copy=True)
        assert copied is not source
        copied.fill_(3)
        copied.sync()
        np.testing.assert_array_equal(source.numpy(), np.ones(shape))
        np.testing.assert_array_equal(copied.numpy(), np.full(shape, 3))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dtype_only_noop_keeps_autograd_and_checks_other_arguments(device):
    require_rng_device(device)
    with jt.flag_scope(auto_graph_replay=0):
        source = torch.tensor([1., 2., 3.], device=device, requires_grad=True)
        same = source.to(torch.float32)
        assert same is source
        converted = same.to(torch.float64)
        assert converted is not source and converted.dtype == torch.float64
        assert converted.device.type == device
        converted.square().sum().backward()
        source.grad.sync()
        assert source.grad.location() == ("cpu" if device == "cpu" else "device")
        np.testing.assert_array_equal(source.grad.numpy(), [2., 4., 6.])
        with pytest.raises(TypeError, match="unexpected keyword"):
            source.to(torch.float32, unknown=True)
        with pytest.raises(NotImplementedError, match="contiguous_format"):
            source.to(torch.float32, memory_format="channels_last")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["to", "type_as"])
@pytest.mark.parametrize("shape", [(), (0,), (2, 2)])
def test_tensor_target_noop_preserves_identity_and_layout(device, method, shape):
    require_rng_device(device)
    source = torch.ones(shape, device=device)
    if shape == (2, 2):
        source = source.t()
    target = torch.zeros(1, device=device)
    assert getattr(source, method)(target) is source
    source.sync()
    assert source.location() == ("cpu" if device == "cpu" else "device")
    assert getattr(source, method)(target) is source
    if shape == (2, 2):
        assert not source.is_contiguous()
    copied = source.to(target, copy=True)
    assert copied is not source
    copied.fill_(3)
    np.testing.assert_array_equal(source.numpy(), np.ones(shape))
    np.testing.assert_array_equal(copied.numpy(), np.full(shape, 3))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("method", ["to", "type_as"])
def test_tensor_target_conversion_preserves_gradient_and_argument_checks(device, method):
    require_rng_device(device)
    source = torch.tensor([1., 2., 3.], device=device, requires_grad=True)
    target = torch.zeros(1, device=device)
    same = getattr(source, method)(target)
    assert same is source
    converted = getattr(same, method)(target.to(torch.float64))
    assert converted is not source and converted.dtype == torch.float64
    assert converted.device.type == device
    converted.square().sum().backward()
    np.testing.assert_array_equal(source.grad.numpy(), [2., 4., 6.])
    with pytest.raises(TypeError, match="unexpected keyword"):
        source.to(target, unknown=True)
    with pytest.raises(NotImplementedError, match="contiguous_format"):
        source.to(target, memory_format="channels_last")


@pytest.mark.parametrize("method", ["to", "type_as"])
def test_tensor_target_still_migrates_and_restores_host_parked_cuda_storage(method):
    require_rng_device("cuda")
    source = torch.tensor([2., 3.], device="cpu", requires_grad=True)
    target = torch.zeros(1, device="cuda")
    moved = getattr(source, method)(target)
    assert moved is not source and moved.device.type == "cuda"
    moved.square().sum().backward()
    np.testing.assert_array_equal(source.grad.numpy(), [4., 6.])
    moved.sync()
    moved.migrate_to_cpu()
    assert moved.placement_backend > 0 and moved.location() == "cpu"
    restored = getattr(moved, method)(target)
    restored.sync()
    assert restored.location() == "device"
    np.testing.assert_array_equal(restored.numpy(), [2., 3.])


class TestToAcceptsTorchDtypes(unittest.TestCase):
    def setUp(self):
        self.x = jt.ones((2, 2))

    def test_positional_dtype_object(self):
        self.assertEqual(str(self.x.to(torch.float32).dtype), "float32")

    def test_keyword_dtype_object(self):
        self.assertEqual(str(self.x.to(dtype=torch.float32).dtype), "float32")

    def test_half_precision_dtype_objects(self):
        for spelling, expected in ((torch.float16, "float16"),
                                   (torch.bfloat16, "bfloat16")):
            with self.subTest(str(spelling)):
                self.assertEqual(str(self.x.to(dtype=spelling).dtype), expected)

    def test_integer_dtype_objects(self):
        self.assertEqual(str(self.x.to(dtype=torch.int64).dtype), "int64")

    def test_string_spelling_still_works(self):
        self.assertEqual(str(self.x.to(dtype="float32").dtype), "float32")

    def test_a_device_object_is_still_a_device(self):
        # `torch.device` has no `.name`, which is what keeps the dtype branch
        # from swallowing it; assert that rather than trusting it.
        moved = self.x.to(torch.device("cpu"))
        self.assertEqual(str(moved.dtype), "float32")

    def test_a_mistyped_device_is_still_rejected(self):
        with self.assertRaises(TypeError):
            self.x.to("gpu")

    def test_dtype_and_device_together(self):
        out = self.x.to(device="cpu", dtype=torch.float16)
        self.assertEqual(str(out.dtype), "float16")


if __name__ == "__main__":
    unittest.main()
