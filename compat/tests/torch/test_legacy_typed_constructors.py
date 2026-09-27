"""Legacy typed CPU constructors retain CPU placement under a CUDA default."""
import os

import numpy as np
import pytest


@pytest.fixture
def torch_runtime():
    import torch

    cuda = os.environ.get("JITTOR_LEGACY_CONSTRUCTOR_DEVICE", "cpu") == "cuda"
    shim = hasattr(torch, "_torch_compat_install_context")
    old_default = torch.get_default_device()
    if shim:
        import jittor as jt
        old_cuda = jt.flags.use_cuda
        jt.flags.use_cuda = int(cuda)
    elif cuda:
        torch.set_default_device("cuda")
    if cuda:
        assert torch.cuda.is_available(), "CUDA acceptance requires an actual GPU"
        assert torch.get_default_device().type == "cuda"
    try:
        yield torch
        if cuda:
            assert torch.get_default_device().type == "cuda"
            if shim:
                assert jt.flags.use_cuda == 1
    finally:
        if shim:
            jt.flags.use_cuda = old_cuda
        elif cuda:
            torch.set_default_device(old_default)


@pytest.mark.parametrize("value,expected", [(-255, 1), (-129, 127), (-128, 128),
                                            (-112, 144), (-1, 255), (0, 0),
                                            (127, 127), (255, 255)])
def test_byte_sequence_accepts_signed_integer_values(torch_runtime, value, expected):
    torch = torch_runtime
    tensor = torch.ByteTensor([value])
    assert tensor.tolist() == [expected]
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.uint8


@pytest.mark.parametrize("value", [-256, 256, -.1, 255.1])
def test_byte_sequence_rejects_overflow(torch_runtime, value):
    with pytest.raises(RuntimeError, match="overflow"):
        torch_runtime.ByteTensor([value])


def test_byte_numpy_array_uses_numpy_cast_semantics(torch_runtime):
    tensor = torch_runtime.ByteTensor(np.array([-256, -112, -1, 256], dtype=np.int64))
    assert tensor.tolist() == [0, 144, 255, 0]
    assert tensor.device.type == "cpu"


@pytest.mark.parametrize("name", ["ByteTensor", "IntTensor", "FloatTensor"])
@pytest.mark.parametrize("kind", ["empty", "dimensions", "size", "data"])
def test_legacy_allocation_stays_on_cpu(torch_runtime, name, kind):
    torch = torch_runtime
    constructor = getattr(torch, name)
    args = {"empty": (), "dimensions": (2, 3),
            "size": (torch.Size([2, 3]),), "data": ([2, 3],)}[kind]
    result = constructor(*args)
    expected_shape = {"empty": (0,), "dimensions": (2, 3),
                      "size": (2, 3), "data": (2,)}[kind]
    assert tuple(result.shape) == expected_shape
    assert result.device.type == "cpu"
    assert not result.requires_grad
    if kind == "data":
        assert result.tolist() == [2, 3]


@pytest.mark.parametrize("dtype_name", ["int32", "float32"])
def test_tensor_input_rejects_mismatched_dtype(torch_runtime, dtype_name):
    torch = torch_runtime
    source = torch.tensor([1, 2], dtype=getattr(torch, dtype_name), device="cpu")
    with pytest.raises(TypeError):
        torch.ByteTensor(source)


def test_tensor_input_preserves_alias_and_grad(torch_runtime):
    torch = torch_runtime
    source = torch.tensor([1., 2.], dtype=torch.float32, device="cpu", requires_grad=True)
    result = torch.FloatTensor(source)
    assert result is not source
    assert result.data_ptr() == source.data_ptr()
    assert result.device.type == "cpu"
    assert result.requires_grad
    result.sum().backward()
    assert source.grad.tolist() == [1., 1.]


def test_tensor_input_rejects_explicit_device_keyword(torch_runtime):
    torch = torch_runtime
    source = torch.tensor([1, 2], dtype=torch.uint8, device="cpu")
    for requested in (None, "cpu"):
        with pytest.raises(RuntimeError, match="Legacy tensor constructor"):
            torch.ByteTensor(source, device=requested)


def test_legacy_keyword_validation(torch_runtime):
    torch = torch_runtime
    assert torch.ByteTensor([1], device="cpu").device.type == "cpu"
    with pytest.raises(RuntimeError, match="legacy constructor expects device type"):
        torch.ByteTensor([1], device="cuda")
    with pytest.raises(TypeError):
        torch.ByteTensor([1], dtype=torch.uint8)


def test_cuda_tensor_input_is_not_implicitly_copied_to_cpu(torch_runtime):
    if os.environ.get("JITTOR_LEGACY_CONSTRUCTOR_DEVICE", "cpu") != "cuda":
        pytest.skip("real CUDA tensor input requires the CUDA acceptance run")
    torch = torch_runtime
    source = torch.tensor([1, 2], dtype=torch.uint8, device="cuda")
    assert source.device.type == "cuda"
    with pytest.raises(TypeError):
        torch.ByteTensor(source)
    assert torch.tensor([1], device="cuda").device.type == "cuda"


@pytest.fixture
def cuda_runtime(torch_runtime):
    if os.environ.get("JITTOR_LEGACY_CONSTRUCTOR_DEVICE", "cpu") != "cuda":
        pytest.skip("CUDA typed constructor requires real CUDA acceptance")
    return torch_runtime


def _assert_cuda_storage(tensor):
    import ctypes

    assert tensor.device.type == "cuda"
    driver = ctypes.CDLL("libcuda.so.1")
    memory_type = ctypes.c_uint()
    assert driver.cuPointerGetAttribute(ctypes.byref(memory_type), 2,
                                        ctypes.c_uint64(tensor.data_ptr())) == 0
    assert memory_type.value == 2


@pytest.mark.parametrize("requested", [None, "cuda:0"])
def test_cuda_typed_data_construction(cuda_runtime, requested):
    torch = cuda_runtime
    result = torch.cuda.ByteTensor([-112, -1], device=requested)
    assert result.cpu().tolist() == [144, 255]
    _assert_cuda_storage(result)


def test_cuda_typed_tensor_input_preserves_alias_and_grad(cuda_runtime):
    torch = cuda_runtime
    source = torch.tensor([1., 2.], device="cuda", requires_grad=True)
    result = torch.cuda.FloatTensor(source)
    assert result is not source
    assert result.data_ptr() == source.data_ptr()
    assert result.requires_grad
    _assert_cuda_storage(result)
    result.sum().backward()
    assert source.grad.cpu().tolist() == [1., 1.]


@pytest.mark.parametrize("source_device,dtype_name", [("cpu", "uint8"), ("cuda", "int32")])
def test_cuda_typed_input_rejects_mismatched_options(cuda_runtime, source_device, dtype_name):
    torch = cuda_runtime
    source = torch.tensor([1, 2], dtype=getattr(torch, dtype_name), device=source_device)
    with pytest.raises(TypeError):
        torch.cuda.ByteTensor(source)


def test_cuda_typed_constructor_keyword_validation(cuda_runtime):
    torch = cuda_runtime
    with pytest.raises(RuntimeError, match="legacy constructor expects device type"):
        torch.cuda.ByteTensor([1], device="cpu")
    source = torch.tensor([1, 2], dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="Legacy tensor constructor"):
        torch.cuda.ByteTensor(source, device="cuda")
    with pytest.raises(TypeError):
        torch.cuda.ByteTensor([1], dtype=torch.uint8)
