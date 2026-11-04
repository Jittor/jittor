"""Torch arange dtype and large-bound regression on CPU and CUDA."""
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_arange_dtype_inference_and_large_integer_arithmetic(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    for kw in ({}, {'dtype':None}, {'dtype':torch.int64}):
        value=torch.arange(2147483648,2147483651,device=device,**kw)
        assert value.dtype==torch.int64
        np.testing.assert_array_equal(value.cpu().numpy(),[2147483648,2147483649,2147483650])
    value=torch.arange(-2147483650,-2147483647,device=device)
    np.testing.assert_array_equal(value.cpu().numpy(),[-2147483650,-2147483649,-2147483648])
    value=torch.arange(0,8589934593,4294967296,device=device)
    np.testing.assert_array_equal(value.cpu().numpy(),[0,4294967296,8589934592])
    value=torch.arange(0.,2.,0.5,device=device)
    assert value.dtype==torch.get_default_dtype()
    np.testing.assert_array_equal(value.cpu().numpy(),[0.,.5,1.,1.5])
    value=torch.arange(2147483648,2147483651,device=device,dtype=torch.float64)
    np.testing.assert_array_equal(value.cpu().numpy(),[2147483648.,2147483649.,2147483650.])


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_arange_uses_current_default_float_dtype(device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    before=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        value=torch.arange(0.,2.,0.5,device=device)
        assert value.dtype==torch.float64
        np.testing.assert_array_equal(value.cpu().numpy(),[0.,.5,1.,1.5])
        assert torch.arange(3,device=device).dtype==torch.int64
    finally:torch.set_default_dtype(before)


def test_native_jittor_default_arange_stays_int32():
    jt=pytest.importorskip('jittor')
    value=jt.arange(3)
    assert str(value.dtype)=='int32'
    np.testing.assert_array_equal(value.numpy(),[0,1,2])
