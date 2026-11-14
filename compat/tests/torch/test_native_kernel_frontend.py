"""The functional kernel API must preserve the Torch frontend and its gradients."""
import numpy as np
import pytest
import jittor as jt
import torch

from jittor.kernels import TensorSpec, triton_call, tilelang_call, _Buffer
from _helpers import capability


def test_executor_buffers_bypass_the_eager_triton_bridge():
    from jittor.compat.triton import _args_have_jittor_var
    buffer = _Buffer((0, (4,), (1,), "float32", None), ("cuda", 0), 2)
    assert not _args_have_jittor_var((buffer,), {"output": buffer})
    assert _args_have_jittor_var((torch.zeros((4,), device="cpu"),), {})


def require_cuda():
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("CUDA required")


def check_double(forward):
    class Double(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return forward(x)
        @staticmethod
        def backward(ctx, dy):
            return dy * 2
    x = torch.arange(32, device="cuda", dtype=torch.float32).requires_grad_()
    y = Double.apply(x)
    assert isinstance(y, torch.Tensor)
    np.testing.assert_array_equal(y.detach().cpu().numpy(), np.arange(32) * 2)
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.cpu().numpy(), np.full(32, 2))


def test_triton_native_kernel_with_torch_frontend():
    require_cuda()
    from jittor.compat.triton import backend
    triton = backend.real_triton()
    if triton is None:
        pytest.skip("Triton compiler is not installed")
    import triton.language as tl
    @triton.jit
    def double(X, Y):
        i = tl.arange(0, 32)
        tl.store(Y + i, tl.load(X + i) * 2)
    check_double(lambda x: triton_call(double, x, TensorSpec(x.shape, x.dtype), grid=(1,)))


def test_tilelang_native_kernel_with_torch_frontend():
    require_cuda()
    tilelang = pytest.importorskip("tilelang")
    import tilelang.language as T
    @T.prim_func
    def double(X: T.Tensor((32,), "float32"), Y: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                Y[i] = X[i] * 2
    kernel = tilelang.compile(double, out_idx=[], target="cuda", execution_backend="tvm_ffi")
    check_double(lambda x: tilelang_call(kernel, x, TensorSpec(x.shape, x.dtype)))
