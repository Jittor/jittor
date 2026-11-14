"""Single-rounding CUDA SiLU forward for gradient-free dense FP32 inputs."""

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import optional_kernel


def _supports_silu_no_grad(x, inplace=False):
    return (not _output_requires_grad(x) and int(x.numel()) > 0
            and x._storage_is_contiguous())


@optional_kernel("nn.silu", "cuda", dtypes={"float32"}, supports=_supports_silu_no_grad)
def _silu_no_grad_cuda(x, inplace=False):
    return jt.code(x.shape, "float32", [x], cuda_src=r'''
        __global__ static void jt_silu_forward(const float* x, float* y, long long n) {
            long long stride = (long long)gridDim.x * blockDim.x;
            for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
                 i < n; i += stride)
                y[i] = x[i] / (1.f + expf(-x[i]));
        }
        long long n = in0->num;
        int blocks = (int)std::min((n + 255) / 256, 65535LL);
        jt_silu_forward<<<blocks, 256>>>(in0_p, out0_p, n);
        ''').stop_grad()
