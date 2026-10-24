"""FP32 CUDA SiLU with an explicit, differentiable rounded VJP.

A product's automatic derivative is algebraically equivalent but rounds in a
different order. Keep the SiLU scalar boundary, including the single-rounding
inner multiply-add, without changing the process-wide CUDA math policy.
"""

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import optional_kernel


_BACKWARD = r'''
__global__ static void silu_backward_fp32(
        const in0_type* x, const in1_type* dy, out0_type* dx,
        long long count) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; index < count; index += stride) {
        float value = (float)x[index];
        float sigmoid = 1.0f / (1.0f + expf(-value));
        float inner = fmaf(value, 1.0f - sigmoid, 1.0f);
        dx[index] = ((float)dy[index] * sigmoid) * inner;
    }
}
long long count = in0->num;
if (count) {
    int blocks = (int)((count + 255) / 256);
    if (blocks > 4096) blocks = 4096;
    silu_backward_fp32<<<blocks, 256>>>(in0_p, in1_p, out0_p, count);
    CHECK(0 == cudaGetLastError());
}
'''


class _SiLUBackwardCUDA(jt.Function):
    def _run_call(self, x, dy):
        # Function's execute inputs become stopped tape boundaries. Keep
        # independent holders of the original graph for higher derivatives,
        # as the native attention and matmul Functions do. Raw reshape also
        # protects this call from later rebinding of the caller's holders.
        self.saved = tuple(jt.ops.reshape(value, value.shape) for value in (x, dy))
        return super()._run_call(x, dy)

    def execute(self, x, dy):
        return jt.code(x.shape, x.dtype, [x.contiguous(), dy.contiguous()],
                       cuda_header="#include <math.h>", cuda_src=_BACKWARD)

    def grad(self, cotangent):
        x, dy = self.saved
        sigmoid = 1.0 / (1.0 + (-x).exp())
        complement = 1.0 - sigmoid
        first = sigmoid * (1.0 + x * complement)
        second = sigmoid * complement * (2.0 + x * (1.0 - 2.0 * sigmoid))
        # Both inputs matter: create_graph=True may differentiate the first
        # VJP with respect to either x or its incoming gradient dy.
        return cotangent * dy * second, cotangent * first


class _SiLUCUDA(jt.Function):
    def _run_call(self, x):
        self.saved = jt.ops.reshape(x, x.shape)
        return super()._run_call(x)

    def execute(self, x):
        return x / (1.0 + (-x).exp())

    def grad(self, dy):
        return _SiLUBackwardCUDA.apply(self.saved, dy)


@optional_kernel("nn.silu", "cuda", dtypes=("float32",))
def _silu_cuda(x, inplace=False):
    # The public frontend owns the existing inplace argument policy. Other
    # devices/dtypes retain their existing implementations and registrations.
    result = _SiLUCUDA.apply(x)
    return result if _output_requires_grad(x) else _stop_grad_outputs(result)


__all__ = []
