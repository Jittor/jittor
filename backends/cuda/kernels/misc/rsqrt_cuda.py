"""FP32 CUDA reciprocal square root with a differentiable rounded VJP.

Forward follows PyTorch v2.8.0 UnaryOpsKernel.cu float rsqrt overload.
VJP follows derivatives.yaml rsqrt: -0.5 * grad * result.pow(3).
PowKernel.cu specializes scalar exponent3 as (result*result)*result.
"""
import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import optional_kernel


_FORWARD = r'''
__global__ static void rsqrt_forward_fp32(
        const in0_type* x, out0_type* y, long long count) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; index < count; index += stride) {
        y[index] = rsqrtf((float)x[index]);
    }
}
long long count = in0->num;
if (count) {
    long long required = (count + 255) / 256;
    int blocks = required > 4096 ? 4096 : (int)required;
    rsqrt_forward_fp32<<<blocks, 256>>>(in0_p, out0_p, count);
    CHECK(0 == cudaGetLastError());
}
'''


class _RsqrtCUDA(jt.Function):
    def _run_call(self, x):
        # Retain an independent holder of the original graph before execute's
        # tape boundary, following the reviewed SiLU CUDA owner. No flag edits.
        self.saved_input = jt.ops.reshape(x, x.shape)
        return super()._run_call(x)

    def execute(self, x):
        return jt.code(x.shape, x.dtype, [x.contiguous()],
                       cuda_header="#include <math.h>", cuda_src=_FORWARD)

    def grad(self, dy):
        # Recompute the SAME deterministic rounded FP32 primitive on the saved
        # original graph, rather than retaining our own output/tape cycle or
        # saving a stopped execute input. This costs a forward kernel in VJP.
        # It remains a graph of differentiable ops for second/higher gradients.
        result = _RsqrtCUDA.apply(self.saved_input)
        cube = (result * result) * result
        return (-0.5 * dy) * cube


@optional_kernel("tensor.rsqrt", "cuda", dtypes=("float32",))
def _rsqrt_cuda(x):
    result = _RsqrtCUDA.apply(x)
    return result if _output_requires_grad(x) else _stop_grad_outputs(result)


__all__ = []
