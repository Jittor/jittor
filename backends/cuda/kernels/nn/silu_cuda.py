"""CUDA SiLU VJP arithmetic, called from the differentiable native owner."""

from jittor_core import Var
from jittor._core.dtypes import dtype_name
from jittor._runtime.dispatch import optional_kernel


def _supports_silu_vjp(x, seed):
    return (isinstance(x, Var) and isinstance(seed, Var)
            and dtype_name(x.dtype) in ("float16", "bfloat16")
            and tuple(x.shape) == tuple(seed.shape)
            and x._storage_is_contiguous() and seed._storage_is_contiguous()
            and x.numel() <= 2147483647)


@optional_kernel("nn.silu.backward_opmath", "cuda",
                 dtypes=("float16", "bfloat16", "float32"),
                 supports=_supports_silu_vjp)
def _silu_vjp_cuda(x, seed):
    import jittor as jt

    # The Function in activation.py owns higher derivatives. This kernel fixes
    # the first VJP's FP32 rounding boundaries independently of --fmad flags.
    return jt.code(x.shape, x.dtype, [x, seed], cuda_src=r'''
    __global__ static void silu_vjp(
            const in0_type* x, const in1_type* seed, out0_type* y, int size) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= size) return;
        float value = static_cast<float>(x[index]);
        float grad = static_cast<float>(seed[index]);
        float sigmoid = 1.0f / (1.0f + expf(-value));
        float affine = __fmaf_rn(value, 1.0f - sigmoid, 1.0f);
        y[index] = out0_type((grad * sigmoid) * affine);
    }
    int size = in0->num;
    if (size) silu_vjp<<<(size + 255) / 256, 256>>>(
        in0_p, in1_p, out0_p, size);
    CHECK(0 == cudaGetLastError());
    ''')
