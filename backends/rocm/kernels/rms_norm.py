"""Native HIP FP32 inference RMS normalization."""

from functools import lru_cache
import math

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs


def supported(x, weight, epsilon=1e-6):
    if not isinstance(x, jt.Var) or not isinstance(weight, jt.Var):
        return False
    if dtype_name(x.dtype) != 'float32' or dtype_name(weight.dtype) != 'float32':
        return False
    if _output_requires_grad(x, weight) or jt.flags.amp_reg:
        return False
    if not x.ndim or not 0 < x.shape[-1] <= 8192:
        return False
    if x.numel() // x.shape[-1] > 2147483647:
        return False
    if tuple(weight.shape) != (x.shape[-1],):
        return False
    return type(epsilon) in (int, float) and math.isfinite(epsilon) and epsilon > 0


@lru_cache(maxsize=128)
def _source(width, threads, epsilon):
    # A shared-memory tree deliberately avoids CUDA's 32-lane warp contract.
    # All threads participate, including when the width is not a power of two.
    return r'''
    __global__ static void hip_rms_norm(
            const float* x, const float* weight, float* y) {
        const long long row = blockIdx.x;
        const int tid = threadIdx.x;
        __shared__ float sums[%(threads)d];
        float sum = 0.0f;
        for (int i = tid; i < %(width)d; i += %(threads)d) {
            float v = x[row * %(width)d + i];
            sum += v * v;
        }
        sums[tid] = sum;
        __syncthreads();
        for (int stride = %(threads)d / 2; stride; stride >>= 1) {
            if (tid < stride) sums[tid] += sums[tid + stride];
            __syncthreads();
        }
        float inverse = rsqrtf(sums[0] / %(width)d.0f + %(epsilon).9ef);
        for (int i = tid; i < %(width)d; i += %(threads)d) {
            const long long index = row * %(width)d + i;
            y[index] = (x[index] * inverse) * weight[i];
        }
    }
    const long long rows = in0->num / %(width)d;
    if (rows) {
        hip_rms_norm<<<rows, %(threads)d>>>(in0_p, in1_p, out0_p);
        CHECK(hipGetLastError() == hipSuccess);
    }
    ''' % dict(width=width, threads=threads, epsilon=float(epsilon))


def rms_norm(x, weight, epsilon=1e-6):
    if not supported(x, weight, epsilon):
        return None
    width = int(x.shape[-1])
    threads = 1
    while threads < min(width, 256):
        threads *= 2
    return _stop_grad_outputs(jt.code(x.shape, x.dtype, [x, weight],
        cuda_header='#include <hip/hip_runtime.h>',
        cuda_src=_source(width, threads, float(epsilon))))
