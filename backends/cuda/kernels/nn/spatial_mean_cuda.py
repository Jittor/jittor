"""Fixed-order CUDA spatial means for gradient-free dense FP32 features."""

from functools import lru_cache

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import optional_kernel


def _supports_spatial_mean(x, keepdim=False):
    if len(x.shape) != 4 or _output_requires_grad(x) or not x._storage_is_contiguous():
        return False
    n, c, h, w = map(int, x.shape)
    spatial = h * w
    return (n * c >= 16 and 32 <= spatial <= 16384
            and (spatial < 128 or spatial % 4 == 0)
            and int(x.numel()) < 2**31)


@lru_cache(maxsize=256)
def _spatial_source(rows, spatial):
    # Four independent accumulators retain a reproducible FP32 reduction tree.
    # A warp handles each short row; long rows use all sixteen warps, folding
    # each warp before the shared-memory fold. No atomic summation is needed.
    vector = spatial >= 128
    split = spatial >= 8192
    stride = 512 if split else 32
    rows_per_block = 1 if split else 16
    return f'''
    __global__ static void jt_spatial_mean(const float* x, float* y) {{
        __shared__ float shared[512];
        int lane = threadIdx.x, warp = threadIdx.y;
        int row = blockIdx.x * {rows_per_block} + ({0 if split else 'warp'});
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int idx = lane + {32 if split else 0} * warp;
        if (row < {rows}) {{
            const float* input = x + (long long)row * {spatial};
            if ({int(vector)}) {{
                for (; idx * 4 + 3 < {spatial}; idx += {stride}) {{
                    a0 += input[idx * 4];
                    a1 += input[idx * 4 + 1];
                    a2 += input[idx * 4 + 2];
                    a3 += input[idx * 4 + 3];
                }}
            }} else {{
                for (; idx + 3 * {stride} < {spatial}; idx += 4 * {stride}) {{
                    a0 += input[idx];
                    a1 += input[idx + {stride}];
                    a2 += input[idx + 2 * {stride}];
                    a3 += input[idx + 3 * {stride}];
                }}
                if (idx < {spatial}) a0 += input[idx];
                if (idx + {stride} < {spatial}) a1 += input[idx + {stride}];
                if (idx + 2 * {stride} < {spatial}) a2 += input[idx + 2 * {stride}];
                if (idx + 3 * {stride} < {spatial}) a3 += input[idx + 3 * {stride}];
            }}
        }}
        float total = ((a0 + a1) + a2) + a3;
        __syncthreads();
        for (int offset = 16; offset; offset /= 2)
            total += __shfl_down_sync(0xffffffffu, total, offset);
        if ({int(split)}) {{
            shared[warp * 32 + lane] = total;
            for (int offset = 8; offset; offset /= 2) {{
                __syncthreads();
                if (warp < offset) {{
                    total += shared[(warp + offset) * 32 + lane];
                    shared[warp * 32 + lane] = total;
                }}
            }}
        }}
        if (lane == 0 && ({int(not split)} || warp == 0) && row < {rows})
            y[row] = total * (float({rows}) / float({rows * spatial}));
    }}
    jt_spatial_mean<<<({rows} + {rows_per_block} - 1) / {rows_per_block}, dim3(32, 16)>>>(
        in0_p, out0_p);
    '''


@optional_kernel("nn.spatial_mean", "cuda", dtypes={"float32"}, supports=_supports_spatial_mean)
def _spatial_mean_cuda(x, keepdim=False):
    n, c, h, w = map(int, x.shape)
    shape = (n, c, 1, 1) if keepdim else (n, c)
    return jt.code(shape, "float32", [x],
                   cuda_src=_spatial_source(n * c, h * w)).stop_grad()
