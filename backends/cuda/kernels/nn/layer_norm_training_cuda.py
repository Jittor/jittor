"""CUDA training fast path for :func:`jittor.nn.layer_norm`."""

from functools import lru_cache
import math

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.backend_libraries import library_resource
from jittor._runtime.dispatch import optional_kernel


#: Rows at most this wide, when there are at least this many of them, go a
#: warp each in the forward and the input gradient; see `_warp_kernels`.
_WARP_ROW_LIMIT = 1024
_WARP_ROWS_MIN = 1024

#: Blocks of the fused backward: eight warps each, every warp walking rows.
#: Each leaves a partial row of each parameter gradient.
_FUSED_BLOCKS = 256



def _warp_kernels(hidden, eps, backward):
    """The forward and the input gradient with a warp per row.

    A block per row is mostly idle at these widths and pays its barriers for
    nothing: ViT-B/16's 12608 rows of 768 took 53 us forward and 115 us for
    the input gradient that way, the gradient reading `grad_y`, `x` and the
    weight three times over. Here a lane keeps its values of the row in
    registers and both reductions are shuffles, so each tensor is read once.
    """
    per = -(-hidden // 32)
    allsum = """
    __device__ __forceinline__ float jt_ln_allsum(float v) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        return v;
    }
    """
    forward = f"""
    __global__ static void layer_norm_forward_warp(
            const in0_type* x, const in1_type* weight, const in2_type* bias,
            out0_type* y, out1_type* mean, out2_type* rstd, long long rows) {{
        long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int lane = threadIdx.x & 31;
        if (row >= rows) return;
        const in0_type* xr = x + row * {hidden};
        float cache[{per}];
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            cache[i] = j < {hidden} ? static_cast<float>(xr[j]) : 0.0f;
            sum += cache[i];
        }}
        float m = jt_ln_allsum(sum) / {hidden}.0f;
        float var = 0.0f;
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            float d = cache[i] - m;
            if (lane + i * 32 < {hidden}) var += d * d;
        }}
        float r = rsqrtf(jt_ln_allsum(var) / {hidden}.0f + {eps:.9g}f);
        if (lane == 0) {{
            mean[row] = out1_type(m);
            rstd[row] = out2_type(r);
        }}
        out0_type* yr = y + row * {hidden};
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            if (j < {hidden})
                yr[j] = out0_type((cache[i] - m) * r * static_cast<float>(weight[j])
                                  + static_cast<float>(bias[j]));
        }}
    }}
    """
    # The backward, fused: each warp takes rows `_FUSED_BLOCKS * 8` apart and,
    # besides the row's input gradient, keeps its lanes' share of the
    # parameter gradients in registers, so `grad_y` and `x` are read once for
    # all three -- they were read again for the parameter gradients, which on
    # ViT's rows made that kernel as slow as the input gradient's. A block's
    # warps then fold theirs through shared memory into one partial row, and
    # `layer_norm_backward_finish` sums the blocks' rows in block order: lanes
    # across channels, the partials split down the block. (No comments inside
    # the source: `jt.code` lifts the kernels out by their text.)
    backward_x = f"""
    __global__ static void layer_norm_backward_fused(
            const in3_type* grad_y, const in0_type* x, const in5_type* mean,
            const in6_type* rstd, const in1_type* weight, out0_type* grad_x,
            float* partial, long long rows) {{
        __shared__ float fold[8][{hidden}];
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        float acc_w[{per}], acc_b[{per}], w[{per}];
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            acc_w[i] = acc_b[i] = 0.0f;
            w[i] = j < {hidden} ? static_cast<float>(weight[j]) : 0.0f;
        }}
        for (long long row = (long long)blockIdx.x * 8 + warp; row < rows;
             row += (long long)gridDim.x * 8) {{
            const in3_type* gr = grad_y + row * {hidden};
            const in0_type* xr = x + row * {hidden};
            float m = static_cast<float>(mean[row]), r = static_cast<float>(rstd[row]);
            float dy[{per}], xhat[{per}];
            float sum_g = 0.0f, sum_gx = 0.0f;
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                int j = lane + i * 32;
                bool in = j < {hidden};
                dy[i] = in ? static_cast<float>(gr[j]) : 0.0f;
                xhat[i] = in ? (static_cast<float>(xr[j]) - m) * r : 0.0f;
                float g = dy[i] * w[i];
                sum_g += g;
                sum_gx += g * xhat[i];
                acc_w[i] += dy[i] * xhat[i];
                acc_b[i] += dy[i];
            }}
            float mean_g = jt_ln_allsum(sum_g) / {hidden}.0f;
            float mean_gx = jt_ln_allsum(sum_gx) / {hidden}.0f;
            out0_type* dr = grad_x + row * {hidden};
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                int j = lane + i * 32;
                if (j < {hidden})
                    dr[j] = out0_type(r * (dy[i] * w[i] - mean_g - xhat[i] * mean_gx));
            }}
        }}
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {{
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                int j = lane + i * 32;
                if (j < {hidden}) fold[warp][j] = pass ? acc_b[i] : acc_w[i];
            }}
            __syncthreads();
            for (int j = threadIdx.x; j < {hidden}; j += 256) {{
                float t = 0.0f;
                #pragma unroll
                for (int k = 0; k < 8; k++) t += fold[k][j];
                partial[((long long)pass * gridDim.x + blockIdx.x) * {hidden} + j] = t;
            }}
            __syncthreads();
        }}
    }}
    __global__ static void layer_norm_backward_finish(
            const float* partial, out1_type* grad_weight, out2_type* grad_bias, int parts) {{
        __shared__ float fold[2][8][33];
        int channel = blockIdx.x * 32 + threadIdx.x;
        float tw = 0.0f, tb = 0.0f;
        if (channel < {hidden}) {{
            for (int p = threadIdx.y; p < parts; p += 8) {{
                tw += partial[(long long)p * {hidden} + channel];
                tb += partial[(long long)(parts + p) * {hidden} + channel];
            }}
        }}
        fold[0][threadIdx.y][threadIdx.x] = tw;
        fold[1][threadIdx.y][threadIdx.x] = tb;
        __syncthreads();
        if (threadIdx.y || channel >= {hidden}) return;
        tw = tb = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) {{
            tw += fold[0][k][threadIdx.x];
            tb += fold[1][k][threadIdx.x];
        }}
        grad_weight[channel] = out1_type(tw);
        grad_bias[channel] = out2_type(tb);
    }}
    """
    return allsum + (backward_x if backward else forward)


@lru_cache(maxsize=256)
def _layer_norm_cuda_sources(hidden, eps, warp, blocks):
    """Cache immutable programs; each call still builds fresh operators.

    CodeOp's grouped VJP receives x, weight, bias, dout, and the three saved
    outputs. Its native gradient construction avoids Python Function callbacks
    and tape nodes without retaining an execution graph or a backward result.
    """
    if warp:
        header = '#include "core/executor.h"'
        forward = _warp_kernels(hidden, eps, False) + f"""
        long long rows = in0->num / {hidden};
        layer_norm_forward_warp<<<(unsigned)((rows + 7) / 8), 256>>>(
            in0_p, in1_p, in2_p, out0_p, out1_p, out2_p, rows);
        CHECK(0 == cudaGetLastError());
        """
        backward = _warp_kernels(hidden, eps, True) + f"""
        auto scratch = runtime_executor().alloc_temp(sizeof(float) * {2 * blocks * hidden});
        float* partial = static_cast<float*>(scratch.ptr);
        long long rows = in3->num / {hidden};
        layer_norm_backward_fused<<<{blocks}, 256>>>(
            in3_p, in0_p, in5_p, in6_p, in1_p, out0_p, partial, rows);
        layer_norm_backward_finish<<<{-(-hidden // 32)}, dim3(32, 8)>>>(
            partial, out1_p, out2_p, {blocks});
        CHECK(0 == cudaGetLastError());
        """
        # Scratch is owned by this backward invocation, through the same
        # allocator as other native kernels, rather than a saved output.
        return header, forward, backward

    threads = 32
    while threads < min(hidden, 256):
        threads *= 2
    tile_x = 32
    tile_y = 32 if hidden >= 32 else 8
    header = f"#include <{library_resource('cub', 'home')}cub/cub.cuh>"
    forward = f"""
    __global__ static void layer_norm_forward(
            const in0_type* x, const in1_type* weight,
            const in2_type* bias, out0_type* y,
            out1_type* mean, out2_type* rstd, int rows) {{
        typedef cub::BlockReduce<float, {threads}> BlockReduce;
        __shared__ typename BlockReduce::TempStorage storage;
        __shared__ float mean_shared;
        __shared__ float rstd_shared;
        int row = blockIdx.x;
        if (row >= rows) return;
        int base = row * {hidden};
        // The row is read once and kept in registers: the mean
        // pass, the variance pass and the write all want the same
        // values. Re-reading made this move 4x the row where
        // torch's Welford moves 2x. Capped so a wide row falls
        // back to re-reading rather than spilling.
        constexpr int kPer = ({hidden} + {threads} - 1) / {threads};
        constexpr bool kCache = kPer <= 8;
        float cache[kCache ? kPer : 1];
        float local = 0.0f;
        if (kCache) {{
            int i = 0;
            for (int j = threadIdx.x; j < {hidden}; j += blockDim.x, ++i) {{
                cache[i] = static_cast<float>(x[base + j]);
                local += cache[i];
            }}
        }} else {{
            for (int j = threadIdx.x; j < {hidden}; j += blockDim.x)
                local += static_cast<float>(x[base + j]);
        }}
        float reduced = BlockReduce(storage).Sum(local);
        if (threadIdx.x == 0) {{
            mean_shared = reduced / {hidden}.0f;
            mean[row] = out1_type(mean_shared);
        }}
        __syncthreads();

        float row_mean = mean_shared;
        local = 0.0f;
        if (kCache) {{
            int i = 0;
            for (int j = threadIdx.x; j < {hidden}; j += blockDim.x, ++i) {{
                float delta = cache[i] - row_mean;
                local += delta * delta;
            }}
        }} else {{
            for (int j = threadIdx.x; j < {hidden}; j += blockDim.x) {{
                float delta = static_cast<float>(x[base + j]) - row_mean;
                local += delta * delta;
            }}
        }}
        __syncthreads();
        reduced = BlockReduce(storage).Sum(local);
        if (threadIdx.x == 0) {{
            rstd_shared = rsqrtf(
                reduced / {hidden}.0f + {eps:.9g}f);
            rstd[row] = out2_type(rstd_shared);
        }}
        __syncthreads();

        float row_rstd = rstd_shared;
        int wi = 0;
        for (int j = threadIdx.x; j < {hidden}; j += blockDim.x, ++wi) {{
            float xv = kCache ? cache[wi]
                              : static_cast<float>(x[base + j]);
            float normalized = (xv - row_mean) * row_rstd;
            y[base + j] = out0_type(
                normalized * static_cast<float>(weight[j])
                + static_cast<float>(bias[j]));
        }}
    }}
    int rows = in0->num / {hidden};
    layer_norm_forward<<<rows, {threads}>>>(
        in0_p, in1_p, in2_p, out0_p, out1_p, out2_p, rows);
    CHECK(0 == cudaGetLastError());
    """
    backward = f"""
    __global__ static void layer_norm_backward_x(
            const in3_type* grad_y, const in0_type* x,
            const in5_type* mean, const in6_type* rstd,
            const in1_type* weight, out0_type* grad_x,
            int rows) {{
        typedef cub::BlockReduce<float, {threads}> BlockReduce;
        __shared__ typename BlockReduce::TempStorage storage;
        __shared__ float mean_g_shared;
        __shared__ float mean_gx_shared;
        int row = blockIdx.x;
        if (row >= rows) return;
        int base = row * {hidden};
        float row_mean = static_cast<float>(mean[row]);
        float row_rstd = static_cast<float>(rstd[row]);
        float local = 0.0f;
        for (int j = threadIdx.x; j < {hidden}; j += blockDim.x)
            local += static_cast<float>(grad_y[base + j])
                * static_cast<float>(weight[j]);
        float reduced = BlockReduce(storage).Sum(local);
        if (threadIdx.x == 0)
            mean_g_shared = reduced / {hidden}.0f;
        __syncthreads();

        local = 0.0f;
        for (int j = threadIdx.x; j < {hidden}; j += blockDim.x) {{
            float normalized =
                (static_cast<float>(x[base + j]) - row_mean)
                * row_rstd;
            local += static_cast<float>(grad_y[base + j])
                * static_cast<float>(weight[j]) * normalized;
        }}
        __syncthreads();
        reduced = BlockReduce(storage).Sum(local);
        if (threadIdx.x == 0)
            mean_gx_shared = reduced / {hidden}.0f;
        __syncthreads();

        float mean_g = mean_g_shared;
        float mean_gx = mean_gx_shared;
        for (int j = threadIdx.x; j < {hidden}; j += blockDim.x) {{
            float normalized =
                (static_cast<float>(x[base + j]) - row_mean)
                * row_rstd;
            float g = static_cast<float>(grad_y[base + j])
                * static_cast<float>(weight[j]);
            grad_x[base + j] = out0_type(row_rstd * (
                g - mean_g - normalized * mean_gx));
        }}
    }}

    __global__ static void layer_norm_backward_affine(
            const in3_type* grad_y, const in0_type* x,
            const in5_type* mean, const in6_type* rstd,
            out1_type* grad_weight, out2_type* grad_bias,
            int rows) {{
        // The parameter gradients reduce down the rows, one result
        // per channel. Giving each block a channel and walking its
        // threads along rows makes every lane of a warp touch a
        // different cache line -- consecutive lanes would be
        // {hidden} floats apart. Lanes run along channels instead
        // and the rows are split across threadIdx.y, so a warp's
        // read is one contiguous transaction. Both gradients come
        // out of a single pass over grad_y rather than two.
        __shared__ float shared_weight[{tile_y}][{tile_x}];
        __shared__ float shared_bias[{tile_y}][{tile_x}];
        int channel = blockIdx.x * {tile_x} + threadIdx.x;
        float local_weight = 0.0f;
        float local_bias = 0.0f;
        if (channel < {hidden}) {{
            for (int row = threadIdx.y; row < rows;
                 row += {tile_y}) {{
                int index = row * {hidden} + channel;
                float g = static_cast<float>(grad_y[index]);
                float normalized =
                    (static_cast<float>(x[index])
                     - static_cast<float>(mean[row]))
                    * static_cast<float>(rstd[row]);
                local_weight += g * normalized;
                local_bias += g;
            }}
        }}
        shared_weight[threadIdx.y][threadIdx.x] = local_weight;
        shared_bias[threadIdx.y][threadIdx.x] = local_bias;
        __syncthreads();
        for (int stride = {tile_y} / 2; stride > 0; stride >>= 1) {{
            if (threadIdx.y < stride) {{
                shared_weight[threadIdx.y][threadIdx.x] +=
                    shared_weight[threadIdx.y + stride][threadIdx.x];
                shared_bias[threadIdx.y][threadIdx.x] +=
                    shared_bias[threadIdx.y + stride][threadIdx.x];
            }}
            __syncthreads();
        }}
        if (threadIdx.y == 0 && channel < {hidden}) {{
            grad_weight[channel] =
                out1_type(shared_weight[0][threadIdx.x]);
            grad_bias[channel] =
                out2_type(shared_bias[0][threadIdx.x]);
        }}
    }}

    int rows = in3->num / {hidden};
    layer_norm_backward_x<<<rows, {threads}>>>(
        in3_p, in0_p, in5_p, in6_p, in1_p, out0_p, rows);
    layer_norm_backward_affine
        <<<({hidden} + {tile_x} - 1) / {tile_x},
           dim3({tile_x}, {tile_y})>>>(
        in3_p, in0_p, in5_p, in6_p, out1_p, out2_p, rows);
    CHECK(0 == cudaGetLastError());
    """
    return header, forward, backward


def _supports_layer_norm_training(x, normalized_shape, weight, bias, eps):
    if not (
        _output_requires_grad(x, weight, bias)
        and isinstance(weight, jt.Var)
        and isinstance(bias, jt.Var)
    ):
        return False
    # This runs on every layer_norm of every step, so the cheap disqualifiers
    # go first and nothing is converted before it is needed: the two
    # `tuple(int(size) for size in ...)` builds and the `any(...)` genexpr used
    # to run even when the rank was already wrong. `x.shape` entries and
    # `numel()` are native ints already, and the last axis is the only one the
    # kernel reads, so only it is compared.
    if len(normalized_shape) != 1:
        return False
    shape = x.shape
    if not len(shape):
        return False
    hidden = shape[-1]
    if (normalized_shape[0] != hidden
            or weight.numel() != hidden
            or bias.numel() != hidden):
        return False
    for size in shape:
        if size <= 0:
            return False
    eps = float(eps)
    return eps > 0.0 and math.isfinite(eps)


@optional_kernel("nn.layer_norm.training", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("float32",),
                 supports=_supports_layer_norm_training)
def _layer_norm_cuda(x, normalized_shape, weight, bias, eps):
    shape = x.shape
    hidden = int(shape[-1])
    rows = int(x.numel()) // hidden
    warp = hidden <= _WARP_ROW_LIMIT and rows >= _WARP_ROWS_MIN
    blocks = min(_FUSED_BLOCKS, -(-rows // 8)) if warp else 0
    header, forward, backward = _layer_norm_cuda_sources(hidden, float(eps), warp, blocks)
    y, _, _ = jt.code(
        [shape, (rows,), (rows,)], [x.dtype, "float32", "float32"],
        [x, weight, bias], cuda_header=header, cuda_src=forward,
        cuda_grad_src=[backward], data={"multi_grad": 1})
    return y
