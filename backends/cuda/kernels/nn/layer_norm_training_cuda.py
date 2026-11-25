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
_WARP_ROWS_MIN = 64

#: Blocks of the fused backward: each warp walks rows and leaves a partial row
#: of each parameter gradient.
_FUSED_BLOCKS = 256


def _warp_warps_per_block(rows, hidden, backward):
    if rows <= 256:
        return 2
    if backward and hidden > 512:
        return 4
    if rows < 1024:
        return 4
    return 8


def _warp_kernels(hidden, backward, wpb=8):
    """The forward and the fused backward with a warp per row.

    A block per row is mostly idle at these widths and pays its barriers for
    nothing. Here a lane keeps its values of the row in registers, both
    reductions are shuffles, and aligned float32 widths move float4 vectors so
    each tensor is read once.
    """
    per = -(-hidden // 32)
    vec4_n = hidden // 4
    vec4_per = -(-vec4_n // 32) if vec4_n > 0 else 1
    threads = wpb * 32
    allsum = """
    __device__ __forceinline__ float jt_ln_allsum(float v) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        return v;
    }
    """
    forward = f"""
    __global__ static void layer_norm_forward_warp(
            const in0_type* __restrict__ x, const in1_type* __restrict__ weight,
            const in2_type* __restrict__ bias, out0_type* __restrict__ y,
            out1_type* __restrict__ mean, out2_type* __restrict__ rstd,
            long long rows, float eps) {{
        long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int lane = threadIdx.x & 31;
        if (row >= rows) return;
        const in0_type* xr = x + row * {hidden}LL;
        out0_type* yr = y + row * {hidden}LL;
        if (sizeof(in0_type) == 4 && sizeof(out0_type) == 4 && ({hidden} % 4 == 0)
                && ((((size_t)x | (size_t)y) & 15u) == 0)) {{
            const float4* x4 = reinterpret_cast<const float4*>(xr);
            float4* y4 = reinterpret_cast<float4*>(yr);
            bool wb_vec = (sizeof(in1_type) == 4 && sizeof(in2_type) == 4)
                          && ((((size_t)weight | (size_t)bias) & 15u) == 0);
            const float4* w4 = wb_vec ? reinterpret_cast<const float4*>(weight) : nullptr;
            const float4* b4 = wb_vec ? reinterpret_cast<const float4*>(bias) : nullptr;
            float4 cache[{vec4_per}];
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < {vec4_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec4_n}) {{
                    float4 val = x4[v];
                    cache[i] = val;
                    sum += val.x + val.y + val.z + val.w;
                }}
            }}
            float m = jt_ln_allsum(sum) / {hidden}.0f;
            float sum_d = 0.0f;
            float var = 0.0f;
            #pragma unroll
            for (int i = 0; i < {vec4_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec4_n}) {{
                    float dx = cache[i].x - m;
                    float dy = cache[i].y - m;
                    float dz = cache[i].z - m;
                    float dw = cache[i].w - m;
                    sum_d += dx + dy + dz + dw;
                    var += dx * dx + dy * dy + dz * dz + dw * dw;
                }}
            }}
            float dm = jt_ln_allsum(sum_d) / {hidden}.0f;
            m += dm;
            float v_mean = jt_ln_allsum(var) / {hidden}.0f - dm * dm;
            float r = rsqrtf((v_mean > 0.0f ? v_mean : 0.0f) + eps);
            if (lane == 0) {{
                mean[row] = out1_type(m);
                rstd[row] = out2_type(r);
            }}
            #pragma unroll
            for (int i = 0; i < {vec4_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec4_n}) {{
                    float4 val = cache[i];
                    float4 wv, bv;
                    if (wb_vec) {{
                        wv = w4[v];
                        bv = b4[v];
                    }} else {{
                        int d = v * 4;
                        wv = make_float4(static_cast<float>(weight[d]),
                                         static_cast<float>(weight[d + 1]),
                                         static_cast<float>(weight[d + 2]),
                                         static_cast<float>(weight[d + 3]));
                        bv = make_float4(static_cast<float>(bias[d]),
                                         static_cast<float>(bias[d + 1]),
                                         static_cast<float>(bias[d + 2]),
                                         static_cast<float>(bias[d + 3]));
                    }}
                    float4 outv;
                    outv.x = (val.x - m) * r * wv.x + bv.x;
                    outv.y = (val.y - m) * r * wv.y + bv.y;
                    outv.z = (val.z - m) * r * wv.z + bv.z;
                    outv.w = (val.w - m) * r * wv.w + bv.w;
                    y4[v] = outv;
                }}
            }}
            return;
        }}
        float cache[{per}];
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            cache[i] = j < {hidden} ? static_cast<float>(xr[j]) : 0.0f;
            sum += cache[i];
        }}
        float m = jt_ln_allsum(sum) / {hidden}.0f;
        float sum_d = 0.0f;
        float var = 0.0f;
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            float d = cache[i] - m;
            if (lane + i * 32 < {hidden}) {{
                sum_d += d;
                var += d * d;
            }}
        }}
        float dm = jt_ln_allsum(sum_d) / {hidden}.0f;
        m += dm;
        float v_mean = jt_ln_allsum(var) / {hidden}.0f - dm * dm;
        float r = rsqrtf((v_mean > 0.0f ? v_mean : 0.0f) + eps);
        if (lane == 0) {{
            mean[row] = out1_type(m);
            rstd[row] = out2_type(r);
        }}
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            if (j < {hidden})
                yr[j] = out0_type((cache[i] - m) * r * static_cast<float>(weight[j])
                                  + static_cast<float>(bias[j]));
        }}
    }}
    """
    # The backward, fused: each warp takes rows `gridDim.x * wpb` apart and
    # keeps its lanes' share of parameter gradients in registers. Weight lives
    # in shared memory rather than per-thread registers so wide rows (e.g. 768)
    # stay at much higher occupancy. (No comments inside the source: `jt.code`
    # lifts the kernels out by their text.)
    backward_x = f"""
    __global__ static void layer_norm_backward_fused_vec4(
            const in3_type* __restrict__ grad_y, const in0_type* __restrict__ x,
            const in5_type* __restrict__ mean, const in6_type* __restrict__ rstd,
            const in1_type* __restrict__ weight, out0_type* __restrict__ grad_x,
            float* __restrict__ partial, long long rows) {{
        __shared__ __align__(16) float fold[{wpb}][{hidden}];
        __shared__ __align__(16) float sw[{hidden}];
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int j = threadIdx.x; j < {hidden}; j += {threads})
            sw[j] = static_cast<float>(weight[j]);
        __syncthreads();
        float4 acc_w[{vec4_per}], acc_b[{vec4_per}];
        #pragma unroll
        for (int i = 0; i < {vec4_per}; i++) {{
            acc_w[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            acc_b[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }}
        const float4* sw4 = reinterpret_cast<const float4*>(sw);
        for (long long row = (long long)blockIdx.x * {wpb} + warp; row < rows;
             row += (long long)gridDim.x * {wpb}) {{
            const float4* gr4 = reinterpret_cast<const float4*>(grad_y + row * {hidden}LL);
            const float4* xr4 = reinterpret_cast<const float4*>(x + row * {hidden}LL);
            float4* dr4 = reinterpret_cast<float4*>(grad_x + row * {hidden}LL);
            float m = static_cast<float>(mean[row]), r = static_cast<float>(rstd[row]);
            float4 gv[{vec4_per}], xhat[{vec4_per}];
            float sum_g = 0.0f, sum_gx = 0.0f;
            #pragma unroll
            for (int i = 0; i < {vec4_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec4_n}) {{
                    float4 dy = gr4[v];
                    float4 xv = xr4[v];
                    float4 wv = sw4[v];
                    float4 xh = make_float4((xv.x - m) * r, (xv.y - m) * r,
                                            (xv.z - m) * r, (xv.w - m) * r);
                    float4 g = make_float4(dy.x * wv.x, dy.y * wv.y,
                                           dy.z * wv.z, dy.w * wv.w);
                    gv[i] = g;
                    xhat[i] = xh;
                    sum_g += g.x + g.y + g.z + g.w;
                    sum_gx += g.x * xh.x + g.y * xh.y + g.z * xh.z + g.w * xh.w;
                    acc_w[i].x += dy.x * xh.x;
                    acc_w[i].y += dy.y * xh.y;
                    acc_w[i].z += dy.z * xh.z;
                    acc_w[i].w += dy.w * xh.w;
                    acc_b[i].x += dy.x;
                    acc_b[i].y += dy.y;
                    acc_b[i].z += dy.z;
                    acc_b[i].w += dy.w;
                }}
            }}
            float mean_g = jt_ln_allsum(sum_g) / {hidden}.0f;
            float mean_gx = jt_ln_allsum(sum_gx) / {hidden}.0f;
            #pragma unroll
            for (int i = 0; i < {vec4_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec4_n}) {{
                    float4 outv;
                    outv.x = r * (gv[i].x - mean_g - xhat[i].x * mean_gx);
                    outv.y = r * (gv[i].y - mean_g - xhat[i].y * mean_gx);
                    outv.z = r * (gv[i].z - mean_g - xhat[i].z * mean_gx);
                    outv.w = r * (gv[i].w - mean_g - xhat[i].w * mean_gx);
                    dr4[v] = outv;
                }}
            }}
        }}
        float4* f4 = reinterpret_cast<float4*>(fold[warp]);
        #pragma unroll
        for (int pass = 0; pass < 2; pass++) {{
            #pragma unroll
            for (int i = 0; i < {vec4_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec4_n}) f4[v] = pass ? acc_b[i] : acc_w[i];
            }}
            __syncthreads();
            for (int j = threadIdx.x; j < {hidden}; j += {threads}) {{
                float t = 0.0f;
                #pragma unroll
                for (int k = 0; k < {wpb}; k++) t += fold[k][j];
                partial[((long long)pass * gridDim.x + blockIdx.x) * {hidden} + j] = t;
            }}
            __syncthreads();
        }}
    }}
    __global__ static void layer_norm_backward_fused(
            const in3_type* __restrict__ grad_y, const in0_type* __restrict__ x,
            const in5_type* __restrict__ mean, const in6_type* __restrict__ rstd,
            const in1_type* __restrict__ weight, out0_type* __restrict__ grad_x,
            float* __restrict__ partial, long long rows) {{
        __shared__ float fold[{wpb}][{hidden}];
        __shared__ float sw[{hidden}];
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int j = threadIdx.x; j < {hidden}; j += {threads})
            sw[j] = static_cast<float>(weight[j]);
        __syncthreads();
        float acc_w[{per}], acc_b[{per}];
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            acc_w[i] = acc_b[i] = 0.0f;
        }}
        for (long long row = (long long)blockIdx.x * {wpb} + warp; row < rows;
             row += (long long)gridDim.x * {wpb}) {{
            const in3_type* gr = grad_y + row * {hidden}LL;
            const in0_type* xr = x + row * {hidden}LL;
            float m = static_cast<float>(mean[row]), r = static_cast<float>(rstd[row]);
            float gv[{per}], xhat[{per}];
            float sum_g = 0.0f, sum_gx = 0.0f;
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                int j = lane + i * 32;
                bool in = j < {hidden};
                float dy = in ? static_cast<float>(gr[j]) : 0.0f;
                float xh = in ? (static_cast<float>(xr[j]) - m) * r : 0.0f;
                float g = in ? dy * sw[j] : 0.0f;
                gv[i] = g;
                xhat[i] = xh;
                sum_g += g;
                sum_gx += g * xh;
                acc_w[i] += dy * xh;
                acc_b[i] += dy;
            }}
            float mean_g = jt_ln_allsum(sum_g) / {hidden}.0f;
            float mean_gx = jt_ln_allsum(sum_gx) / {hidden}.0f;
            out0_type* dr = grad_x + row * {hidden}LL;
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                int j = lane + i * 32;
                if (j < {hidden})
                    dr[j] = out0_type(r * (gv[i] - mean_g - xhat[i] * mean_gx));
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
            for (int j = threadIdx.x; j < {hidden}; j += {threads}) {{
                float t = 0.0f;
                #pragma unroll
                for (int k = 0; k < {wpb}; k++) t += fold[k][j];
                partial[((long long)pass * gridDim.x + blockIdx.x) * {hidden} + j] = t;
            }}
            __syncthreads();
        }}
    }}
    __global__ static void layer_norm_backward_finish(
            const float* __restrict__ partial, out1_type* __restrict__ grad_weight,
            out2_type* __restrict__ grad_bias, int parts) {{
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
def _layer_norm_cuda_sources(hidden, warp, blocks, fwd_wpb=8, bwd_wpb=8):
    """Cache immutable programs; each call still builds fresh operators.

    CodeOp's grouped VJP receives x, weight, bias, dout, and the three saved
    outputs. Its native gradient construction avoids Python Function callbacks
    and tape nodes without retaining an execution graph or a backward result.
    """
    if warp:
        header = '#include "core/executor.h"'
        forward = _warp_kernels(hidden, False, fwd_wpb) + f"""
        float eps = static_cast<float>(data["eps"]);
        long long rows = in0->num / {hidden};
        layer_norm_forward_warp<<<(unsigned)((rows + {fwd_wpb - 1}) / {fwd_wpb}), {fwd_wpb * 32}>>>(
            in0_p, in1_p, in2_p, out0_p, out1_p, out2_p, rows, eps);
        CHECK(0 == cudaGetLastError());
        """
        backward = _warp_kernels(hidden, True, bwd_wpb) + f"""
        auto scratch = runtime_executor().alloc_temp(sizeof(float) * {2 * blocks * hidden});
        float* partial = static_cast<float*>(scratch.ptr);
        long long rows = in3->num / {hidden};
        if (sizeof(in3_type) == 4 && sizeof(in0_type) == 4 && sizeof(out0_type) == 4
                && ({hidden} % 4 == 0)
                && ((((size_t)in3_p | (size_t)in0_p | (size_t)out0_p) & 15u) == 0)) {{
            layer_norm_backward_fused_vec4<<<{blocks}, {bwd_wpb * 32}>>>(
                in3_p, in0_p, in5_p, in6_p, in1_p, out0_p, partial, rows);
        }} else {{
            layer_norm_backward_fused<<<{blocks}, {bwd_wpb * 32}>>>(
                in3_p, in0_p, in5_p, in6_p, in1_p, out0_p, partial, rows);
        }}
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
            out1_type* mean, out2_type* rstd, int rows, float eps) {{
        typedef cub::BlockReduce<float, {threads}> BlockReduce;
        __shared__ typename BlockReduce::TempStorage storage;
        __shared__ float mean_shared;
        __shared__ float rstd_shared;
        int row = blockIdx.x;
        if (row >= rows) return;
        long long base = (long long)row * {hidden};
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
                reduced / {hidden}.0f + eps);
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
    float eps = static_cast<float>(data["eps"]);
    int rows = in0->num / {hidden};
    layer_norm_forward<<<rows, {threads}>>>(
        in0_p, in1_p, in2_p, out0_p, out1_p, out2_p, rows, eps);
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
        long long base = (long long)row * {hidden};
        float row_mean = static_cast<float>(mean[row]);
        float row_rstd = static_cast<float>(rstd[row]);
        constexpr int kPer = ({hidden} + {threads} - 1) / {threads};
        constexpr bool kCache = kPer <= 8;
        float g_cache[kCache ? kPer : 1];
        float xhat_cache[kCache ? kPer : 1];
        float local_g = 0.0f;
        float local_gx = 0.0f;
        if (kCache) {{
            int i = 0;
            for (int j = threadIdx.x; j < {hidden}; j += blockDim.x, ++i) {{
                float xhat = (static_cast<float>(x[base + j]) - row_mean) * row_rstd;
                float g = static_cast<float>(grad_y[base + j]) * static_cast<float>(weight[j]);
                g_cache[i] = g;
                xhat_cache[i] = xhat;
                local_g += g;
                local_gx += g * xhat;
            }}
        }} else {{
            for (int j = threadIdx.x; j < {hidden}; j += blockDim.x) {{
                float xhat = (static_cast<float>(x[base + j]) - row_mean) * row_rstd;
                float g = static_cast<float>(grad_y[base + j]) * static_cast<float>(weight[j]);
                local_g += g;
                local_gx += g * xhat;
            }}
        }}
        float reduced_g = BlockReduce(storage).Sum(local_g);
        if (threadIdx.x == 0)
            mean_g_shared = reduced_g / {hidden}.0f;
        __syncthreads();
        float reduced_gx = BlockReduce(storage).Sum(local_gx);
        if (threadIdx.x == 0)
            mean_gx_shared = reduced_gx / {hidden}.0f;
        __syncthreads();

        float mean_g = mean_g_shared;
        float mean_gx = mean_gx_shared;
        if (kCache) {{
            int wi = 0;
            for (int j = threadIdx.x; j < {hidden}; j += blockDim.x, ++wi) {{
                grad_x[base + j] = out0_type(row_rstd * (
                    g_cache[wi] - mean_g - xhat_cache[wi] * mean_gx));
            }}
        }} else {{
            for (int j = threadIdx.x; j < {hidden}; j += blockDim.x) {{
                float xhat = (static_cast<float>(x[base + j]) - row_mean) * row_rstd;
                float g = static_cast<float>(grad_y[base + j]) * static_cast<float>(weight[j]);
                grad_x[base + j] = out0_type(row_rstd * (
                    g - mean_g - xhat * mean_gx));
            }}
        }}
    }}

    __global__ static void layer_norm_backward_affine(
            const in3_type* grad_y, const in0_type* x,
            const in5_type* mean, const in6_type* rstd,
            out1_type* grad_weight, out2_type* grad_bias,
            int rows) {{
        __shared__ float shared_weight[{tile_y}][{tile_x}];
        __shared__ float shared_bias[{tile_y}][{tile_x}];
        int channel = blockIdx.x * {tile_x} + threadIdx.x;
        float local_weight = 0.0f;
        float local_bias = 0.0f;
        if (channel < {hidden}) {{
            for (int row = threadIdx.y; row < rows;
                 row += {tile_y}) {{
                long long index = (long long)row * {hidden} + channel;
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
    fwd_wpb = _warp_warps_per_block(rows, hidden, False) if warp else 8
    bwd_wpb = _warp_warps_per_block(rows, hidden, True) if warp else 8
    blocks = min(_FUSED_BLOCKS, -(-rows // bwd_wpb)) if warp else 0
    header, forward, backward = _layer_norm_cuda_sources(
        hidden, warp, blocks, fwd_wpb, bwd_wpb)
    y, _, _ = jt.code(
        [shape, (rows,), (rows,)], [x.dtype, "float32", "float32"],
        [x, weight, bias], cuda_header=header, cuda_src=forward,
        cuda_grad_src=[backward], data={"multi_grad": 1, "eps": float(eps)})
    return y
