"""CUDA training fast path for standard RMS normalization."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

from functools import lru_cache
import math

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad
from jittor._runtime.dispatch import native_rule, optional_kernel

from .layer_norm_training_cuda import _warp_warps_per_block
from .rms_norm_cuda import _autocast_enabled


def _gamma_segments(hidden_size, rows):
    """Row segments of the gamma gradient: about a thousand blocks, none with
    fewer than eight rows a thread."""
    columns = -(-hidden_size // 32)
    return max(1, min(-(-1024 // columns), rows // 64))


#: As in layer_norm_training_cuda: rows at most this wide, when there are at
#: least this many, go a warp each, and the backward is one fused pass.
_WARP_ROW_LIMIT = 1024
_WARP_ROWS_MIN = 64
_FUSED_BLOCKS = 256


def _warp_kernels(hidden, eps, backward, wpb=8):
    """The forward, or the fused backward, with a warp per row; see
    `layer_norm_training_cuda._warp_kernels`. (No comments inside the source:
    `jt.code` lifts the kernels out by their text.)"""
    per = -(-hidden // 32)
    vec4_n = hidden // 4
    vec4_per = -(-vec4_n // 32) if vec4_n > 0 else 1
    vec8_n = hidden // 8
    vec8_per = -(-vec8_n // 32) if vec8_n > 0 else 1
    threads = wpb * 32
    allsum = """
    __device__ __forceinline__ float jt_rms_allsum(float v) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        return v;
    }
    """
    forward = f"""
    __global__ static void rms_norm_forward_warp(
            const in0_type* __restrict__ x, const in1_type* __restrict__ gamma,
            out0_type* __restrict__ y, out1_type* __restrict__ rstd,
            long long rows) {{
        long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int lane = threadIdx.x & 31;
        if (row >= rows) return;
        const in0_type* xr = x + row * {hidden}LL;
        out0_type* yr = y + row * {hidden}LL;
        if (sizeof(in0_type) == 4 && sizeof(out0_type) == 4 && ({hidden} % 4 == 0)
                && ((((size_t)x | (size_t)y) & 15u) == 0)) {{
            const float4* x4 = reinterpret_cast<const float4*>(xr);
            float4* y4 = reinterpret_cast<float4*>(yr);
            bool g_vec = (sizeof(in1_type) == 4) && ((((size_t)gamma) & 15u) == 0);
            const float4* g4 = g_vec ? reinterpret_cast<const float4*>(gamma) : nullptr;
            float4 cache[{vec4_per}];
            float sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < {vec4_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec4_n}) {{
                    float4 val = x4[v];
                    cache[i] = val;
                    sq += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
                }}
            }}
            float r = rsqrtf(jt_rms_allsum(sq) / {hidden}.0f + {eps:.9g}f);
            if (lane == 0) rstd[row] = out1_type(r);
            #pragma unroll
            for (int i = 0; i < {vec4_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec4_n}) {{
                    float4 val = cache[i];
                    float4 gv = g_vec ? g4[v] : make_float4(
                        static_cast<float>(gamma[v * 4]),
                        static_cast<float>(gamma[v * 4 + 1]),
                        static_cast<float>(gamma[v * 4 + 2]),
                        static_cast<float>(gamma[v * 4 + 3]));
                    float4 outv;
                    outv.x = (val.x * r) * gv.x;
                    outv.y = (val.y * r) * gv.y;
                    outv.z = (val.z * r) * gv.z;
                    outv.w = (val.w * r) * gv.w;
                    y4[v] = outv;
                }}
            }}
            return;
        }}
        if (sizeof(in0_type) == 2 && sizeof(out0_type) == 2 && ({hidden} % 8 == 0)
                && ((((size_t)x | (size_t)y) & 15u) == 0)) {{
            const uint4* x8 = reinterpret_cast<const uint4*>(xr);
            uint4* y8 = reinterpret_cast<uint4*>(yr);
            bool g_vec = (sizeof(in1_type) == 2) && ((((size_t)gamma) & 15u) == 0);
            const uint4* g8 = g_vec ? reinterpret_cast<const uint4*>(gamma) : nullptr;
            uint4 cache[{vec8_per}];
            float sq = 0.0f;
            #pragma unroll
            for (int i = 0; i < {vec8_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec8_n}) {{
                    uint4 raw = x8[v];
                    cache[i] = raw;
                    const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {{
                        float val = static_cast<float>(vp[k]);
                        sq += val * val;
                    }}
                }}
            }}
            float r = rsqrtf(jt_rms_allsum(sq) / {hidden}.0f + {eps:.9g}f);
            if (lane == 0) rstd[row] = out1_type(r);
            #pragma unroll
            for (int i = 0; i < {vec8_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec8_n}) {{
                    uint4 raw = cache[i];
                    const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                    uint4 outraw;
                    out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                    if (g_vec) {{
                        uint4 graw = g8[v];
                        const in1_type* gp = reinterpret_cast<const in1_type*>(&graw);
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {{
                            float norm = static_cast<float>(out0_type(static_cast<float>(vp[k]) * r));
                            op[k] = out0_type(norm * static_cast<float>(gp[k]));
                        }}
                    }} else {{
                        int d = v * 8;
                        #pragma unroll
                        for (int k = 0; k < 8; k++) {{
                            float norm = static_cast<float>(out0_type(static_cast<float>(vp[k]) * r));
                            op[k] = out0_type(norm * static_cast<float>(gamma[d + k]));
                        }}
                    }}
                    y8[v] = outraw;
                }}
            }}
            return;
        }}
        float cache[{per}];
        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            cache[i] = j < {hidden} ? static_cast<float>(xr[j]) : 0.0f;
            sq += cache[i] * cache[i];
        }}
        float r = rsqrtf(jt_rms_allsum(sq) / {hidden}.0f + {eps:.9g}f);
        if (lane == 0) rstd[row] = out1_type(r);
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            if (j < {hidden}) yr[j] = out0_type(static_cast<float>(out0_type(cache[i] * r))
                                                * static_cast<float>(gamma[j]));
        }}
    }}
    """
    backward_src = f"""
    __global__ static void rms_norm_backward_fused(
            const in0_type* __restrict__ grad_y, const in1_type* __restrict__ x,
            const in2_type* __restrict__ rstd, const in3_type* __restrict__ gamma,
            out0_type* __restrict__ grad_x, float* __restrict__ partial, long long rows) {{
        __shared__ __align__(16) float fold[{wpb}][{hidden}];
        __shared__ __align__(16) float sw[{hidden}];
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int j = threadIdx.x; j < {hidden}; j += {threads})
            sw[j] = static_cast<float>(gamma[j]);
        __syncthreads();
        if (sizeof(in0_type) == 4 && sizeof(in1_type) == 4 && sizeof(out0_type) == 4
                && ({hidden} % 4 == 0)
                && ((((size_t)grad_y | (size_t)x | (size_t)grad_x) & 15u) == 0)) {{
            float4 acc[{vec4_per}];
            #pragma unroll
            for (int i = 0; i < {vec4_per}; i++)
                acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4* sw4 = reinterpret_cast<const float4*>(sw);
            for (long long row = (long long)blockIdx.x * {wpb} + warp; row < rows;
                 row += (long long)gridDim.x * {wpb}) {{
                const float4* gr4 = reinterpret_cast<const float4*>(grad_y + row * {hidden}LL);
                const float4* xr4 = reinterpret_cast<const float4*>(x + row * {hidden}LL);
                float4* dr4 = reinterpret_cast<float4*>(grad_x + row * {hidden}LL);
                float r = static_cast<float>(rstd[row]);
                float4 gv[{vec4_per}], xhat[{vec4_per}];
                float sum_gx = 0.0f;
                #pragma unroll
                for (int i = 0; i < {vec4_per}; i++) {{
                    int v = lane + i * 32;
                    if (v < {vec4_n}) {{
                        float4 dy = gr4[v];
                        float4 xv = xr4[v];
                        float4 wv = sw4[v];
                        float4 xh = make_float4(xv.x * r, xv.y * r, xv.z * r, xv.w * r);
                        float4 g = make_float4(dy.x * wv.x, dy.y * wv.y, dy.z * wv.z, dy.w * wv.w);
                        gv[i] = g;
                        xhat[i] = xh;
                        sum_gx += g.x * xh.x + g.y * xh.y + g.z * xh.z + g.w * xh.w;
                        acc[i].x += dy.x * xh.x;
                        acc[i].y += dy.y * xh.y;
                        acc[i].z += dy.z * xh.z;
                        acc[i].w += dy.w * xh.w;
                    }}
                }}
                float mean_gx = jt_rms_allsum(sum_gx) / {hidden}.0f;
                #pragma unroll
                for (int i = 0; i < {vec4_per}; i++) {{
                    int v = lane + i * 32;
                    if (v < {vec4_n}) {{
                        float4 outv;
                        outv.x = r * (gv[i].x - xhat[i].x * mean_gx);
                        outv.y = r * (gv[i].y - xhat[i].y * mean_gx);
                        outv.z = r * (gv[i].z - xhat[i].z * mean_gx);
                        outv.w = r * (gv[i].w - xhat[i].w * mean_gx);
                        dr4[v] = outv;
                    }}
                }}
            }}
            float4* f4 = reinterpret_cast<float4*>(fold[warp]);
            #pragma unroll
            for (int i = 0; i < {vec4_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec4_n}) f4[v] = acc[i];
            }}
        }} else {{
            float acc[{per}];
            #pragma unroll
            for (int i = 0; i < {per}; i++) acc[i] = 0.0f;
            for (long long row = (long long)blockIdx.x * {wpb} + warp; row < rows;
                 row += (long long)gridDim.x * {wpb}) {{
                const in0_type* gr = grad_y + row * {hidden}LL;
                const in1_type* xr = x + row * {hidden}LL;
                float r = static_cast<float>(rstd[row]);
                float gv[{per}], xhat[{per}];
                float sum_gx = 0.0f;
                #pragma unroll
                for (int i = 0; i < {per}; i++) {{
                    int j = lane + i * 32;
                    bool in = j < {hidden};
                    float dy = in ? static_cast<float>(gr[j]) : 0.0f;
                    float xh = in ? static_cast<float>(xr[j]) * r : 0.0f;
                    float g = in ? dy * sw[j] : 0.0f;
                    gv[i] = g;
                    xhat[i] = xh;
                    sum_gx += g * xh;
                    acc[i] += dy * xh;
                }}
                float mean_gx = jt_rms_allsum(sum_gx) / {hidden}.0f;
                out0_type* dr = grad_x + row * {hidden}LL;
                #pragma unroll
                for (int i = 0; i < {per}; i++) {{
                    int j = lane + i * 32;
                    if (j < {hidden}) dr[j] = out0_type(r * (gv[i] - xhat[i] * mean_gx));
                }}
            }}
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                int j = lane + i * 32;
                if (j < {hidden}) fold[warp][j] = acc[i];
            }}
        }}
        __syncthreads();
        for (int j = threadIdx.x; j < {hidden}; j += {threads}) {{
            float t = 0.0f;
            #pragma unroll
            for (int k = 0; k < {wpb}; k++) t += fold[k][j];
            partial[(long long)blockIdx.x * {hidden} + j] = t;
        }}
    }}
    __global__ static void rms_norm_backward_finish(
            const float* __restrict__ partial, out1_type* __restrict__ grad_gamma, int parts) {{
        __shared__ float fold[8][33];
        int channel = blockIdx.x * 32 + threadIdx.x;
        float t = 0.0f;
        if (channel < {hidden})
            for (int p = threadIdx.y; p < parts; p += 8)
                t += partial[(long long)p * {hidden} + channel];
        fold[threadIdx.y][threadIdx.x] = t;
        __syncthreads();
        if (threadIdx.y || channel >= {hidden}) return;
        t = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) t += fold[k][threadIdx.x];
        grad_gamma[channel] = out1_type(t);
    }}
    """
    return allsum + (backward_src if backward else forward)


def _block_kernels(hidden, eps, threads, backward):
    """Block-per-row forward and grid-stride fused backward for wide rows
    (hidden > 1024) or tiny row counts (rows < 64)."""
    warps = threads // 32
    per = -(-hidden // threads)
    can_cache = per <= 16
    vec4_n = hidden // 4
    vec4_per = -(-vec4_n // threads) if vec4_n > 0 else 1
    vec8_n = hidden // 8
    vec8_per = -(-vec8_n // threads) if vec8_n > 0 else 1
    helpers = f"""
    __device__ __forceinline__ float jt_rms_blk_warp_sum(float v) {{
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        return v;
    }}
    """
    if not backward:
        return helpers + f"""
        __global__ static void rms_norm_forward_block(
                const in0_type* __restrict__ x, const in1_type* __restrict__ gamma,
                out0_type* __restrict__ y, out1_type* __restrict__ rstd, long long rows) {{
            __shared__ float warp_buf[{warps}];
            __shared__ float rstd_shared;
            long long row = blockIdx.x;
            if (row >= rows) return;
            int tid = threadIdx.x;
            int lane = tid & 31;
            int warp = tid >> 5;
            const in0_type* xr = x + row * {hidden}LL;
            out0_type* yr = y + row * {hidden}LL;
            if (sizeof(in0_type) == 4 && sizeof(out0_type) == 4 && ({hidden} % 4 == 0)
                    && ((((size_t)x | (size_t)y) & 15u) == 0)) {{
                constexpr bool kVecCache = {vec4_per} <= 4;
                const float4* x4 = reinterpret_cast<const float4*>(xr);
                float4* y4 = reinterpret_cast<float4*>(yr);
                bool g_vec = (sizeof(in1_type) == 4) && ((((size_t)gamma) & 15u) == 0);
                const float4* g4 = g_vec ? reinterpret_cast<const float4*>(gamma) : nullptr;
                float4 cache[kVecCache ? {vec4_per} : 1];
                float local = 0.0f;
                if (kVecCache) {{
                    #pragma unroll
                    for (int i = 0; i < {vec4_per}; ++i) {{
                        int v = tid + i * {threads};
                        if (v < {vec4_n}) {{
                            float4 val = x4[v];
                            cache[i] = val;
                            local += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
                        }}
                    }}
                }} else {{
                    for (int v = tid; v < {vec4_n}; v += {threads}) {{
                        float4 val = x4[v];
                        local += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
                    }}
                }}
                local = jt_rms_blk_warp_sum(local);
                if (lane == 0) warp_buf[warp] = local;
                __syncthreads();
                if (warp == 0) {{
                    float total = lane < {warps} ? warp_buf[lane] : 0.0f;
                    total = jt_rms_blk_warp_sum(total);
                    if (lane == 0) {{
                        rstd_shared = rsqrtf(total / {hidden}.0f + {eps:.9g}f);
                        rstd[row] = out1_type(rstd_shared);
                    }}
                }}
                __syncthreads();
                float r = rstd_shared;
                if (kVecCache) {{
                    #pragma unroll
                    for (int i = 0; i < {vec4_per}; ++i) {{
                        int v = tid + i * {threads};
                        if (v < {vec4_n}) {{
                            float4 val = cache[i];
                            float4 gv = g_vec ? g4[v] : make_float4(
                                static_cast<float>(gamma[v * 4]),
                                static_cast<float>(gamma[v * 4 + 1]),
                                static_cast<float>(gamma[v * 4 + 2]),
                                static_cast<float>(gamma[v * 4 + 3]));
                            float4 outv;
                            outv.x = (val.x * r) * gv.x;
                            outv.y = (val.y * r) * gv.y;
                            outv.z = (val.z * r) * gv.z;
                            outv.w = (val.w * r) * gv.w;
                            y4[v] = outv;
                        }}
                    }}
                }} else {{
                    for (int v = tid; v < {vec4_n}; v += {threads}) {{
                        float4 val = x4[v];
                        float4 gv = g_vec ? g4[v] : make_float4(
                            static_cast<float>(gamma[v * 4]),
                            static_cast<float>(gamma[v * 4 + 1]),
                            static_cast<float>(gamma[v * 4 + 2]),
                            static_cast<float>(gamma[v * 4 + 3]));
                        float4 outv;
                        outv.x = (val.x * r) * gv.x;
                        outv.y = (val.y * r) * gv.y;
                        outv.z = (val.z * r) * gv.z;
                        outv.w = (val.w * r) * gv.w;
                        y4[v] = outv;
                    }}
                }}
                return;
            }}
            if (sizeof(in0_type) == 2 && sizeof(out0_type) == 2 && ({hidden} % 8 == 0)
                    && ((((size_t)x | (size_t)y) & 15u) == 0)) {{
                constexpr bool kVecCache = {vec8_per} <= 2;
                const uint4* x8 = reinterpret_cast<const uint4*>(xr);
                uint4* y8 = reinterpret_cast<uint4*>(yr);
                bool g_vec = (sizeof(in1_type) == 2) && ((((size_t)gamma) & 15u) == 0);
                const uint4* g8 = g_vec ? reinterpret_cast<const uint4*>(gamma) : nullptr;
                uint4 cache[kVecCache ? {vec8_per} : 1];
                float local = 0.0f;
                if (kVecCache) {{
                    #pragma unroll
                    for (int i = 0; i < {vec8_per}; ++i) {{
                        int v = tid + i * {threads};
                        if (v < {vec8_n}) {{
                            uint4 raw = x8[v];
                            cache[i] = raw;
                            const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) {{
                                float val = static_cast<float>(vp[k]);
                                local += val * val;
                            }}
                        }}
                    }}
                }} else {{
                    for (int v = tid; v < {vec8_n}; v += {threads}) {{
                        uint4 raw = x8[v];
                        const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {{
                            float val = static_cast<float>(vp[k]);
                            local += val * val;
                        }}
                    }}
                }}
                local = jt_rms_blk_warp_sum(local);
                if (lane == 0) warp_buf[warp] = local;
                __syncthreads();
                if (warp == 0) {{
                    float total = lane < {warps} ? warp_buf[lane] : 0.0f;
                    total = jt_rms_blk_warp_sum(total);
                    if (lane == 0) {{
                        rstd_shared = rsqrtf(total / {hidden}.0f + {eps:.9g}f);
                        rstd[row] = out1_type(rstd_shared);
                    }}
                }}
                __syncthreads();
                float r = rstd_shared;
                if (kVecCache) {{
                    #pragma unroll
                    for (int i = 0; i < {vec8_per}; ++i) {{
                        int v = tid + i * {threads};
                        if (v < {vec8_n}) {{
                            uint4 raw = cache[i];
                            const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                            uint4 outraw;
                            out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                            if (g_vec) {{
                                uint4 graw = g8[v];
                                const in1_type* gp = reinterpret_cast<const in1_type*>(&graw);
                                #pragma unroll
                                for (int k = 0; k < 8; ++k) {{
                                    float norm = static_cast<float>(out0_type(static_cast<float>(vp[k]) * r));
                                    op[k] = out0_type(norm * static_cast<float>(gp[k]));
                                }}
                            }} else {{
                                int d = v * 8;
                                #pragma unroll
                                for (int k = 0; k < 8; ++k) {{
                                    float norm = static_cast<float>(out0_type(static_cast<float>(vp[k]) * r));
                                    op[k] = out0_type(norm * static_cast<float>(gamma[d + k]));
                                }}
                            }}
                            y8[v] = outraw;
                        }}
                    }}
                }} else {{
                    for (int v = tid; v < {vec8_n}; v += {threads}) {{
                        uint4 raw = x8[v];
                        const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                        uint4 outraw;
                        out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                        int d = v * 8;
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {{
                            float norm = static_cast<float>(out0_type(static_cast<float>(vp[k]) * r));
                            op[k] = out0_type(norm * static_cast<float>(gamma[d + k]));
                        }}
                        y8[v] = outraw;
                    }}
                }}
                return;
            }}
            constexpr bool kCache = {int(can_cache)} != 0;
            float cache[kCache ? {per} : 1];
            float local = 0.0f;
            if (kCache) {{
                #pragma unroll
                for (int i = 0; i < {per}; ++i) {{
                    int j = tid + i * {threads};
                    if (j < {hidden}) {{
                        float val = static_cast<float>(xr[j]);
                        cache[i] = val;
                        local += val * val;
                    }}
                }}
            }} else {{
                for (int j = tid; j < {hidden}; j += {threads}) {{
                    float val = static_cast<float>(xr[j]);
                    local += val * val;
                }}
            }}
            local = jt_rms_blk_warp_sum(local);
            if (lane == 0) warp_buf[warp] = local;
            __syncthreads();
            if (warp == 0) {{
                float total = lane < {warps} ? warp_buf[lane] : 0.0f;
                total = jt_rms_blk_warp_sum(total);
                if (lane == 0) {{
                    rstd_shared = rsqrtf(total / {hidden}.0f + {eps:.9g}f);
                    rstd[row] = out1_type(rstd_shared);
                }}
            }}
            __syncthreads();
            float r = rstd_shared;
            if (kCache) {{
                #pragma unroll
                for (int i = 0; i < {per}; ++i) {{
                    int j = tid + i * {threads};
                    if (j < {hidden})
                        yr[j] = out0_type(static_cast<float>(out0_type(cache[i] * r))
                                          * static_cast<float>(gamma[j]));
                }}
            }} else {{
                for (int j = tid; j < {hidden}; j += {threads})
                    yr[j] = out0_type(static_cast<float>(out0_type(static_cast<float>(xr[j]) * r))
                                      * static_cast<float>(gamma[j]));
            }}
        }}
        """

    # Grid-stride fused block backward: each thread owns fixed columns, keeping
    # its gamma partial in registers (zero shared-memory fold!) and reading x,
    # grad_y once per row.
    return helpers + f"""
    __global__ static void rms_norm_backward_block_fused(
            const in0_type* __restrict__ grad_y, const in1_type* __restrict__ x,
            const in2_type* __restrict__ rstd, const in3_type* __restrict__ gamma,
            out0_type* __restrict__ grad_x, float* __restrict__ partial, long long rows) {{
        __shared__ float warp_buf[{warps}];
        __shared__ float mean_gx_shared;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        if (sizeof(in0_type) == 4 && sizeof(in1_type) == 4 && sizeof(out0_type) == 4
                && ({hidden} % 4 == 0) && ({vec4_per} <= 4)
                && ((((size_t)grad_y | (size_t)x | (size_t)grad_x | (size_t)partial) & 15u) == 0)) {{
            bool g_vec = (sizeof(in3_type) == 4) && ((((size_t)gamma) & 15u) == 0);
            const float4* g4 = g_vec ? reinterpret_cast<const float4*>(gamma) : nullptr;
            float4 wv[{vec4_per}], acc[{vec4_per}];
            #pragma unroll
            for (int i = 0; i < {vec4_per}; ++i) {{
                int v = tid + i * {threads};
                acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (v < {vec4_n}) {{
                    wv[i] = g_vec ? g4[v] : make_float4(
                        static_cast<float>(gamma[v * 4]),
                        static_cast<float>(gamma[v * 4 + 1]),
                        static_cast<float>(gamma[v * 4 + 2]),
                        static_cast<float>(gamma[v * 4 + 3]));
                }}
            }}
            for (long long row = blockIdx.x; row < rows; row += gridDim.x) {{
                const float4* gr4 = reinterpret_cast<const float4*>(grad_y + row * {hidden}LL);
                const float4* xr4 = reinterpret_cast<const float4*>(x + row * {hidden}LL);
                float4* dr4 = reinterpret_cast<float4*>(grad_x + row * {hidden}LL);
                float r = static_cast<float>(rstd[row]);
                float4 gv[{vec4_per}], xhat[{vec4_per}];
                float local = 0.0f;
                #pragma unroll
                for (int i = 0; i < {vec4_per}; ++i) {{
                    int v = tid + i * {threads};
                    if (v < {vec4_n}) {{
                        float4 dy = gr4[v];
                        float4 xv = xr4[v];
                        float4 xh = make_float4(xv.x * r, xv.y * r, xv.z * r, xv.w * r);
                        float4 g = make_float4(dy.x * wv[i].x, dy.y * wv[i].y, dy.z * wv[i].z, dy.w * wv[i].w);
                        gv[i] = g;
                        xhat[i] = xh;
                        local += g.x * xh.x + g.y * xh.y + g.z * xh.z + g.w * xh.w;
                        acc[i].x += dy.x * xh.x;
                        acc[i].y += dy.y * xh.y;
                        acc[i].z += dy.z * xh.z;
                        acc[i].w += dy.w * xh.w;
                    }}
                }}
                local = jt_rms_blk_warp_sum(local);
                if (lane == 0) warp_buf[warp] = local;
                __syncthreads();
                if (warp == 0) {{
                    float total = lane < {warps} ? warp_buf[lane] : 0.0f;
                    total = jt_rms_blk_warp_sum(total);
                    if (lane == 0) mean_gx_shared = total / {hidden}.0f;
                }}
                __syncthreads();
                float mean_gx = mean_gx_shared;
                #pragma unroll
                for (int i = 0; i < {vec4_per}; ++i) {{
                    int v = tid + i * {threads};
                    if (v < {vec4_n}) {{
                        float4 outv;
                        outv.x = r * (gv[i].x - xhat[i].x * mean_gx);
                        outv.y = r * (gv[i].y - xhat[i].y * mean_gx);
                        outv.z = r * (gv[i].z - xhat[i].z * mean_gx);
                        outv.w = r * (gv[i].w - xhat[i].w * mean_gx);
                        dr4[v] = outv;
                    }}
                }}
            }}
            float4* p4 = reinterpret_cast<float4*>(partial + (long long)blockIdx.x * {hidden});
            #pragma unroll
            for (int i = 0; i < {vec4_per}; ++i) {{
                int v = tid + i * {threads};
                if (v < {vec4_n}) p4[v] = acc[i];
            }}
            return;
        }}
        if (sizeof(in0_type) == 2 && sizeof(in1_type) == 2 && sizeof(out0_type) == 2
                && ({hidden} % 8 == 0) && ({vec8_per} <= 2)
                && ((((size_t)grad_y | (size_t)x | (size_t)grad_x) & 15u) == 0)) {{
            float wv[{vec8_per}][8], acc[{vec8_per}][8];
            #pragma unroll
            for (int i = 0; i < {vec8_per}; ++i) {{
                int v = tid + i * {threads};
                int d = v * 8;
                #pragma unroll
                for (int k = 0; k < 8; ++k) {{
                    acc[i][k] = 0.0f;
                    wv[i][k] = v < {vec8_n} ? static_cast<float>(gamma[d + k]) : 0.0f;
                }}
            }}
            for (long long row = blockIdx.x; row < rows; row += gridDim.x) {{
                const uint4* gr8 = reinterpret_cast<const uint4*>(grad_y + row * {hidden}LL);
                const uint4* xr8 = reinterpret_cast<const uint4*>(x + row * {hidden}LL);
                uint4* dr8 = reinterpret_cast<uint4*>(grad_x + row * {hidden}LL);
                float r = static_cast<float>(rstd[row]);
                float gv[{vec8_per}][8], xhat[{vec8_per}][8];
                float local = 0.0f;
                #pragma unroll
                for (int i = 0; i < {vec8_per}; ++i) {{
                    int v = tid + i * {threads};
                    if (v < {vec8_n}) {{
                        uint4 draw = gr8[v];
                        uint4 xraw = xr8[v];
                        const in0_type* dp = reinterpret_cast<const in0_type*>(&draw);
                        const in1_type* xp = reinterpret_cast<const in1_type*>(&xraw);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {{
                            float dy = static_cast<float>(dp[k]);
                            float xh = static_cast<float>(xp[k]) * r;
                            float g = dy * wv[i][k];
                            gv[i][k] = g;
                            xhat[i][k] = xh;
                            local += g * xh;
                            acc[i][k] += dy * xh;
                        }}
                    }}
                }}
                local = jt_rms_blk_warp_sum(local);
                if (lane == 0) warp_buf[warp] = local;
                __syncthreads();
                if (warp == 0) {{
                    float total = lane < {warps} ? warp_buf[lane] : 0.0f;
                    total = jt_rms_blk_warp_sum(total);
                    if (lane == 0) mean_gx_shared = total / {hidden}.0f;
                }}
                __syncthreads();
                float mean_gx = mean_gx_shared;
                #pragma unroll
                for (int i = 0; i < {vec8_per}; ++i) {{
                    int v = tid + i * {threads};
                    if (v < {vec8_n}) {{
                        uint4 outraw;
                        out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k)
                            op[k] = out0_type(r * (gv[i][k] - xhat[i][k] * mean_gx));
                        dr8[v] = outraw;
                    }}
                }}
            }}
            float* pr = partial + (long long)blockIdx.x * {hidden};
            #pragma unroll
            for (int i = 0; i < {vec8_per}; ++i) {{
                int v = tid + i * {threads};
                if (v < {vec8_n}) {{
                    int d = v * 8;
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) pr[d + k] = acc[i][k];
                }}
            }}
            return;
        }}
        float wv[{per}], acc[{per}];
        #pragma unroll
        for (int i = 0; i < {per}; ++i) {{
            int j = tid + i * {threads};
            acc[i] = 0.0f;
            wv[i] = j < {hidden} ? static_cast<float>(gamma[j]) : 0.0f;
        }}
        for (long long row = blockIdx.x; row < rows; row += gridDim.x) {{
            const in0_type* gr = grad_y + row * {hidden}LL;
            const in1_type* xr = x + row * {hidden}LL;
            out0_type* dr = grad_x + row * {hidden}LL;
            float r = static_cast<float>(rstd[row]);
            float gv[{per}], xhat[{per}];
            float local = 0.0f;
            #pragma unroll
            for (int i = 0; i < {per}; ++i) {{
                int j = tid + i * {threads};
                if (j < {hidden}) {{
                    float dy = static_cast<float>(gr[j]);
                    float xh = static_cast<float>(xr[j]) * r;
                    float g = dy * wv[i];
                    gv[i] = g;
                    xhat[i] = xh;
                    local += g * xh;
                    acc[i] += dy * xh;
                }}
            }}
            local = jt_rms_blk_warp_sum(local);
            if (lane == 0) warp_buf[warp] = local;
            __syncthreads();
            if (warp == 0) {{
                float total = lane < {warps} ? warp_buf[lane] : 0.0f;
                total = jt_rms_blk_warp_sum(total);
                if (lane == 0) mean_gx_shared = total / {hidden}.0f;
            }}
            __syncthreads();
            float mean_gx = mean_gx_shared;
            #pragma unroll
            for (int i = 0; i < {per}; ++i) {{
                int j = tid + i * {threads};
                if (j < {hidden}) dr[j] = out0_type(r * (gv[i] - xhat[i] * mean_gx));
            }}
        }}
        float* pr = partial + (long long)blockIdx.x * {hidden};
        #pragma unroll
        for (int i = 0; i < {per}; ++i) {{
            int j = tid + i * {threads};
            if (j < {hidden}) pr[j] = acc[i];
        }}
    }}
    __global__ static void rms_norm_backward_finish(
            const float* __restrict__ partial, out1_type* __restrict__ grad_gamma, int parts) {{
        __shared__ float fold[8][33];
        int channel = blockIdx.x * 32 + threadIdx.x;
        float t = 0.0f;
        if (channel < {hidden})
            for (int p = threadIdx.y; p < parts; p += 8)
                t += partial[(long long)p * {hidden} + channel];
        fold[threadIdx.y][threadIdx.x] = t;
        __syncthreads();
        if (threadIdx.y || channel >= {hidden}) return;
        t = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) t += fold[k][threadIdx.x];
        grad_gamma[channel] = out1_type(t);
    }}
    """


@lru_cache(maxsize=128)
def _rms_norm_training_cuda_cls(hidden_size, epsilon):
    threads = 32
    max_threads = 512 if hidden_size > 2048 else 256
    while threads < min(hidden_size, max_threads):
        threads *= 2
    per = -(-hidden_size // threads)
    can_block_fuse = per <= 16

    class RMSNormTrainingCUDA(jt.Function):
        def execute(self, x, gamma):
            rows = int(x.numel()) // hidden_size
            self.warp = hidden_size <= _WARP_ROW_LIMIT and rows >= _WARP_ROWS_MIN
            if self.warp:
                wpb = _warp_warps_per_block(rows, hidden_size, False)
                y, rstd = jt.code(
                    [x.shape, (rows,)], [x.dtype, "float32"], [x, gamma],
                    cuda_src=_warp_kernels(hidden_size, epsilon, False, wpb) + f"""
                    long long rows = in0->num / {hidden_size};
                    rms_norm_forward_warp<<<(unsigned)((rows + {wpb - 1}) / {wpb}), {wpb * 32}>>>(
                        in0_p, in1_p, out0_p, out1_p, rows);
                    CHECK(0 == cudaGetLastError());
                    """)
                self.saved = x, rstd, gamma
                return y
            y, rstd = jt.code(
                [x.shape, (rows,)],
                [x.dtype, "float32"],
                [x, gamma],
                cuda_src=_block_kernels(hidden_size, epsilon, threads, False) + f"""
                long long rows = in0->num / {hidden_size};
                rms_norm_forward_block<<<(unsigned)rows, {threads}>>>(
                    in0_p, in1_p, out0_p, out1_p, rows);
                CHECK(0 == cudaGetLastError());
                """,
            )
            self.saved = x, rstd, gamma
            return y

        def grad(self, grad_y):
            x, rstd, gamma = self.saved
            rows = int(x.numel()) // hidden_size
            if self.warp:
                wpb = _warp_warps_per_block(rows, hidden_size, True)
                blocks = min(_FUSED_BLOCKS, -(-rows // wpb))
                grad_x, grad_gamma, _ = jt.code(
                    [x.shape, gamma.shape, (blocks * hidden_size,)],
                    [x.dtype, gamma.dtype, "float32"],
                    [grad_y, x, rstd, gamma],
                    cuda_src=_warp_kernels(hidden_size, epsilon, True, wpb) + f"""
                    long long rows = in1->num / {hidden_size};
                    rms_norm_backward_fused<<<{blocks}, {wpb * 32}>>>(
                        in0_p, in1_p, in2_p, in3_p, out0_p, out2_p, rows);
                    rms_norm_backward_finish<<<{-(-hidden_size // 32)}, dim3(32, 8)>>>(
                        out2_p, out1_p, {blocks});
                    CHECK(0 == cudaGetLastError());
                    """)
                return grad_x, grad_gamma
            if can_block_fuse:
                blocks = min(_FUSED_BLOCKS, rows)
                grad_x, grad_gamma, _ = jt.code(
                    [x.shape, gamma.shape, (blocks * hidden_size,)],
                    [x.dtype, gamma.dtype, "float32"],
                    [grad_y, x, rstd, gamma],
                    cuda_src=_block_kernels(hidden_size, epsilon, threads, True) + f"""
                    long long rows = in1->num / {hidden_size};
                    rms_norm_backward_block_fused<<<{blocks}, {threads}>>>(
                        in0_p, in1_p, in2_p, in3_p, out0_p, out2_p, rows);
                    rms_norm_backward_finish<<<{-(-hidden_size // 32)}, dim3(32, 8)>>>(
                        out2_p, out1_p, {blocks});
                    CHECK(0 == cudaGetLastError());
                    """)
                return grad_x, grad_gamma
            segments = _gamma_segments(hidden_size, rows)
            per_segment = -(-rows // segments)
            warps = threads // 32
            grad_x, grad_gamma, _ = jt.code(
                [x.shape, gamma.shape, (segments * hidden_size,)],
                [x.dtype, gamma.dtype, "float32"],
                [grad_y, x, rstd, gamma],
                cuda_src=f"""
                __device__ __forceinline__ float jt_rms_b2_warp_sum(float v) {{
                    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
                    return v;
                }}
                __global__ static void rms_norm_backward_x(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* rstd, const in3_type* gamma,
                        out0_type* grad_x, long long rows) {{
                    __shared__ float warp_buf[{warps}];
                    __shared__ float mean_gx_shared;
                    long long row = blockIdx.x;
                    if (row >= rows) return;
                    int tid = threadIdx.x;
                    int lane = tid & 31;
                    int warp = tid >> 5;
                    long long base = row * {hidden_size}LL;
                    float row_rstd = static_cast<float>(rstd[row]);
                    float local = 0.0f;
                    for (int j = tid; j < {hidden_size}; j += blockDim.x) {{
                        long long index = base + j;
                        float normalized = static_cast<float>(x[index]) * row_rstd;
                        local += static_cast<float>(grad_y[index])
                            * static_cast<float>(gamma[j]) * normalized;
                    }}
                    local = jt_rms_b2_warp_sum(local);
                    if (lane == 0) warp_buf[warp] = local;
                    __syncthreads();
                    if (warp == 0) {{
                        float total = lane < {warps} ? warp_buf[lane] : 0.0f;
                        total = jt_rms_b2_warp_sum(total);
                        if (lane == 0) mean_gx_shared = total / {hidden_size}.0f;
                    }}
                    __syncthreads();
                    float mean_gx = mean_gx_shared;
                    for (int j = tid; j < {hidden_size}; j += blockDim.x) {{
                        long long index = base + j;
                        float normalized = static_cast<float>(x[index]) * row_rstd;
                        float g = static_cast<float>(grad_y[index])
                            * static_cast<float>(gamma[j]);
                        grad_x[index] = out0_type(
                            row_rstd * (g - normalized * mean_gx));
                    }}
                }}

                __global__ static void rms_norm_backward_gamma_partial(
                        const in0_type* grad_y, const in1_type* x,
                        const in2_type* rstd, float* partial) {{
                    __shared__ float sums[8][33];
                    int channel = blockIdx.x * 32 + threadIdx.x;
                    int begin = blockIdx.y * {per_segment};
                    int end = min(begin + {per_segment}, {rows});
                    float local = 0.0f;
                    if (channel < {hidden_size})
                        for (int row = begin + threadIdx.y; row < end; row += 8) {{
                            long long index = (long long)row * {hidden_size} + channel;
                            local += static_cast<float>(grad_y[index])
                                * static_cast<float>(x[index])
                                * static_cast<float>(rstd[row]);
                        }}
                    sums[threadIdx.y][threadIdx.x] = local;
                    __syncthreads();
                    if (threadIdx.y == 0 && channel < {hidden_size}) {{
                        float total = 0.0f;
                        for (int k = 0; k < 8; k++) total += sums[k][threadIdx.x];
                        partial[(long long)blockIdx.y * {hidden_size} + channel] = total;
                    }}
                }}

                __global__ static void rms_norm_backward_gamma_finish(
                        const float* partial, out1_type* grad_gamma) {{
                    int channel = blockIdx.x * blockDim.x + threadIdx.x;
                    if (channel >= {hidden_size}) return;
                    float total = 0.0f;
                    for (int s = 0; s < {segments}; s++)
                        total += partial[(long long)s * {hidden_size} + channel];
                    grad_gamma[channel] = out1_type(total);
                }}

                long long rows = in1->num / {hidden_size};
                rms_norm_backward_x<<<(unsigned)rows, {threads}>>>(
                    in0_p, in1_p, in2_p, in3_p, out0_p, rows);
                rms_norm_backward_gamma_partial<<<dim3({-(-hidden_size // 32)}, {segments}),
                                                  dim3(32, 8)>>>(
                    in0_p, in1_p, in2_p, out2_p);
                rms_norm_backward_gamma_finish<<<{-(-hidden_size // 256)}, 256>>>(
                    out2_p, out1_p);
                CHECK(0 == cudaGetLastError());
                """,
            )
            return grad_x, grad_gamma

    return RMSNormTrainingCUDA


@native_rule("rms_norm_training")
def _supports_rms_norm_training(x, gamma, epsilon=1e-6):
    if not (
        isinstance(x, jt.Var)
        and isinstance(gamma, jt.Var)
        and _output_requires_grad(x, gamma)
        and not _autocast_enabled()
    ):
        return False
    if _jittor_dtype_name(x.dtype) != _jittor_dtype_name(gamma.dtype):
        return False
    try:
        x_shape = tuple(int(size) for size in x.shape)
        gamma_shape = tuple(int(size) for size in gamma.shape)
        epsilon_value = float(epsilon)
    except (TypeError, ValueError, OverflowError):
        return False
    if not x_shape or any(size <= 0 for size in x_shape):
        return False
    hidden_size = x_shape[-1]
    if (
        hidden_size > 16384
        or gamma_shape != (hidden_size,)
        or not math.isfinite(epsilon_value)
        or epsilon_value <= 0.0
    ):
        return False
    return True


@optional_kernel("nn.rms_norm.training", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("float32", "float16", "bfloat16"),
                 supports=_supports_rms_norm_training)
def _rms_norm_training_cuda(x, gamma, epsilon=1e-6):
    cls = _rms_norm_training_cuda_cls(int(x.shape[-1]), float(epsilon))
    return cls.apply(x, gamma)
