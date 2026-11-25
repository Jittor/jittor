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


_FUSABLE_PRODUCERS = ("binary.", "unary.", "ternary", "broadcast_to")


def _can_fuse_residual_grad_externally(grad_total):
    """True when `grad_total` is a pending fusable elementwise op so adding it
    outside the CUDA kernel lets Jittor fuse its producer with `+ grad_rms`
    instead of materializing an extra full-tensor temporary."""
    if grad_total.is_finished or grad_total.is_stop_fuse():
        return False
    return grad_total._producer_op().startswith(_FUSABLE_PRODUCERS)


_ADD_LDCS_HELPERS = """
    __device__ __forceinline__ float4 jt_add_rms_ldcs_f4(const float4* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        return __ldcs(p);
#else
        return *p;
#endif
    }
    __device__ __forceinline__ uint4 jt_add_rms_ldcs_u4(const uint4* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        return __ldcs(p);
#else
        return *p;
#endif
    }
    __device__ __forceinline__ float4 jt_add_rms_ldg_f4(const float4* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        return __ldg(p);
#else
        return *p;
#endif
    }
    __device__ __forceinline__ uint4 jt_add_rms_ldg_u4(const uint4* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        return __ldg(p);
#else
        return *p;
#endif
    }
"""


@lru_cache(maxsize=256)
def _add_warp_fwd_src(hidden, eps, dtype, wpb):
    """Cached warp-per-row forward source for fused add+RMSNorm with
    host-dispatched vectorized vs scalar kernels."""
    per = -(-hidden // 32)
    threads = wpb * 32
    is_f32 = dtype == "float32"
    vec_w = 4 if is_f32 else 8
    has_vec = (hidden % vec_w) == 0
    vec_n = hidden // vec_w if has_vec else 1
    vec_per = -(-vec_n // 32) if has_vec else 1

    allsum = _ADD_LDCS_HELPERS + """
    __device__ __forceinline__ float jt_add_rms_allsum(float v) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        return v;
    }
    """
    vec_kernel = ""
    if has_vec and is_f32:
        vec_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_forward_warp_vec(
            const float4* __restrict__ x, const float4* __restrict__ residual,
            const float4* __restrict__ gamma, float4* __restrict__ y,
            float4* __restrict__ total_out, out2_type* __restrict__ rstd,
            long long rows) {{
        long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int lane = threadIdx.x & 31;
        if (row >= rows) return;
        long long off = row * {vec_n}LL;
        const float4* x4 = x + off;
        const float4* r4 = residual + off;
        float4* y4 = y + off;
        float4* t4 = total_out + off;
        float4 cache[{vec_per}];
        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < {vec_per}; i++) {{
            int v = lane + i * 32;
            if (v < {vec_n}) {{
                float4 xv = jt_add_rms_ldcs_f4(x4 + v);
                float4 rv = jt_add_rms_ldcs_f4(r4 + v);
                float4 tv = make_float4(xv.x + rv.x, xv.y + rv.y, xv.z + rv.z, xv.w + rv.w);
                t4[v] = tv;
                cache[i] = tv;
                sq += tv.x * tv.x + tv.y * tv.y + tv.z * tv.z + tv.w * tv.w;
            }}
        }}
        float r = rsqrtf(jt_add_rms_allsum(sq) / {hidden}.0f + {eps:.9g}f);
        if (lane == 0) rstd[row] = out2_type(r);
        #pragma unroll
        for (int i = 0; i < {vec_per}; i++) {{
            int v = lane + i * 32;
            if (v < {vec_n}) {{
                float4 tv = cache[i];
                float4 gv = jt_add_rms_ldg_f4(gamma + v);
                y4[v] = make_float4((tv.x * r) * gv.x, (tv.y * r) * gv.y,
                                    (tv.z * r) * gv.z, (tv.w * r) * gv.w);
            }}
        }}
    }}
        """
    elif has_vec:
        vec_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_forward_warp_vec(
            const uint4* __restrict__ x, const uint4* __restrict__ residual,
            const uint4* __restrict__ gamma, uint4* __restrict__ y,
            uint4* __restrict__ total_out, out2_type* __restrict__ rstd,
            long long rows) {{
        long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int lane = threadIdx.x & 31;
        if (row >= rows) return;
        long long off = row * {vec_n}LL;
        const uint4* x8 = x + off;
        const uint4* r8 = residual + off;
        uint4* y8 = y + off;
        uint4* t8 = total_out + off;
        uint4 cache[{vec_per}];
        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < {vec_per}; i++) {{
            int v = lane + i * 32;
            if (v < {vec_n}) {{
                uint4 xraw = jt_add_rms_ldcs_u4(x8 + v);
                uint4 rraw = jt_add_rms_ldcs_u4(r8 + v);
                uint4 traw;
                const in0_type* xp = reinterpret_cast<const in0_type*>(&xraw);
                const in1_type* rp = reinterpret_cast<const in1_type*>(&rraw);
                out1_type* tp = reinterpret_cast<out1_type*>(&traw);
                #pragma unroll
                for (int k = 0; k < 8; k++) {{
                    out1_type tv = out1_type(static_cast<float>(xp[k]) + static_cast<float>(rp[k]));
                    tp[k] = tv;
                    float val = static_cast<float>(tv);
                    sq += val * val;
                }}
                t8[v] = traw;
                cache[i] = traw;
            }}
        }}
        float r = rsqrtf(jt_add_rms_allsum(sq) / {hidden}.0f + {eps:.9g}f);
        if (lane == 0) rstd[row] = out2_type(r);
        #pragma unroll
        for (int i = 0; i < {vec_per}; i++) {{
            int v = lane + i * 32;
            if (v < {vec_n}) {{
                uint4 traw = cache[i];
                uint4 graw = jt_add_rms_ldg_u4(gamma + v);
                uint4 outraw;
                const out1_type* tp = reinterpret_cast<const out1_type*>(&traw);
                const in2_type* gp = reinterpret_cast<const in2_type*>(&graw);
                out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                #pragma unroll
                for (int k = 0; k < 8; k++) {{
                    float norm = static_cast<float>(out0_type(static_cast<float>(tp[k]) * r));
                    op[k] = out0_type(norm * static_cast<float>(gp[k]));
                }}
                y8[v] = outraw;
            }}
        }}
    }}
        """

    scalar_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_forward_warp_scalar(
            const in0_type* __restrict__ x, const in1_type* __restrict__ residual,
            const in2_type* __restrict__ gamma, out0_type* __restrict__ y,
            out1_type* __restrict__ total_out, out2_type* __restrict__ rstd,
            long long rows) {{
        long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int lane = threadIdx.x & 31;
        if (row >= rows) return;
        long long row_off = row * {hidden}LL;
        const in0_type* xr = x + row_off;
        const in1_type* rr = residual + row_off;
        out0_type* yr = y + row_off;
        out1_type* tr = total_out + row_off;
        float cache[{per}];
        float sq = 0.0f;
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            float val = 0.0f;
            if (j < {hidden}) {{
                out1_type tv = out1_type(static_cast<float>(xr[j]) + static_cast<float>(rr[j]));
                tr[j] = tv;
                val = static_cast<float>(tv);
            }}
            cache[i] = val;
            sq += val * val;
        }}
        float r = rsqrtf(jt_add_rms_allsum(sq) / {hidden}.0f + {eps:.9g}f);
        if (lane == 0) rstd[row] = out2_type(r);
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            if (j < {hidden}) yr[j] = out0_type(static_cast<float>(out0_type(cache[i] * r))
                                                * static_cast<float>(gamma[j]));
        }}
    }}
    """
    vtype = "float4" if is_f32 else "uint4"
    if has_vec:
        launch = f"""
        long long rows = in0->num / {hidden};
        unsigned grid = (unsigned)((rows + {wpb - 1}) / {wpb});
        if ((((size_t)in0_p | (size_t)in1_p | (size_t)in2_p | (size_t)out0_p | (size_t)out1_p) & 15u) == 0) {{
            add_rms_norm_forward_warp_vec<<<grid, {threads}>>>(
                reinterpret_cast<const {vtype}*>(in0_p), reinterpret_cast<const {vtype}*>(in1_p),
                reinterpret_cast<const {vtype}*>(in2_p), reinterpret_cast<{vtype}*>(out0_p),
                reinterpret_cast<{vtype}*>(out1_p), out2_p, rows);
        }} else {{
            add_rms_norm_forward_warp_scalar<<<grid, {threads}>>>(
                in0_p, in1_p, in2_p, out0_p, out1_p, out2_p, rows);
        }}
        CHECK(0 == cudaGetLastError());
        """
    else:
        launch = f"""
        long long rows = in0->num / {hidden};
        unsigned grid = (unsigned)((rows + {wpb - 1}) / {wpb});
        add_rms_norm_forward_warp_scalar<<<grid, {threads}>>>(
            in0_p, in1_p, in2_p, out0_p, out1_p, out2_p, rows);
        CHECK(0 == cudaGetLastError());
        """
    return allsum + vec_kernel + scalar_kernel + launch


@lru_cache(maxsize=256)
def _add_warp_bwd_src(hidden, dtype, wpb, with_grad_total):
    """Cached warp-per-row backward source for fused add+RMSNorm."""
    per = -(-hidden // 32)
    threads = wpb * 32
    is_f32 = dtype == "float32"
    vec_w = 4 if is_f32 else 8
    has_vec = (hidden % vec_w) == 0
    vec_n = hidden // vec_w if has_vec else 1
    vec_per = -(-vec_n // 32) if has_vec else 1
    allsum = _ADD_LDCS_HELPERS + """
    __device__ __forceinline__ float jt_add_rms_allsum(float v) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        return v;
    }
    """
    gt_param_f4 = "const float4* __restrict__ grad_total," if with_grad_total else ""
    gt_param_u4 = "const uint4* __restrict__ grad_total," if with_grad_total else ""
    gt_param_sc = "const in1_type* __restrict__ grad_total," if with_grad_total else ""
    tot_in_t = "in2_type" if with_grad_total else "in1_type"
    rstd_in_t = "in3_type" if with_grad_total else "in2_type"
    gam_in_t = "in4_type" if with_grad_total else "in3_type"

    vec_kernel = ""
    if has_vec and is_f32:
        gt_ptr_decl = "const float4* gt4 = grad_total + off;" if with_grad_total else ""
        gt_load = "float4 dt = jt_add_rms_ldcs_f4(gt4 + v);" if with_grad_total else ""
        gt_add_x = " + dt.x" if with_grad_total else ""
        gt_add_y = " + dt.y" if with_grad_total else ""
        gt_add_z = " + dt.z" if with_grad_total else ""
        gt_add_w = " + dt.w" if with_grad_total else ""
        vec_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_backward_warp_vec(
            const float4* __restrict__ grad_y, {gt_param_f4}
            const float4* __restrict__ total, const {rstd_in_t}* __restrict__ rstd,
            const float4* __restrict__ gamma,
            float4* __restrict__ grad_x, float* __restrict__ partial, long long rows) {{
        __shared__ __align__(16) float fold[{wpb}][{hidden}];
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        float4 wv[{vec_per}], acc[{vec_per}];
        #pragma unroll
        for (int i = 0; i < {vec_per}; i++) {{
            int v = lane + i * 32;
            acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (v < {vec_n}) wv[i] = jt_add_rms_ldg_f4(gamma + v);
        }}
        for (long long row = (long long)blockIdx.x * {wpb} + warp; row < rows;
             row += (long long)gridDim.x * {wpb}) {{
            long long off = row * {vec_n}LL;
            const float4* gr4 = grad_y + off;
            {gt_ptr_decl}
            const float4* tr4 = total + off;
            float4* dr4 = grad_x + off;
            float r = static_cast<float>(rstd[row]);
            float4 gv[{vec_per}], xhat[{vec_per}];
            float sum_gx = 0.0f;
            #pragma unroll
            for (int i = 0; i < {vec_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec_n}) {{
                    float4 dy = jt_add_rms_ldcs_f4(gr4 + v);
                    float4 tv = jt_add_rms_ldcs_f4(tr4 + v);
                    float4 xh = make_float4(tv.x * r, tv.y * r, tv.z * r, tv.w * r);
                    float4 g = make_float4(dy.x * wv[i].x, dy.y * wv[i].y, dy.z * wv[i].z, dy.w * wv[i].w);
                    gv[i] = g;
                    xhat[i] = xh;
                    sum_gx += g.x * xh.x + g.y * xh.y + g.z * xh.z + g.w * xh.w;
                    acc[i].x += dy.x * xh.x;
                    acc[i].y += dy.y * xh.y;
                    acc[i].z += dy.z * xh.z;
                    acc[i].w += dy.w * xh.w;
                }}
            }}
            float mean_gx = jt_add_rms_allsum(sum_gx) / {hidden}.0f;
            #pragma unroll
            for (int i = 0; i < {vec_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec_n}) {{
                    {gt_load}
                    dr4[v] = make_float4(
                        r * (gv[i].x - xhat[i].x * mean_gx){gt_add_x},
                        r * (gv[i].y - xhat[i].y * mean_gx){gt_add_y},
                        r * (gv[i].z - xhat[i].z * mean_gx){gt_add_z},
                        r * (gv[i].w - xhat[i].w * mean_gx){gt_add_w});
                }}
            }}
        }}
        float4* f4 = reinterpret_cast<float4*>(fold[warp]);
        #pragma unroll
        for (int i = 0; i < {vec_per}; i++) {{
            int v = lane + i * 32;
            if (v < {vec_n}) f4[v] = acc[i];
        }}
        __syncthreads();
        for (int j = threadIdx.x; j < {hidden}; j += {threads}) {{
            float t = 0.0f;
            #pragma unroll
            for (int k = 0; k < {wpb}; k++) t += fold[k][j];
            partial[(long long)blockIdx.x * {hidden} + j] = t;
        }}
    }}
        """
    elif has_vec:
        gt_ptr_decl = "const uint4* gt8 = grad_total + off;" if with_grad_total else ""
        gt_load = (
            "uint4 gtraw = jt_add_rms_ldcs_u4(gt8 + v);\n"
            "                    const in1_type* gtp = reinterpret_cast<const in1_type*>(&gtraw);"
        ) if with_grad_total else ""
        gt_add = " + static_cast<float>(gtp[k])" if with_grad_total else ""
        vec_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_backward_warp_vec(
            const uint4* __restrict__ grad_y, {gt_param_u4}
            const uint4* __restrict__ total, const {rstd_in_t}* __restrict__ rstd,
            const uint4* __restrict__ gamma,
            uint4* __restrict__ grad_x, float* __restrict__ partial, long long rows) {{
        __shared__ __align__(16) float fold[{wpb}][{hidden}];
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        uint4 wraw[{vec_per}];
        float acc[{vec_per}][8];
        #pragma unroll
        for (int i = 0; i < {vec_per}; i++) {{
            int v = lane + i * 32;
            if (v < {vec_n}) wraw[i] = jt_add_rms_ldg_u4(gamma + v);
            #pragma unroll
            for (int k = 0; k < 8; k++) acc[i][k] = 0.0f;
        }}
        for (long long row = (long long)blockIdx.x * {wpb} + warp; row < rows;
             row += (long long)gridDim.x * {wpb}) {{
            long long off = row * {vec_n}LL;
            const uint4* gr8 = grad_y + off;
            {gt_ptr_decl}
            const uint4* tr8 = total + off;
            uint4* dr8 = grad_x + off;
            float r = static_cast<float>(rstd[row]);
            uint4 draw[{vec_per}], traw[{vec_per}];
            float sum_gx = 0.0f;
            #pragma unroll
            for (int i = 0; i < {vec_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec_n}) {{
                    draw[i] = jt_add_rms_ldcs_u4(gr8 + v);
                    traw[i] = jt_add_rms_ldcs_u4(tr8 + v);
                    const in0_type* dp = reinterpret_cast<const in0_type*>(&draw[i]);
                    const {tot_in_t}* tp = reinterpret_cast<const {tot_in_t}*>(&traw[i]);
                    const {gam_in_t}* wp = reinterpret_cast<const {gam_in_t}*>(&wraw[i]);
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {{
                        float dy = static_cast<float>(dp[k]);
                        float xh = static_cast<float>(tp[k]) * r;
                        float g = dy * static_cast<float>(wp[k]);
                        sum_gx += g * xh;
                        acc[i][k] += dy * xh;
                    }}
                }}
            }}
            float mean_gx = jt_add_rms_allsum(sum_gx) / {hidden}.0f;
            #pragma unroll
            for (int i = 0; i < {vec_per}; i++) {{
                int v = lane + i * 32;
                if (v < {vec_n}) {{
                    {gt_load}
                    uint4 outraw;
                    const in0_type* dp = reinterpret_cast<const in0_type*>(&draw[i]);
                    const {tot_in_t}* tp = reinterpret_cast<const {tot_in_t}*>(&traw[i]);
                    const {gam_in_t}* wp = reinterpret_cast<const {gam_in_t}*>(&wraw[i]);
                    out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                    #pragma unroll
                    for (int k = 0; k < 8; k++) {{
                        float xh = static_cast<float>(tp[k]) * r;
                        float g = static_cast<float>(dp[k]) * static_cast<float>(wp[k]);
                        op[k] = out0_type(r * (g - xh * mean_gx){gt_add});
                    }}
                    dr8[v] = outraw;
                }}
            }}
        }}
        #pragma unroll
        for (int i = 0; i < {vec_per}; i++) {{
            int v = lane + i * 32;
            if (v < {vec_n}) {{
                int d = v * 8;
                #pragma unroll
                for (int k = 0; k < 8; k++) fold[warp][d + k] = acc[i][k];
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
        """

    gt_sc_decl = "const in1_type* gt = grad_total + row_off;" if with_grad_total else ""
    gt_sc_add = " + static_cast<float>(gt[j])" if with_grad_total else ""
    scalar_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_backward_warp_scalar(
            const in0_type* __restrict__ grad_y, {gt_param_sc}
            const {tot_in_t}* __restrict__ total, const {rstd_in_t}* __restrict__ rstd,
            const {gam_in_t}* __restrict__ gamma,
            out0_type* __restrict__ grad_x, float* __restrict__ partial, long long rows) {{
        __shared__ __align__(16) float fold[{wpb}][{hidden}];
        __shared__ __align__(16) float sw[{hidden}];
        int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
        for (int j = threadIdx.x; j < {hidden}; j += {threads})
            sw[j] = static_cast<float>(gamma[j]);
        __syncthreads();
        float acc[{per}];
        #pragma unroll
        for (int i = 0; i < {per}; i++) acc[i] = 0.0f;
        for (long long row = (long long)blockIdx.x * {wpb} + warp; row < rows;
             row += (long long)gridDim.x * {wpb}) {{
            long long row_off = row * {hidden}LL;
            const in0_type* gr = grad_y + row_off;
            {gt_sc_decl}
            const {tot_in_t}* tr = total + row_off;
            float r = static_cast<float>(rstd[row]);
            float gv[{per}], xhat[{per}];
            float sum_gx = 0.0f;
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                int j = lane + i * 32;
                bool in = j < {hidden};
                float dy = in ? static_cast<float>(gr[j]) : 0.0f;
                float xh = in ? static_cast<float>(tr[j]) * r : 0.0f;
                float g = in ? dy * sw[j] : 0.0f;
                gv[i] = g;
                xhat[i] = xh;
                sum_gx += g * xh;
                acc[i] += dy * xh;
            }}
            float mean_gx = jt_add_rms_allsum(sum_gx) / {hidden}.0f;
            out0_type* dr = grad_x + row_off;
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                int j = lane + i * 32;
                if (j < {hidden}) dr[j] = out0_type(r * (gv[i] - xhat[i] * mean_gx){gt_sc_add});
            }}
        }}
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            if (j < {hidden}) fold[warp][j] = acc[i];
        }}
        __syncthreads();
        for (int j = threadIdx.x; j < {hidden}; j += {threads}) {{
            float t = 0.0f;
            #pragma unroll
            for (int k = 0; k < {wpb}; k++) t += fold[k][j];
            partial[(long long)blockIdx.x * {hidden} + j] = t;
        }}
    }}
    __global__ static void add_rms_norm_backward_finish(
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
    vtype = "float4" if is_f32 else "uint4"
    tot_p = "in2_p" if with_grad_total else "in1_p"
    rstd_p = "in3_p" if with_grad_total else "in2_p"
    gam_p = "in4_p" if with_grad_total else "in3_p"
    tot_in = "in2" if with_grad_total else "in1"
    gt_arg_vec = f"reinterpret_cast<const {vtype}*>(in1_p)," if with_grad_total else ""
    gt_arg_sc = "in1_p," if with_grad_total else ""
    align_expr = (
        f"((size_t)in0_p | (size_t)in1_p | (size_t)in2_p | (size_t)in4_p | (size_t)out0_p)"
        if with_grad_total
        else "((size_t)in0_p | (size_t)in1_p | (size_t)in3_p | (size_t)out0_p)"
    )
    if has_vec:
        launch = f"""
        long long rows = {tot_in}->num / {hidden};
        int blocks = (int)(out2->num / {hidden});
        if (({align_expr} & 15u) == 0) {{
            add_rms_norm_backward_warp_vec<<<blocks, {threads}>>>(
                reinterpret_cast<const {vtype}*>(in0_p), {gt_arg_vec}
                reinterpret_cast<const {vtype}*>({tot_p}), {rstd_p},
                reinterpret_cast<const {vtype}*>({gam_p}),
                reinterpret_cast<{vtype}*>(out0_p), out2_p, rows);
        }} else {{
            add_rms_norm_backward_warp_scalar<<<blocks, {threads}>>>(
                in0_p, {gt_arg_sc} {tot_p}, {rstd_p}, {gam_p}, out0_p, out2_p, rows);
        }}
        add_rms_norm_backward_finish<<<{-(-hidden // 32)}, dim3(32, 8)>>>(
            out2_p, out1_p, blocks);
        CHECK(0 == cudaGetLastError());
        """
    else:
        launch = f"""
        long long rows = {tot_in}->num / {hidden};
        int blocks = (int)(out2->num / {hidden});
        add_rms_norm_backward_warp_scalar<<<blocks, {threads}>>>(
            in0_p, {gt_arg_sc} {tot_p}, {rstd_p}, {gam_p}, out0_p, out2_p, rows);
        add_rms_norm_backward_finish<<<{-(-hidden // 32)}, dim3(32, 8)>>>(
            out2_p, out1_p, blocks);
        CHECK(0 == cudaGetLastError());
        """
    return allsum + vec_kernel + scalar_kernel + launch


@lru_cache(maxsize=256)
def _add_block_fwd_src(hidden, eps, dtype, threads):
    """Cached block-per-row forward source for fused add+RMSNorm with
    host-dispatched vectorized vs scalar kernels and streaming L2 loads."""
    warps = threads // 32
    per = -(-hidden // threads)
    can_cache = per <= 16
    is_f32 = dtype == "float32"
    vec_w = 4 if is_f32 else 8
    has_vec = (hidden % vec_w) == 0
    vec_n = hidden // vec_w if has_vec else 1
    vec_per = -(-vec_n // threads) if has_vec else 1
    vec_cache = vec_per <= (4 if is_f32 else 2)

    helpers = _ADD_LDCS_HELPERS + """
    __device__ __forceinline__ float jt_add_rms_blk_warp_sum(float v) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        return v;
    }
    """
    vec_kernel = ""
    if has_vec and is_f32:
        if vec_cache:
            pass1 = f"""
        float4 cache[{vec_per}];
        float local = 0.0f;
        #pragma unroll
        for (int i = 0; i < {vec_per}; ++i) {{
            int v = tid + i * {threads};
            if (v < {vec_n}) {{
                float4 xv = jt_add_rms_ldcs_f4(x4 + v);
                float4 rv = jt_add_rms_ldcs_f4(r4 + v);
                float4 tv = make_float4(xv.x + rv.x, xv.y + rv.y, xv.z + rv.z, xv.w + rv.w);
                t4[v] = tv;
                cache[i] = tv;
                local += tv.x * tv.x + tv.y * tv.y + tv.z * tv.z + tv.w * tv.w;
            }}
        }}
            """
            pass2 = f"""
        #pragma unroll
        for (int i = 0; i < {vec_per}; ++i) {{
            int v = tid + i * {threads};
            if (v < {vec_n}) {{
                float4 tv = cache[i];
                float4 gv = jt_add_rms_ldg_f4(gamma + v);
                y4[v] = make_float4((tv.x * r) * gv.x, (tv.y * r) * gv.y,
                                    (tv.z * r) * gv.z, (tv.w * r) * gv.w);
            }}
        }}
            """
        else:
            pass1 = f"""
        float local = 0.0f;
        for (int v = tid; v < {vec_n}; v += {threads}) {{
            float4 xv = jt_add_rms_ldcs_f4(x4 + v);
            float4 rv = jt_add_rms_ldcs_f4(r4 + v);
            float4 tv = make_float4(xv.x + rv.x, xv.y + rv.y, xv.z + rv.z, xv.w + rv.w);
            t4[v] = tv;
            local += tv.x * tv.x + tv.y * tv.y + tv.z * tv.z + tv.w * tv.w;
        }}
            """
            pass2 = f"""
        for (int v = tid; v < {vec_n}; v += {threads}) {{
            float4 tv = t4[v];
            float4 gv = jt_add_rms_ldg_f4(gamma + v);
            y4[v] = make_float4((tv.x * r) * gv.x, (tv.y * r) * gv.y,
                                (tv.z * r) * gv.z, (tv.w * r) * gv.w);
        }}
            """
        vec_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_forward_block_vec(
            const float4* __restrict__ x, const float4* __restrict__ residual,
            const float4* __restrict__ gamma, float4* __restrict__ y,
            float4* __restrict__ total_out, out2_type* __restrict__ rstd,
            long long rows) {{
        __shared__ float warp_buf[{warps}];
        __shared__ float rstd_shared;
        long long row = blockIdx.x;
        if (row >= rows) return;
        int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        long long off = row * {vec_n}LL;
        const float4* x4 = x + off;
        const float4* r4 = residual + off;
        float4* y4 = y + off;
        float4* t4 = total_out + off;
        {pass1}
        local = jt_add_rms_blk_warp_sum(local);
        if (lane == 0) warp_buf[warp] = local;
        __syncthreads();
        if (warp == 0) {{
            float total_sq = lane < {warps} ? warp_buf[lane] : 0.0f;
            total_sq = jt_add_rms_blk_warp_sum(total_sq);
            if (lane == 0) {{
                rstd_shared = rsqrtf(total_sq / {hidden}.0f + {eps:.9g}f);
                rstd[row] = out2_type(rstd_shared);
            }}
        }}
        __syncthreads();
        float r = rstd_shared;
        {pass2}
    }}
        """
    elif has_vec:
        if vec_cache:
            pass1 = f"""
        uint4 cache[{vec_per}];
        float local = 0.0f;
        #pragma unroll
        for (int i = 0; i < {vec_per}; ++i) {{
            int v = tid + i * {threads};
            if (v < {vec_n}) {{
                uint4 xraw = jt_add_rms_ldcs_u4(x8 + v);
                uint4 rraw = jt_add_rms_ldcs_u4(r8 + v);
                uint4 traw;
                const in0_type* xp = reinterpret_cast<const in0_type*>(&xraw);
                const in1_type* rp = reinterpret_cast<const in1_type*>(&rraw);
                out1_type* tp = reinterpret_cast<out1_type*>(&traw);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {{
                    out1_type tv = out1_type(static_cast<float>(xp[k]) + static_cast<float>(rp[k]));
                    tp[k] = tv;
                    float val = static_cast<float>(tv);
                    local += val * val;
                }}
                t8[v] = traw;
                cache[i] = traw;
            }}
        }}
            """
            pass2 = f"""
        #pragma unroll
        for (int i = 0; i < {vec_per}; ++i) {{
            int v = tid + i * {threads};
            if (v < {vec_n}) {{
                uint4 traw = cache[i];
                uint4 graw = jt_add_rms_ldg_u4(gamma + v);
                uint4 outraw;
                const out1_type* tp = reinterpret_cast<const out1_type*>(&traw);
                const in2_type* gp = reinterpret_cast<const in2_type*>(&graw);
                out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                #pragma unroll
                for (int k = 0; k < 8; ++k) {{
                    float norm = static_cast<float>(out0_type(static_cast<float>(tp[k]) * r));
                    op[k] = out0_type(norm * static_cast<float>(gp[k]));
                }}
                y8[v] = outraw;
            }}
        }}
            """
        else:
            pass1 = f"""
        float local = 0.0f;
        for (int v = tid; v < {vec_n}; v += {threads}) {{
            uint4 xraw = jt_add_rms_ldcs_u4(x8 + v);
            uint4 rraw = jt_add_rms_ldcs_u4(r8 + v);
            uint4 traw;
            const in0_type* xp = reinterpret_cast<const in0_type*>(&xraw);
            const in1_type* rp = reinterpret_cast<const in1_type*>(&rraw);
            out1_type* tp = reinterpret_cast<out1_type*>(&traw);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {{
                out1_type tv = out1_type(static_cast<float>(xp[k]) + static_cast<float>(rp[k]));
                tp[k] = tv;
                float val = static_cast<float>(tv);
                local += val * val;
            }}
            t8[v] = traw;
        }}
            """
            pass2 = f"""
        for (int v = tid; v < {vec_n}; v += {threads}) {{
            uint4 traw = t8[v];
            uint4 graw = jt_add_rms_ldg_u4(gamma + v);
            uint4 outraw;
            const out1_type* tp = reinterpret_cast<const out1_type*>(&traw);
            const in2_type* gp = reinterpret_cast<const in2_type*>(&graw);
            out0_type* op = reinterpret_cast<out0_type*>(&outraw);
            #pragma unroll
            for (int k = 0; k < 8; ++k) {{
                float norm = static_cast<float>(out0_type(static_cast<float>(tp[k]) * r));
                op[k] = out0_type(norm * static_cast<float>(gp[k]));
            }}
            y8[v] = outraw;
        }}
            """
        vec_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_forward_block_vec(
            const uint4* __restrict__ x, const uint4* __restrict__ residual,
            const uint4* __restrict__ gamma, uint4* __restrict__ y,
            uint4* __restrict__ total_out, out2_type* __restrict__ rstd,
            long long rows) {{
        __shared__ float warp_buf[{warps}];
        __shared__ float rstd_shared;
        long long row = blockIdx.x;
        if (row >= rows) return;
        int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        long long off = row * {vec_n}LL;
        const uint4* x8 = x + off;
        const uint4* r8 = residual + off;
        uint4* y8 = y + off;
        uint4* t8 = total_out + off;
        {pass1}
        local = jt_add_rms_blk_warp_sum(local);
        if (lane == 0) warp_buf[warp] = local;
        __syncthreads();
        if (warp == 0) {{
            float total_sq = lane < {warps} ? warp_buf[lane] : 0.0f;
            total_sq = jt_add_rms_blk_warp_sum(total_sq);
            if (lane == 0) {{
                rstd_shared = rsqrtf(total_sq / {hidden}.0f + {eps:.9g}f);
                rstd[row] = out2_type(rstd_shared);
            }}
        }}
        __syncthreads();
        float r = rstd_shared;
        {pass2}
    }}
        """

    if can_cache:
        sc_p1 = f"""
        float cache[{per}];
        float local = 0.0f;
        #pragma unroll
        for (int i = 0; i < {per}; ++i) {{
            int j = tid + i * {threads};
            if (j < {hidden}) {{
                out1_type tv = out1_type(static_cast<float>(xr[j]) + static_cast<float>(rr[j]));
                tr[j] = tv;
                float val = static_cast<float>(tv);
                cache[i] = val;
                local += val * val;
            }}
        }}
        """
        sc_p2 = f"""
        #pragma unroll
        for (int i = 0; i < {per}; ++i) {{
            int j = tid + i * {threads};
            if (j < {hidden})
                yr[j] = out0_type(static_cast<float>(out0_type(cache[i] * r))
                                  * static_cast<float>(gamma[j]));
        }}
        """
    else:
        sc_p1 = f"""
        float local = 0.0f;
        for (int j = tid; j < {hidden}; j += {threads}) {{
            out1_type tv = out1_type(static_cast<float>(xr[j]) + static_cast<float>(rr[j]));
            tr[j] = tv;
            float val = static_cast<float>(tv);
            local += val * val;
        }}
        """
        sc_p2 = f"""
        for (int j = tid; j < {hidden}; j += {threads})
            yr[j] = out0_type(static_cast<float>(out0_type(static_cast<float>(tr[j]) * r))
                              * static_cast<float>(gamma[j]));
        """
    scalar_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_forward_block_scalar(
            const in0_type* __restrict__ x, const in1_type* __restrict__ residual,
            const in2_type* __restrict__ gamma, out0_type* __restrict__ y,
            out1_type* __restrict__ total_out, out2_type* __restrict__ rstd,
            long long rows) {{
        __shared__ float warp_buf[{warps}];
        __shared__ float rstd_shared;
        long long row = blockIdx.x;
        if (row >= rows) return;
        int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        long long row_off = row * {hidden}LL;
        const in0_type* xr = x + row_off;
        const in1_type* rr = residual + row_off;
        out0_type* yr = y + row_off;
        out1_type* tr = total_out + row_off;
        {sc_p1}
        local = jt_add_rms_blk_warp_sum(local);
        if (lane == 0) warp_buf[warp] = local;
        __syncthreads();
        if (warp == 0) {{
            float total_sq = lane < {warps} ? warp_buf[lane] : 0.0f;
            total_sq = jt_add_rms_blk_warp_sum(total_sq);
            if (lane == 0) {{
                rstd_shared = rsqrtf(total_sq / {hidden}.0f + {eps:.9g}f);
                rstd[row] = out2_type(rstd_shared);
            }}
        }}
        __syncthreads();
        float r = rstd_shared;
        {sc_p2}
    }}
    """
    vtype = "float4" if is_f32 else "uint4"
    if has_vec:
        launch = f"""
        long long rows = in0->num / {hidden};
        if ((((size_t)in0_p | (size_t)in1_p | (size_t)in2_p | (size_t)out0_p | (size_t)out1_p) & 15u) == 0) {{
            add_rms_norm_forward_block_vec<<<(unsigned)rows, {threads}>>>(
                reinterpret_cast<const {vtype}*>(in0_p), reinterpret_cast<const {vtype}*>(in1_p),
                reinterpret_cast<const {vtype}*>(in2_p), reinterpret_cast<{vtype}*>(out0_p),
                reinterpret_cast<{vtype}*>(out1_p), out2_p, rows);
        }} else {{
            add_rms_norm_forward_block_scalar<<<(unsigned)rows, {threads}>>>(
                in0_p, in1_p, in2_p, out0_p, out1_p, out2_p, rows);
        }}
        CHECK(0 == cudaGetLastError());
        """
    else:
        launch = f"""
        long long rows = in0->num / {hidden};
        add_rms_norm_forward_block_scalar<<<(unsigned)rows, {threads}>>>(
            in0_p, in1_p, in2_p, out0_p, out1_p, out2_p, rows);
        CHECK(0 == cudaGetLastError());
        """
    return helpers + vec_kernel + scalar_kernel + launch


@lru_cache(maxsize=256)
def _add_block_bwd_src(hidden, dtype, threads, with_grad_total):
    """Cached grid-stride fused block backward source for add+RMSNorm.
    Keeps per-thread weights in registers (zero gamma shared memory) and
    dispatches aligned vectorized kernels separately from the scalar fallback."""
    warps = threads // 32
    per = -(-hidden // threads)
    is_f32 = dtype == "float32"
    vec_w = 4 if is_f32 else 8
    vec_n = hidden // vec_w if (hidden % vec_w == 0) else 0
    vec_per = -(-vec_n // threads) if vec_n > 0 else 1
    has_vec = (vec_n > 0) and (vec_per <= (4 if is_f32 else 2))

    helpers = _ADD_LDCS_HELPERS + """
    __device__ __forceinline__ float jt_add_rms_blk_warp_sum(float v) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        return v;
    }
    """
    gt_param_f4 = "const float4* __restrict__ grad_total," if with_grad_total else ""
    gt_param_u4 = "const uint4* __restrict__ grad_total," if with_grad_total else ""
    gt_param_sc = "const in1_type* __restrict__ grad_total," if with_grad_total else ""
    tot_in_t = "in2_type" if with_grad_total else "in1_type"
    rstd_in_t = "in3_type" if with_grad_total else "in2_type"
    gam_in_t = "in4_type" if with_grad_total else "in3_type"

    vec_kernel = ""
    if has_vec and is_f32:
        gt_ptr_decl = "const float4* gt4 = grad_total + off;" if with_grad_total else ""
        gt_load = "float4 dt = jt_add_rms_ldcs_f4(gt4 + v);" if with_grad_total else ""
        gt_add_x = " + dt.x" if with_grad_total else ""
        gt_add_y = " + dt.y" if with_grad_total else ""
        gt_add_z = " + dt.z" if with_grad_total else ""
        gt_add_w = " + dt.w" if with_grad_total else ""
        vec_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_backward_block_vec(
            const float4* __restrict__ grad_y, {gt_param_f4}
            const float4* __restrict__ total, const {rstd_in_t}* __restrict__ rstd,
            const float4* __restrict__ gamma,
            float4* __restrict__ grad_x, float4* __restrict__ partial, long long rows) {{
        __shared__ float warp_buf[{warps}];
        __shared__ float mean_gx_shared;
        int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        float4 wv[{vec_per}], acc[{vec_per}];
        #pragma unroll
        for (int i = 0; i < {vec_per}; ++i) {{
            int v = tid + i * {threads};
            acc[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (v < {vec_n}) wv[i] = jt_add_rms_ldg_f4(gamma + v);
        }}
        for (long long row = blockIdx.x; row < rows; row += gridDim.x) {{
            long long off = row * {vec_n}LL;
            const float4* gr4 = grad_y + off;
            {gt_ptr_decl}
            const float4* tr4 = total + off;
            float4* dr4 = grad_x + off;
            float r = static_cast<float>(rstd[row]);
            float4 gv[{vec_per}], xhat[{vec_per}];
            float local = 0.0f;
            #pragma unroll
            for (int i = 0; i < {vec_per}; ++i) {{
                int v = tid + i * {threads};
                if (v < {vec_n}) {{
                    float4 dy = jt_add_rms_ldcs_f4(gr4 + v);
                    float4 tv = jt_add_rms_ldcs_f4(tr4 + v);
                    float4 xh = make_float4(tv.x * r, tv.y * r, tv.z * r, tv.w * r);
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
            local = jt_add_rms_blk_warp_sum(local);
            if (lane == 0) warp_buf[warp] = local;
            __syncthreads();
            if (warp == 0) {{
                float total_gx = lane < {warps} ? warp_buf[lane] : 0.0f;
                total_gx = jt_add_rms_blk_warp_sum(total_gx);
                if (lane == 0) mean_gx_shared = total_gx / {hidden}.0f;
            }}
            __syncthreads();
            float mean_gx = mean_gx_shared;
            #pragma unroll
            for (int i = 0; i < {vec_per}; ++i) {{
                int v = tid + i * {threads};
                if (v < {vec_n}) {{
                    {gt_load}
                    dr4[v] = make_float4(
                        r * (gv[i].x - xhat[i].x * mean_gx){gt_add_x},
                        r * (gv[i].y - xhat[i].y * mean_gx){gt_add_y},
                        r * (gv[i].z - xhat[i].z * mean_gx){gt_add_z},
                        r * (gv[i].w - xhat[i].w * mean_gx){gt_add_w});
                }}
            }}
        }}
        float4* p4 = partial + (long long)blockIdx.x * {vec_n};
        #pragma unroll
        for (int i = 0; i < {vec_per}; ++i) {{
            int v = tid + i * {threads};
            if (v < {vec_n}) p4[v] = acc[i];
        }}
    }}
        """
    elif has_vec:
        gt_ptr_decl = "const uint4* gt8 = grad_total + off;" if with_grad_total else ""
        gt_load = (
            "uint4 gtraw = jt_add_rms_ldcs_u4(gt8 + v);\n"
            "                    const in1_type* gtp = reinterpret_cast<const in1_type*>(&gtraw);"
        ) if with_grad_total else ""
        gt_add = " + static_cast<float>(gtp[k])" if with_grad_total else ""
        vec_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_backward_block_vec(
            const uint4* __restrict__ grad_y, {gt_param_u4}
            const uint4* __restrict__ total, const {rstd_in_t}* __restrict__ rstd,
            const uint4* __restrict__ gamma,
            uint4* __restrict__ grad_x, float* __restrict__ partial, long long rows) {{
        __shared__ float warp_buf[{warps}];
        __shared__ float mean_gx_shared;
        int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        uint4 wraw[{vec_per}];
        float acc[{vec_per}][8];
        #pragma unroll
        for (int i = 0; i < {vec_per}; ++i) {{
            int v = tid + i * {threads};
            if (v < {vec_n}) wraw[i] = jt_add_rms_ldg_u4(gamma + v);
            #pragma unroll
            for (int k = 0; k < 8; ++k) acc[i][k] = 0.0f;
        }}
        for (long long row = blockIdx.x; row < rows; row += gridDim.x) {{
            long long off = row * {vec_n}LL;
            const uint4* gr8 = grad_y + off;
            {gt_ptr_decl}
            const uint4* tr8 = total + off;
            uint4* dr8 = grad_x + off;
            float r = static_cast<float>(rstd[row]);
            uint4 draw[{vec_per}], traw[{vec_per}];
            float local = 0.0f;
            #pragma unroll
            for (int i = 0; i < {vec_per}; ++i) {{
                int v = tid + i * {threads};
                if (v < {vec_n}) {{
                    draw[i] = jt_add_rms_ldcs_u4(gr8 + v);
                    traw[i] = jt_add_rms_ldcs_u4(tr8 + v);
                    const in0_type* dp = reinterpret_cast<const in0_type*>(&draw[i]);
                    const {tot_in_t}* tp = reinterpret_cast<const {tot_in_t}*>(&traw[i]);
                    const {gam_in_t}* wp = reinterpret_cast<const {gam_in_t}*>(&wraw[i]);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {{
                        float dy = static_cast<float>(dp[k]);
                        float xh = static_cast<float>(tp[k]) * r;
                        float g = dy * static_cast<float>(wp[k]);
                        local += g * xh;
                        acc[i][k] += dy * xh;
                    }}
                }}
            }}
            local = jt_add_rms_blk_warp_sum(local);
            if (lane == 0) warp_buf[warp] = local;
            __syncthreads();
            if (warp == 0) {{
                float total_gx = lane < {warps} ? warp_buf[lane] : 0.0f;
                total_gx = jt_add_rms_blk_warp_sum(total_gx);
                if (lane == 0) mean_gx_shared = total_gx / {hidden}.0f;
            }}
            __syncthreads();
            float mean_gx = mean_gx_shared;
            #pragma unroll
            for (int i = 0; i < {vec_per}; ++i) {{
                int v = tid + i * {threads};
                if (v < {vec_n}) {{
                    {gt_load}
                    uint4 outraw;
                    const in0_type* dp = reinterpret_cast<const in0_type*>(&draw[i]);
                    const {tot_in_t}* tp = reinterpret_cast<const {tot_in_t}*>(&traw[i]);
                    const {gam_in_t}* wp = reinterpret_cast<const {gam_in_t}*>(&wraw[i]);
                    out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {{
                        float xh = static_cast<float>(tp[k]) * r;
                        float g = static_cast<float>(dp[k]) * static_cast<float>(wp[k]);
                        op[k] = out0_type(r * (g - xh * mean_gx){gt_add});
                    }}
                    dr8[v] = outraw;
                }}
            }}
        }}
        float* pr = partial + (long long)blockIdx.x * {hidden};
        #pragma unroll
        for (int i = 0; i < {vec_per}; ++i) {{
            int v = tid + i * {threads};
            if (v < {vec_n}) {{
                int d = v * 8;
                #pragma unroll
                for (int k = 0; k < 8; ++k) pr[d + k] = acc[i][k];
            }}
        }}
    }}
        """

    gt_sc_decl = "const in1_type* gt = grad_total + row_off;" if with_grad_total else ""
    gt_sc_add = " + static_cast<float>(gt[j])" if with_grad_total else ""
    scalar_kernel = f"""
    __global__ __launch_bounds__({threads}) static void add_rms_norm_backward_block_scalar(
            const in0_type* __restrict__ grad_y, {gt_param_sc}
            const {tot_in_t}* __restrict__ total, const {rstd_in_t}* __restrict__ rstd,
            const {gam_in_t}* __restrict__ gamma,
            out0_type* __restrict__ grad_x, float* __restrict__ partial, long long rows) {{
        __shared__ float warp_buf[{warps}];
        __shared__ float mean_gx_shared;
        int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
        float wv[{per}], acc[{per}];
        #pragma unroll
        for (int i = 0; i < {per}; ++i) {{
            int j = tid + i * {threads};
            acc[i] = 0.0f;
            wv[i] = j < {hidden} ? static_cast<float>(gamma[j]) : 0.0f;
        }}
        for (long long row = blockIdx.x; row < rows; row += gridDim.x) {{
            long long row_off = row * {hidden}LL;
            const in0_type* gr = grad_y + row_off;
            {gt_sc_decl}
            const {tot_in_t}* tr = total + row_off;
            out0_type* dr = grad_x + row_off;
            float r = static_cast<float>(rstd[row]);
            float gv[{per}], xhat[{per}];
            float local = 0.0f;
            #pragma unroll
            for (int i = 0; i < {per}; ++i) {{
                int j = tid + i * {threads};
                if (j < {hidden}) {{
                    float dy = static_cast<float>(gr[j]);
                    float xh = static_cast<float>(tr[j]) * r;
                    float g = dy * wv[i];
                    gv[i] = g;
                    xhat[i] = xh;
                    local += g * xh;
                    acc[i] += dy * xh;
                }}
            }}
            local = jt_add_rms_blk_warp_sum(local);
            if (lane == 0) warp_buf[warp] = local;
            __syncthreads();
            if (warp == 0) {{
                float total_gx = lane < {warps} ? warp_buf[lane] : 0.0f;
                total_gx = jt_add_rms_blk_warp_sum(total_gx);
                if (lane == 0) mean_gx_shared = total_gx / {hidden}.0f;
            }}
            __syncthreads();
            float mean_gx = mean_gx_shared;
            #pragma unroll
            for (int i = 0; i < {per}; ++i) {{
                int j = tid + i * {threads};
                if (j < {hidden}) dr[j] = out0_type(r * (gv[i] - xhat[i] * mean_gx){gt_sc_add});
            }}
        }}
        float* pr = partial + (long long)blockIdx.x * {hidden};
        #pragma unroll
        for (int i = 0; i < {per}; ++i) {{
            int j = tid + i * {threads};
            if (j < {hidden}) pr[j] = acc[i];
        }}
    }}
    __global__ static void add_rms_norm_backward_finish(
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
    vtype = "float4" if is_f32 else "uint4"
    tot_p = "in2_p" if with_grad_total else "in1_p"
    rstd_p = "in3_p" if with_grad_total else "in2_p"
    gam_p = "in4_p" if with_grad_total else "in3_p"
    tot_in = "in2" if with_grad_total else "in1"
    gt_arg_vec = f"reinterpret_cast<const {vtype}*>(in1_p)," if with_grad_total else ""
    gt_arg_sc = "in1_p," if with_grad_total else ""
    out2_vec = "reinterpret_cast<float4*>(out2_p)" if is_f32 else "out2_p"
    if with_grad_total:
        align_expr = (
            "((size_t)in0_p | (size_t)in1_p | (size_t)in2_p | (size_t)in4_p | (size_t)out0_p | (size_t)out2_p)"
            if is_f32
            else "((size_t)in0_p | (size_t)in1_p | (size_t)in2_p | (size_t)in4_p | (size_t)out0_p)"
        )
    else:
        align_expr = (
            "((size_t)in0_p | (size_t)in1_p | (size_t)in3_p | (size_t)out0_p | (size_t)out2_p)"
            if is_f32
            else "((size_t)in0_p | (size_t)in1_p | (size_t)in3_p | (size_t)out0_p)"
        )
    if has_vec:
        launch = f"""
        long long rows = {tot_in}->num / {hidden};
        int blocks = (int)(out2->num / {hidden});
        if (({align_expr} & 15u) == 0) {{
            add_rms_norm_backward_block_vec<<<blocks, {threads}>>>(
                reinterpret_cast<const {vtype}*>(in0_p), {gt_arg_vec}
                reinterpret_cast<const {vtype}*>({tot_p}), {rstd_p},
                reinterpret_cast<const {vtype}*>({gam_p}),
                reinterpret_cast<{vtype}*>(out0_p), {out2_vec}, rows);
        }} else {{
            add_rms_norm_backward_block_scalar<<<blocks, {threads}>>>(
                in0_p, {gt_arg_sc} {tot_p}, {rstd_p}, {gam_p}, out0_p, out2_p, rows);
        }}
        add_rms_norm_backward_finish<<<{-(-hidden // 32)}, dim3(32, 8)>>>(
            out2_p, out1_p, blocks);
        CHECK(0 == cudaGetLastError());
        """
    else:
        launch = f"""
        long long rows = {tot_in}->num / {hidden};
        int blocks = (int)(out2->num / {hidden});
        add_rms_norm_backward_block_scalar<<<blocks, {threads}>>>(
            in0_p, {gt_arg_sc} {tot_p}, {rstd_p}, {gam_p}, out0_p, out2_p, rows);
        add_rms_norm_backward_finish<<<{-(-hidden // 32)}, dim3(32, 8)>>>(
            out2_p, out1_p, blocks);
        CHECK(0 == cudaGetLastError());
        """
    return helpers + vec_kernel + scalar_kernel + launch


@lru_cache(maxsize=128)
def _add_wide_bwd_src(hidden, threads, with_grad_total):
    """Cached three-pass backward source for very wide rows (hidden > 8192)."""
    warps = threads // 32
    gt_param = "const in1_type* grad_total," if with_grad_total else ""
    tot_in_t = "in2_type" if with_grad_total else "in1_type"
    rstd_in_t = "in3_type" if with_grad_total else "in2_type"
    gam_in_t = "in4_type" if with_grad_total else "in3_type"
    tot_p = "in2_p" if with_grad_total else "in1_p"
    rstd_p = "in3_p" if with_grad_total else "in2_p"
    gam_p = "in4_p" if with_grad_total else "in3_p"
    tot_in = "in2" if with_grad_total else "in1"
    gt_arg = "in1_p," if with_grad_total else ""
    gt_add = "+ static_cast<float>(grad_total[index])" if with_grad_total else ""
    return f"""
    __device__ __forceinline__ float jt_add_rms_b2_warp_sum(float v) {{
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        return v;
    }}
    __global__ static void add_rms_norm_backward_x(
            const in0_type* grad_y, {gt_param}
            const {tot_in_t}* total, const {rstd_in_t}* rstd,
            const {gam_in_t}* gamma, out0_type* grad_x, long long rows) {{
        __shared__ float warp_buf[{warps}];
        __shared__ float mean_gx_shared;
        long long row = blockIdx.x;
        if (row >= rows) return;
        int tid = threadIdx.x;
        int lane = tid & 31;
        int warp = tid >> 5;
        long long base = row * {hidden}LL;
        float row_rstd = static_cast<float>(rstd[row]);
        float local = 0.0f;
        for (int j = tid; j < {hidden}; j += blockDim.x) {{
            long long index = base + j;
            float normalized = static_cast<float>(total[index]) * row_rstd;
            local += static_cast<float>(grad_y[index])
                * static_cast<float>(gamma[j]) * normalized;
        }}
        local = jt_add_rms_b2_warp_sum(local);
        if (lane == 0) warp_buf[warp] = local;
        __syncthreads();
        if (warp == 0) {{
            float total_gx = lane < {warps} ? warp_buf[lane] : 0.0f;
            total_gx = jt_add_rms_b2_warp_sum(total_gx);
            if (lane == 0) mean_gx_shared = total_gx / {hidden}.0f;
        }}
        __syncthreads();
        float mean_gx = mean_gx_shared;
        for (int j = tid; j < {hidden}; j += blockDim.x) {{
            long long index = base + j;
            float normalized = static_cast<float>(total[index]) * row_rstd;
            float g = static_cast<float>(grad_y[index])
                * static_cast<float>(gamma[j]);
            grad_x[index] = out0_type(
                row_rstd * (g - normalized * mean_gx) {gt_add});
        }}
    }}
    __global__ static void add_rms_norm_backward_gamma_partial(
            const in0_type* grad_y, const {tot_in_t}* total,
            const {rstd_in_t}* rstd, float* partial,
            long long rows, long long per_segment) {{
        __shared__ float sums[8][33];
        int channel = blockIdx.x * 32 + threadIdx.x;
        long long begin = (long long)blockIdx.y * per_segment;
        long long end = min(begin + per_segment, rows);
        float local = 0.0f;
        if (channel < {hidden})
            for (long long row = begin + threadIdx.y; row < end; row += 8) {{
                long long index = row * {hidden}LL + channel;
                local += static_cast<float>(grad_y[index])
                    * static_cast<float>(total[index])
                    * static_cast<float>(rstd[row]);
            }}
        sums[threadIdx.y][threadIdx.x] = local;
        __syncthreads();
        if (threadIdx.y == 0 && channel < {hidden}) {{
            float total_g = 0.0f;
            for (int k = 0; k < 8; k++) total_g += sums[k][threadIdx.x];
            partial[(long long)blockIdx.y * {hidden} + channel] = total_g;
        }}
    }}
    __global__ static void add_rms_norm_backward_gamma_finish(
            const float* partial, out1_type* grad_gamma, int segments) {{
        int channel = blockIdx.x * blockDim.x + threadIdx.x;
        if (channel >= {hidden}) return;
        float total_g = 0.0f;
        for (int s = 0; s < segments; s++)
            total_g += partial[(long long)s * {hidden} + channel];
        grad_gamma[channel] = out1_type(total_g);
    }}
    long long rows = {tot_in}->num / {hidden};
    int segments = (int)(out2->num / {hidden});
    long long per_segment = (rows + segments - 1) / segments;
    add_rms_norm_backward_x<<<(unsigned)rows, {threads}>>>(
        in0_p, {gt_arg} {tot_p}, {rstd_p}, {gam_p}, out0_p, rows);
    add_rms_norm_backward_gamma_partial<<<dim3({-(-hidden // 32)}, segments), dim3(32, 8)>>>(
        in0_p, {tot_p}, {rstd_p}, out2_p, rows, per_segment);
    add_rms_norm_backward_gamma_finish<<<{-(-hidden // 256)}, 256>>>(
        out2_p, out1_p, segments);
    CHECK(0 == cudaGetLastError());
    """


@lru_cache(maxsize=256)
def _add_rms_norm_training_cuda_cls(hidden_size, epsilon, dtype):
    threads = 32
    max_threads = 512 if (dtype == "float32" and hidden_size >= 2048) or hidden_size > 2048 else 256
    while threads < min(hidden_size, max_threads):
        threads *= 2
    per = -(-hidden_size // threads)
    can_block_fuse = per <= 16
    elem_bytes = 4 if dtype == "float32" else 2
    block_fwd_src = _add_block_fwd_src(hidden_size, epsilon, dtype, threads)
    if can_block_fuse:
        block_bwd_gt_src = _add_block_bwd_src(hidden_size, dtype, threads, True)
        block_bwd_nogt_src = _add_block_bwd_src(hidden_size, dtype, threads, False)
    else:
        block_bwd_gt_src = _add_wide_bwd_src(hidden_size, threads, True)
        block_bwd_nogt_src = _add_wide_bwd_src(hidden_size, threads, False)

    class AddRMSNormTrainingCUDA(jt.Function):
        def execute(self, x, residual, gamma):
            if not x._storage_is_contiguous():
                x = x.contiguous()
            if not residual._storage_is_contiguous():
                residual = residual.contiguous()
            if not gamma._storage_is_contiguous():
                gamma = gamma.contiguous()
            rows = int(x.numel()) // hidden_size
            self.warp = hidden_size <= _WARP_ROW_LIMIT and rows >= _WARP_ROWS_MIN
            if self.warp:
                wpb = _warp_warps_per_block(rows, hidden_size, False)
                y, total, rstd = jt.code(
                    [x.shape, x.shape, (rows,)],
                    [dtype, dtype, "float32"],
                    [x, residual, gamma],
                    cuda_src=_add_warp_fwd_src(hidden_size, epsilon, dtype, wpb),
                )
            else:
                y, total, rstd = jt.code(
                    [x.shape, x.shape, (rows,)],
                    [dtype, dtype, "float32"],
                    [x, residual, gamma],
                    cuda_src=block_fwd_src,
                )
            self.saved = total, rstd, gamma
            return y, total

        def grad(self, grad_y, grad_total):
            if grad_y is None and grad_total is None:
                return None, None, None
            total, rstd, gamma = self.saved
            if grad_y is None:
                if _jittor_dtype_name(grad_total.dtype) != dtype:
                    grad_total = grad_total.cast(dtype)
                return grad_total, grad_total, None
            if _jittor_dtype_name(grad_y.dtype) != dtype:
                grad_y = grad_y.cast(dtype)
            if not grad_y._storage_is_contiguous():
                grad_y = grad_y.contiguous()
            rows = int(total.numel()) // hidden_size

            # When grad_total is absent or comes from a pending fusable
            # elementwise op (e.g. loss scaling / dropout), keep the 4-input
            # RMSNorm backward so Jittor fuses `grad_total`'s producer directly
            # with `+ grad_rms` instead of writing & re-reading a full tensor.
            fuse_externally = (
                grad_total is None
                or _can_fuse_residual_grad_externally(grad_total)
            )
            if fuse_externally:
                if self.warp:
                    wpb = _warp_warps_per_block(rows, hidden_size, True)
                    blocks = min(_FUSED_BLOCKS, -(-rows // wpb))
                    grad_rms, grad_gamma, _ = jt.code(
                        [total.shape, gamma.shape, (blocks * hidden_size,)],
                        [dtype, dtype, "float32"],
                        [grad_y, total, rstd, gamma],
                        cuda_src=_add_warp_bwd_src(hidden_size, dtype, wpb, False),
                    )
                elif can_block_fuse:
                    target_blocks = 128 if (rows * hidden_size * elem_bytes >= (32 << 20)) else _FUSED_BLOCKS
                    blocks = min(target_blocks, rows)
                    grad_rms, grad_gamma, _ = jt.code(
                        [total.shape, gamma.shape, (blocks * hidden_size,)],
                        [dtype, dtype, "float32"],
                        [grad_y, total, rstd, gamma],
                        cuda_src=block_bwd_nogt_src,
                    )
                else:
                    segments = _gamma_segments(hidden_size, rows)
                    grad_rms, grad_gamma, _ = jt.code(
                        [total.shape, gamma.shape, (segments * hidden_size,)],
                        [dtype, dtype, "float32"],
                        [grad_y, total, rstd, gamma],
                        cuda_src=block_bwd_nogt_src,
                    )
                if grad_total is not None:
                    if _jittor_dtype_name(grad_total.dtype) != dtype:
                        grad_total = grad_total.cast(dtype)
                    grad_sum = grad_rms + grad_total
                else:
                    grad_sum = grad_rms
                return grad_sum, grad_sum, grad_gamma

            if _jittor_dtype_name(grad_total.dtype) != dtype:
                grad_total = grad_total.cast(dtype)
            if not grad_total._storage_is_contiguous():
                grad_total = grad_total.contiguous()
            if self.warp:
                wpb = _warp_warps_per_block(rows, hidden_size, True)
                blocks = min(_FUSED_BLOCKS, -(-rows // wpb))
                grad_sum, grad_gamma, _ = jt.code(
                    [total.shape, gamma.shape, (blocks * hidden_size,)],
                    [dtype, dtype, "float32"],
                    [grad_y, grad_total, total, rstd, gamma],
                    cuda_src=_add_warp_bwd_src(hidden_size, dtype, wpb, True),
                )
                return grad_sum, grad_sum, grad_gamma
            if can_block_fuse:
                target_blocks = 128 if (rows * hidden_size * elem_bytes >= (32 << 20)) else _FUSED_BLOCKS
                blocks = min(target_blocks, rows)
                grad_sum, grad_gamma, _ = jt.code(
                    [total.shape, gamma.shape, (blocks * hidden_size,)],
                    [dtype, dtype, "float32"],
                    [grad_y, grad_total, total, rstd, gamma],
                    cuda_src=block_bwd_gt_src,
                )
                return grad_sum, grad_sum, grad_gamma
            segments = _gamma_segments(hidden_size, rows)
            grad_sum, grad_gamma, _ = jt.code(
                [total.shape, gamma.shape, (segments * hidden_size,)],
                [dtype, dtype, "float32"],
                [grad_y, grad_total, total, rstd, gamma],
                cuda_src=block_bwd_gt_src,
            )
            return grad_sum, grad_sum, grad_gamma

    return AddRMSNormTrainingCUDA


def _supports_add_rms_norm_training(x, residual, gamma, epsilon=1e-6):
    if not (
        isinstance(x, jt.Var)
        and isinstance(residual, jt.Var)
        and isinstance(gamma, jt.Var)
        and _output_requires_grad(x, residual, gamma)
        and not _autocast_enabled()
    ):
        return False
    dtype = _jittor_dtype_name(x.dtype)
    if dtype != _jittor_dtype_name(residual.dtype) or dtype != _jittor_dtype_name(gamma.dtype):
        return False
    try:
        x_shape = tuple(int(size) for size in x.shape)
        res_shape = tuple(int(size) for size in residual.shape)
        gamma_shape = tuple(int(size) for size in gamma.shape)
        epsilon_value = float(epsilon)
    except (TypeError, ValueError, OverflowError):
        return False
    if not x_shape or x_shape != res_shape or any(size <= 0 for size in x_shape):
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


@optional_kernel("nn.add_rms_norm.training", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("float32", "float16", "bfloat16"),
                 supports=_supports_add_rms_norm_training)
def _add_rms_norm_training_cuda(x, residual, gamma, epsilon=1e-6):
    cls = _add_rms_norm_training_cuda_cls(
        int(x.shape[-1]), float(epsilon), _jittor_dtype_name(x.dtype)
    )
    return cls.apply(x, residual, gamma)
