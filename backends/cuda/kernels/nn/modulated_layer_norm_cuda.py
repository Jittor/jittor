"""Vectorized CUDA inference fast path for modulated LayerNorm."""

import functools
import math

from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import optional_kernel


def _resolve_mod_layout(x_shape, mod_shape):
    """Return ``(mod_rows, rows_per_mod)`` or ``None`` if unsupported."""
    hidden = x_shape[-1]
    if not mod_shape or mod_shape[-1] != hidden:
        return None
    total_rows = 1
    for s in x_shape[:-1]:
        total_rows *= s
    mod_rows = 1
    for s in mod_shape[:-1]:
        mod_rows *= s
    if mod_rows == 1:
        return 1, total_rows
    if mod_rows == total_rows and tuple(mod_shape) == tuple(x_shape):
        return total_rows, 1
    if len(x_shape) == 3:
        b, seq, _ = x_shape
        if mod_rows == b and mod_shape in ((b, hidden), (b, 1, hidden)):
            return b, seq
    return None


def _strided_modulation_operand(value, hidden, rows):
    """Find the dense storage and row pitch of a uniformly strided view."""
    strides = tuple(value._storage_strides())
    if strides[-1] != 1:
        return None
    pitch, outer = hidden, 1
    for extent, stride in reversed(tuple(zip(value.shape[:-1], strides[:-1]))):
        if extent == 1:
            continue
        if outer == 1:
            pitch = stride
            if pitch < hidden:
                return None
        elif stride != outer * pitch:
            return None
        outer *= extent

    root = value
    while not root._storage_is_contiguous():
        parent = root._view_base_object()
        if parent is None:
            if root.is_finished or not root._producer_is_view():
                return None
            parent = root._input(0)
        if parent is None or parent.var_ptr == root.var_ptr:
            return None
        root = parent
    offset = value._storage_offset() - root._storage_offset()
    if root.dtype != value.dtype or offset < 0 or offset + (rows - 1) * pitch + hidden > root.numel():
        return None
    return root, pitch, offset


def _modulated_layer_norm_contract(x, scale, shift, eps, one_plus=True):
    import jittor as jt
    if not all(isinstance(value, jt.Var) for value in (x, scale, shift)):
        return None
    if _output_requires_grad((x, scale, shift)):
        return None
    dtype = _jittor_dtype_name(x.dtype)
    if dtype not in ("bfloat16", "float16", "float32"):
        return None
    if _jittor_dtype_name(scale.dtype) != dtype or _jittor_dtype_name(shift.dtype) != dtype:
        return None
    try:
        x_shape = tuple(int(size) for size in x.shape)
        scale_shape = tuple(int(size) for size in scale.shape)
        shift_shape = tuple(int(size) for size in shift.shape)
        eps_value = float(eps)
    except (TypeError, ValueError, OverflowError):
        return None
    if len(x_shape) not in (2, 3) or any(size <= 0 for size in x_shape):
        return None
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        return None
    sc_layout = _resolve_mod_layout(x_shape, scale_shape)
    sh_layout = _resolve_mod_layout(x_shape, shift_shape)
    if sc_layout is None or sh_layout is None or sc_layout != sh_layout:
        return None
    mod_rows, rows_per_mod = sc_layout
    return x_shape, x_shape[-1], mod_rows, rows_per_mod, eps_value, bool(one_plus)


def _supports_modulated_layer_norm(x, scale, shift, eps, one_plus=True):
    return _modulated_layer_norm_contract(x, scale, shift, eps, one_plus=one_plus) is not None


@functools.lru_cache(maxsize=256)
def _modulated_ln_source(hidden, one_plus, mod_mode):
    """Build vectorized + register-cached CUDA kernel for modulated LayerNorm.

    ``mod_mode``: 0 = broadcast single row, 1 = per-batch (row / rows_per_mod),
    2 = per-row (row).
    """
    from jittor.backends.cuda.kernels.nn.layer_norm_cuda import _ln_vec_threads
    vec2_n, vec2_t, vec2_w = _ln_vec_threads(hidden, 8)
    vec4_n, vec4_t, vec4_w = _ln_vec_threads(hidden, 4)
    mod_idx_expr = (
        "0LL" if mod_mode == 0 else ("(row / rows_per_mod)" if mod_mode == 1 else "row")
    )
    one_plus_int = 1 if one_plus else 0
    return f"""
    __device__ __forceinline__ float mln_warp_sum(float v) {{
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        return v;
    }}
    __device__ __forceinline__ double mln_warp_sum_d(double v) {{
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
        return v;
    }}
    __device__ __forceinline__ float mln_block_sum(float v, float* buf, int tid, int warps) {{
        int lane = tid & 31, warp = tid >> 5;
        v = mln_warp_sum(v);
        if (lane == 0) buf[warp] = v;
        __syncthreads();
        if (warp == 0) {{
            float t = lane < warps ? buf[lane] : 0.0f;
            t = mln_warp_sum(t);
            if (lane == 0) buf[0] = t;
        }}
        __syncthreads();
        return buf[0];
    }}
    __device__ __forceinline__ double mln_block_sum_d(double v, double* buf, int tid, int warps) {{
        int lane = tid & 31, warp = tid >> 5;
        v = mln_warp_sum_d(v);
        if (lane == 0) buf[warp] = v;
        __syncthreads();
        if (warp == 0) {{
            double t = lane < warps ? buf[lane] : 0.0;
            t = mln_warp_sum_d(t);
            if (lane == 0) buf[0] = t;
        }}
        __syncthreads();
        return buf[0];
    }}
    __global__ __launch_bounds__({vec2_t}) static void mln_vec2b(
            const in0_type* __restrict__ x,
            const in1_type* __restrict__ scale,
            const in2_type* __restrict__ shift,
            out0_type* __restrict__ y,
            long long rows_per_mod, long long scale_pitch, long long shift_pitch, double eps) {{
        if (sizeof(in0_type) == 2 && sizeof(out0_type) == 2 && {vec2_n} > 0) {{
            constexpr int kVecN = {vec2_n};
            constexpr int kThreads = {vec2_t};
            constexpr int kWarps = {vec2_w};
            constexpr int kVecPer = (kVecN + kThreads - 1) / kThreads;
            constexpr bool kVecCache = kVecPer <= 4;
            long long row = blockIdx.x;
            long long mrow = {mod_idx_expr};
            scale += mrow * scale_pitch;
            shift += mrow * shift_pitch;
            int tid = threadIdx.x;
            __shared__ float wbuf[kWarps];
            __shared__ double dwbuf[kWarps];
            __shared__ float s_mean, s_inv;
            __shared__ double d_mean, d_inv;
            __shared__ int use_d;
            const uint4* x8 = reinterpret_cast<const uint4*>(x + row * {hidden}LL);
            const uint4* sc8 = reinterpret_cast<const uint4*>(scale);
            const uint4* sh8 = reinterpret_cast<const uint4*>(shift);
            uint4* y8 = reinterpret_cast<uint4*>(y + row * {hidden}LL);
            uint4 cache[kVecCache && kVecPer > 0 ? kVecPer : 1];
            float sum = 0.0f;
            if (kVecCache) {{
                #pragma unroll
                for (int i = 0; i < kVecPer; ++i) {{
                    int v = tid + i * kThreads;
                    if (v < kVecN) {{
                        uint4 raw = x8[v];
                        cache[i] = raw;
                        const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) sum += static_cast<float>(vp[k]);
                    }}
                }}
            }} else {{
                for (int v = tid; v < kVecN; v += kThreads) {{
                    uint4 raw = x8[v];
                    const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) sum += static_cast<float>(vp[k]);
                }}
            }}
            float total = mln_block_sum(sum, wbuf, tid, kWarps);
            if (tid == 0) {{
                use_d = !isfinite(total);
                if (!use_d) s_mean = total / {hidden}.0f;
            }}
            __syncthreads();
            if (!use_d) {{
                float mean = s_mean, var = 0.0f;
                if (kVecCache) {{
                    #pragma unroll
                    for (int i = 0; i < kVecPer; ++i) {{
                        int v = tid + i * kThreads;
                        if (v < kVecN) {{
                            const in0_type* vp = reinterpret_cast<const in0_type*>(&cache[i]);
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) {{
                                float d = static_cast<float>(vp[k]) - mean;
                                var += d * d;
                            }}
                        }}
                    }}
                }} else {{
                    for (int v = tid; v < kVecN; v += kThreads) {{
                        uint4 raw = x8[v];
                        const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {{
                            float d = static_cast<float>(vp[k]) - mean;
                            var += d * d;
                        }}
                    }}
                }}
                float vtot = mln_block_sum(var, wbuf, tid, kWarps);
                if (tid == 0) {{
                    use_d = !isfinite(vtot);
                    if (!use_d) s_inv = rsqrtf(vtot / {hidden}.0f + static_cast<float>(eps));
                }}
                __syncthreads();
            }}
            if (!use_d) {{
                float mean = s_mean, inv = s_inv;
                if (kVecCache) {{
                    #pragma unroll
                    for (int i = 0; i < kVecPer; ++i) {{
                        int v = tid + i * kThreads;
                        if (v < kVecN) {{
                            const in0_type* xp = reinterpret_cast<const in0_type*>(&cache[i]);
                            uint4 sc_raw = sc8[v], sh_raw = sh8[v], out_raw;
                            const in1_type* scp = reinterpret_cast<const in1_type*>(&sc_raw);
                            const in2_type* shp = reinterpret_cast<const in2_type*>(&sh_raw);
                            out0_type* yp = reinterpret_cast<out0_type*>(&out_raw);
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) {{
                                out0_type nv = out0_type((static_cast<float>(xp[k]) - mean) * inv);
                                out0_type ms = {one_plus_int}
                                    ? out0_type(1.0f + static_cast<float>(scp[k]))
                                    : out0_type(static_cast<float>(scp[k]));
                                out0_type pr = out0_type(static_cast<float>(nv) * static_cast<float>(ms));
                                yp[k] = out0_type(static_cast<float>(pr) + static_cast<float>(shp[k]));
                            }}
                            y8[v] = out_raw;
                        }}
                    }}
                }} else {{
                    for (int v = tid; v < kVecN; v += kThreads) {{
                        uint4 x_raw = x8[v], sc_raw = sc8[v], sh_raw = sh8[v], out_raw;
                        const in0_type* xp = reinterpret_cast<const in0_type*>(&x_raw);
                        const in1_type* scp = reinterpret_cast<const in1_type*>(&sc_raw);
                        const in2_type* shp = reinterpret_cast<const in2_type*>(&sh_raw);
                        out0_type* yp = reinterpret_cast<out0_type*>(&out_raw);
                        #pragma unroll
                        for (int k = 0; k < 8; ++k) {{
                            out0_type nv = out0_type((static_cast<float>(xp[k]) - mean) * inv);
                            out0_type ms = {one_plus_int}
                                ? out0_type(1.0f + static_cast<float>(scp[k]))
                                : out0_type(static_cast<float>(scp[k]));
                            out0_type pr = out0_type(static_cast<float>(nv) * static_cast<float>(ms));
                            yp[k] = out0_type(static_cast<float>(pr) + static_cast<float>(shp[k]));
                        }}
                        y8[v] = out_raw;
                    }}
                }}
                return;
            }}
            double dsum = 0.0;
            for (int d = tid; d < {hidden}; d += kThreads) dsum += static_cast<double>(x[row * {hidden}LL + d]);
            double dtot = mln_block_sum_d(dsum, dwbuf, tid, kWarps);
            if (tid == 0) d_mean = dtot / {hidden}.0;
            __syncthreads();
            double dm = d_mean, dvar = 0.0;
            for (int d = tid; d < {hidden}; d += kThreads) {{
                double df = static_cast<double>(x[row * {hidden}LL + d]) - dm;
                dvar += df * df;
            }}
            double dvtot = mln_block_sum_d(dvar, dwbuf, tid, kWarps);
            if (tid == 0) d_inv = 1.0 / sqrt(dvtot / {hidden}.0 + eps);
            __syncthreads();
            double di = d_inv;
            for (int d = tid; d < {hidden}; d += kThreads) {{
                out0_type nv = out0_type(static_cast<float>((static_cast<double>(x[row * {hidden}LL + d]) - dm) * di));
                out0_type ms = {one_plus_int}
                    ? out0_type(1.0f + static_cast<float>(scale[d]))
                    : out0_type(static_cast<float>(scale[d]));
                out0_type pr = out0_type(static_cast<float>(nv) * static_cast<float>(ms));
                y[row * {hidden}LL + d] = out0_type(static_cast<float>(pr) + static_cast<float>(shift[d]));
            }}
        }}
    }}
    __global__ __launch_bounds__({vec4_t}) static void mln_vec4b(
            const in0_type* __restrict__ x,
            const in1_type* __restrict__ scale,
            const in2_type* __restrict__ shift,
            out0_type* __restrict__ y,
            long long rows_per_mod, long long scale_pitch, long long shift_pitch, double eps) {{
        if (sizeof(in0_type) == 4 && sizeof(out0_type) == 4 && {vec4_n} > 0) {{
            constexpr int kVecN = {vec4_n};
            constexpr int kThreads = {vec4_t};
            constexpr int kWarps = {vec4_w};
            constexpr int kVecPer = (kVecN + kThreads - 1) / kThreads;
            constexpr bool kVecCache = kVecPer <= 4;
            long long row = blockIdx.x;
            long long mrow = {mod_idx_expr};
            scale += mrow * scale_pitch;
            shift += mrow * shift_pitch;
            int tid = threadIdx.x;
            __shared__ float wbuf[kWarps];
            __shared__ double dwbuf[kWarps];
            __shared__ float s_mean, s_inv;
            __shared__ double d_mean, d_inv;
            __shared__ int use_d;
            const float4* x4 = reinterpret_cast<const float4*>(x + row * {hidden}LL);
            const float4* sc4 = reinterpret_cast<const float4*>(scale);
            const float4* sh4 = reinterpret_cast<const float4*>(shift);
            float4* y4 = reinterpret_cast<float4*>(y + row * {hidden}LL);
            float4 cache[kVecCache && kVecPer > 0 ? kVecPer : 1];
            float sum = 0.0f;
            if (kVecCache) {{
                #pragma unroll
                for (int i = 0; i < kVecPer; ++i) {{
                    int v = tid + i * kThreads;
                    if (v < kVecN) {{
                        float4 r = x4[v];
                        cache[i] = r;
                        sum += r.x + r.y + r.z + r.w;
                    }}
                }}
            }} else {{
                for (int v = tid; v < kVecN; v += kThreads) {{
                    float4 r = x4[v];
                    sum += r.x + r.y + r.z + r.w;
                }}
            }}
            float total = mln_block_sum(sum, wbuf, tid, kWarps);
            if (tid == 0) {{
                use_d = !isfinite(total);
                if (!use_d) s_mean = total / {hidden}.0f;
            }}
            __syncthreads();
            if (!use_d) {{
                float mean = s_mean, var = 0.0f;
                if (kVecCache) {{
                    #pragma unroll
                    for (int i = 0; i < kVecPer; ++i) {{
                        int v = tid + i * kThreads;
                        if (v < kVecN) {{
                            float4 r = cache[i];
                            float dx = r.x - mean, dy = r.y - mean, dz = r.z - mean, dw = r.w - mean;
                            var += dx * dx + dy * dy + dz * dz + dw * dw;
                        }}
                    }}
                }} else {{
                    for (int v = tid; v < kVecN; v += kThreads) {{
                        float4 r = x4[v];
                        float dx = r.x - mean, dy = r.y - mean, dz = r.z - mean, dw = r.w - mean;
                        var += dx * dx + dy * dy + dz * dz + dw * dw;
                    }}
                }}
                float vtot = mln_block_sum(var, wbuf, tid, kWarps);
                if (tid == 0) {{
                    use_d = !isfinite(vtot);
                    if (!use_d) s_inv = rsqrtf(vtot / {hidden}.0f + static_cast<float>(eps));
                }}
                __syncthreads();
            }}
            if (!use_d) {{
                float mean = s_mean, inv = s_inv;
                if (kVecCache) {{
                    #pragma unroll
                    for (int i = 0; i < kVecPer; ++i) {{
                        int v = tid + i * kThreads;
                        if (v < kVecN) {{
                            float4 r = cache[i], sc = sc4[v], sh = sh4[v];
                            float s0 = {one_plus_int} ? (1.0f + sc.x) : sc.x;
                            float s1 = {one_plus_int} ? (1.0f + sc.y) : sc.y;
                            float s2 = {one_plus_int} ? (1.0f + sc.z) : sc.z;
                            float s3 = {one_plus_int} ? (1.0f + sc.w) : sc.w;
                            y4[v] = make_float4(
                                ((r.x - mean) * inv) * s0 + sh.x,
                                ((r.y - mean) * inv) * s1 + sh.y,
                                ((r.z - mean) * inv) * s2 + sh.z,
                                ((r.w - mean) * inv) * s3 + sh.w);
                        }}
                    }}
                }} else {{
                    for (int v = tid; v < kVecN; v += kThreads) {{
                        float4 r = x4[v], sc = sc4[v], sh = sh4[v];
                        float s0 = {one_plus_int} ? (1.0f + sc.x) : sc.x;
                        float s1 = {one_plus_int} ? (1.0f + sc.y) : sc.y;
                        float s2 = {one_plus_int} ? (1.0f + sc.z) : sc.z;
                        float s3 = {one_plus_int} ? (1.0f + sc.w) : sc.w;
                        y4[v] = make_float4(
                            ((r.x - mean) * inv) * s0 + sh.x,
                            ((r.y - mean) * inv) * s1 + sh.y,
                            ((r.z - mean) * inv) * s2 + sh.z,
                            ((r.w - mean) * inv) * s3 + sh.w);
                    }}
                }}
                return;
            }}
            double dsum = 0.0;
            for (int d = tid; d < {hidden}; d += kThreads) dsum += static_cast<double>(x[row * {hidden}LL + d]);
            double dtot = mln_block_sum_d(dsum, dwbuf, tid, kWarps);
            if (tid == 0) d_mean = dtot / {hidden}.0;
            __syncthreads();
            double dm = d_mean, dvar = 0.0;
            for (int d = tid; d < {hidden}; d += kThreads) {{
                double df = static_cast<double>(x[row * {hidden}LL + d]) - dm;
                dvar += df * df;
            }}
            double dvtot = mln_block_sum_d(dvar, dwbuf, tid, kWarps);
            if (tid == 0) d_inv = 1.0 / sqrt(dvtot / {hidden}.0 + eps);
            __syncthreads();
            double di = d_inv;
            for (int d = tid; d < {hidden}; d += kThreads) {{
                float nv = static_cast<float>((static_cast<double>(x[row * {hidden}LL + d]) - dm) * di);
                float sv = static_cast<float>(scale[d]);
                float ms = {one_plus_int} ? (1.0f + sv) : sv;
                y[row * {hidden}LL + d] = out0_type(nv * ms + static_cast<float>(shift[d]));
            }}
        }}
    }}
    __global__ static void mln_scalar(
            const in0_type* __restrict__ x,
            const in1_type* __restrict__ scale,
            const in2_type* __restrict__ shift,
            out0_type* __restrict__ y,
            long long rows_per_mod, long long scale_pitch, long long shift_pitch, double eps) {{
        long long row = blockIdx.x;
        long long mrow = {mod_idx_expr};
        scale += mrow * scale_pitch;
        shift += mrow * shift_pitch;
        int tid = threadIdx.x;
        __shared__ float wbuf[4];
        __shared__ double dwbuf[4];
        __shared__ float s_mean, s_inv;
        __shared__ double d_mean, d_inv;
        __shared__ int use_d;
        float sum = 0.0f;
        for (int d = tid; d < {hidden}; d += 128) sum += static_cast<float>(x[row * {hidden}LL + d]);
        float total = mln_block_sum(sum, wbuf, tid, 4);
        if (tid == 0) {{
            use_d = !isfinite(total);
            if (!use_d) s_mean = total / {hidden}.0f;
        }}
        __syncthreads();
        if (!use_d) {{
            float mean = s_mean, var = 0.0f;
            for (int d = tid; d < {hidden}; d += 128) {{
                float df = static_cast<float>(x[row * {hidden}LL + d]) - mean;
                var += df * df;
            }}
            float vtot = mln_block_sum(var, wbuf, tid, 4);
            if (tid == 0) {{
                use_d = !isfinite(vtot);
                if (!use_d) s_inv = rsqrtf(vtot / {hidden}.0f + static_cast<float>(eps));
            }}
            __syncthreads();
        }}
        if (use_d) {{
            double dsum = 0.0;
            for (int d = tid; d < {hidden}; d += 128) dsum += static_cast<double>(x[row * {hidden}LL + d]);
            double dtot = mln_block_sum_d(dsum, dwbuf, tid, 4);
            if (tid == 0) d_mean = dtot / {hidden}.0;
            __syncthreads();
            double dm = d_mean, dvar = 0.0;
            for (int d = tid; d < {hidden}; d += 128) {{
                double df = static_cast<double>(x[row * {hidden}LL + d]) - dm;
                dvar += df * df;
            }}
            double dvtot = mln_block_sum_d(dvar, dwbuf, tid, 4);
            if (tid == 0) d_inv = 1.0 / sqrt(dvtot / {hidden}.0 + eps);
            __syncthreads();
        }}
        for (int d = tid; d < {hidden}; d += 128) {{
            float nf = use_d
                ? static_cast<float>((static_cast<double>(x[row * {hidden}LL + d]) - d_mean) * d_inv)
                : (static_cast<float>(x[row * {hidden}LL + d]) - s_mean) * s_inv;
            float sv = static_cast<float>(scale[d]);
            float shv = static_cast<float>(shift[d]);
            if (sizeof(out0_type) == 2) {{
                out0_type nv = out0_type(nf);
                out0_type ms = {one_plus_int} ? out0_type(1.0f + sv) : out0_type(sv);
                out0_type pr = out0_type(static_cast<float>(nv) * static_cast<float>(ms));
                y[row * {hidden}LL + d] = out0_type(static_cast<float>(pr) + shv);
            }} else {{
                float ms = {one_plus_int} ? (1.0f + sv) : sv;
                y[row * {hidden}LL + d] = out0_type(nf * ms + shv);
            }}
        }}
    }}
    double eps = data["eps"];
    long long rows = in0->num / {hidden};
    long long rows_per_mod = (long long)data["rows_per_mod"];
    long long scale_pitch = (long long)data["scale_pitch"];
    long long shift_pitch = (long long)data["shift_pitch"];
    const auto* scale = in1_p + (long long)data["scale_offset"];
    const auto* shift = in2_p + (long long)data["shift_offset"];
    bool aligned16 = ((((size_t)in0_p | (size_t)scale | (size_t)shift | (size_t)out0_p |
                        (size_t)scale_pitch * sizeof(in1_type) |
                        (size_t)shift_pitch * sizeof(in2_type)) & 15u) == 0);
    if (rows > 0) {{
        if (sizeof(in0_type) == 2 && sizeof(out0_type) == 2 && {vec2_n} > 0 && aligned16) {{
            mln_vec2b<<<(unsigned)rows, {vec2_t}>>>(
                in0_p, scale, shift, out0_p, rows_per_mod, scale_pitch, shift_pitch, eps);
        }} else if (sizeof(in0_type) == 4 && sizeof(out0_type) == 4 && {vec4_n} > 0 && aligned16) {{
            mln_vec4b<<<(unsigned)rows, {vec4_t}>>>(
                in0_p, scale, shift, out0_p, rows_per_mod, scale_pitch, shift_pitch, eps);
        }} else {{
            mln_scalar<<<(unsigned)rows, 128>>>(
                in0_p, scale, shift, out0_p, rows_per_mod, scale_pitch, shift_pitch, eps);
        }}
    }}
    CHECK(0 == cudaGetLastError());
    """


@optional_kernel("nn.layer_norm.modulated", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("bfloat16", "float16", "float32"),
                 supports=_supports_modulated_layer_norm)
def _modulated_layer_norm_no_grad_cuda(x, scale, shift, eps, one_plus=True):
    """Fuse inference LayerNorm with per-channel scale and shift."""
    import jittor as jt
    contract = _modulated_layer_norm_contract(x, scale, shift, eps, one_plus=one_plus)
    if contract is None:
        return None
    x_shape, hidden, mod_rows, rows_per_mod, eps_value, one_plus_flag = contract
    total_rows = int(x.numel()) // hidden
    if mod_rows == 1:
        mod_mode = 0
    elif mod_rows == total_rows and rows_per_mod == 1:
        mod_mode = 2
    else:
        mod_mode = 1
    scale_layout = _strided_modulation_operand(scale, hidden, mod_rows)
    shift_layout = _strided_modulation_operand(shift, hidden, mod_rows)
    scale, scale_pitch, scale_offset = scale_layout or (scale.reshape((mod_rows, hidden)), hidden, 0)
    shift, shift_pitch, shift_offset = shift_layout or (shift.reshape((mod_rows, hidden)), hidden, 0)
    return _stop_grad_outputs(jt.code(
        x.shape,
        x.dtype,
        [x, scale, shift],
        cuda_src=_modulated_ln_source(hidden, one_plus_flag, mod_mode),
        data={
            "eps": eps_value,
            "rows_per_mod": float(rows_per_mod),
            "scale_pitch": float(scale_pitch),
            "shift_pitch": float(shift_pitch),
            "scale_offset": float(scale_offset),
            "shift_offset": float(shift_offset),
        },
    ))


__all__ = ["_modulated_layer_norm_no_grad_cuda"]
