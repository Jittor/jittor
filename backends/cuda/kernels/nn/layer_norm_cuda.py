"""CUDA inference fast path for :func:`jittor.nn.layer_norm`."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import functools
import os

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import dispatch_context, native_rule, optional_kernel


@native_rule("layer_norm_inference")
def _supports_layer_norm_inference(
        x, normalized_shape, weight, bias, eps, *, allow_bfloat16=True):
    if _output_requires_grad(x, weight, bias):
        return False
    input_dtype = _jittor_dtype_name(x.dtype)
    supported_dtypes = ("float16", "float32")
    if allow_bfloat16:
        supported_dtypes += ("bfloat16",)
    if len(normalized_shape) != 1 or _jittor_dtype_name(input_dtype) not in supported_dtypes:
        return False
    hidden = int(normalized_shape[0])
    has_weight = isinstance(weight, jt.Var)
    has_bias = isinstance(bias, jt.Var)
    if has_weight and has_bias:
        if int(weight.numel()) != hidden or int(bias.numel()) != hidden:
            return False
    elif has_weight and not has_bias:
        if int(weight.numel()) != hidden:
            return False
    elif not has_weight and not has_bias:
        if os.environ.get("JITTOR_LAYERNORM_SCALAR_FAST", "1") == "0":
            return False
    else:
        return False
    if int(x.shape[-1]) != hidden:
        return False
    return True


#: Rows at most this wide go a warp each, when there are at least this many
#: of them; see `_warp_rows_source`. Fewer rows leave the device mostly idle
#: at eight a block: BERT inference's 128 x 768 took 4.7 us that way, 3.4 a
#: block per row.
_WARP_ROW_LIMIT = 1024
_WARP_ROWS_MIN = 1024


def _warp_rows(x, hidden):
    return hidden <= _WARP_ROW_LIMIT and int(x.numel()) // hidden >= _WARP_ROWS_MIN


@functools.lru_cache(maxsize=256)
def _warp_rows_source(hidden, affine):
    """A warp per row: the row in registers, both reductions by shuffles.

    The block-per-row kernel below gives a 320-wide row -- a Stable Diffusion
    transformer's -- 128 threads with two or three values each and four block
    barriers, and an SD1.5 sampling step spent 7.6 us a call on it, PyTorch
    4. A warp per row needs no shared memory and no barrier. The double-
    precision pass for a row whose float sums overflow is kept, warp-wide.

    `affine` is (args, float expression, double expression) of the scale and
    offset of element `j`.
    """
    args, scale_f, scale_d = affine
    per = -(-hidden // 32)
    return f"""
    __device__ __forceinline__ float jt_ln_allsum(float v) {{
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        return v;
    }}
    __device__ __forceinline__ double jt_ln_allsum_double(double v) {{
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        return v;
    }}
    __global__ static void kernel_warp_rows(in0_type* x, {args}out0_type* y, long long rows, double eps) {{
        long long row = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        int lane = threadIdx.x & 31;
        if (row >= rows) return;
        const in0_type* xr = x + row * {hidden};
        out0_type* yr = y + row * {hidden};
        float cache[{per}];
        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            cache[i] = j < {hidden} ? static_cast<float>(xr[j]) : 0.0f;
            sum += cache[i];
        }}
        sum = jt_ln_allsum(sum);
        bool use_double = !isfinite(sum);
        float mean = sum / {hidden}, inv_std = 0.0f;
        if (!use_double) {{
            float var = 0.0f;
            #pragma unroll
            for (int i = 0; i < {per}; i++) {{
                float d = cache[i] - mean;
                if (lane + i * 32 < {hidden}) var += d * d;
            }}
            var = jt_ln_allsum(var);
            use_double = !isfinite(var);
            inv_std = rsqrtf(var / {hidden} + static_cast<float>(eps));
        }}
        if (use_double) {{
            double dsum = 0.0;
            for (int j = lane; j < {hidden}; j += 32) dsum += static_cast<double>(xr[j]);
            double dmean = jt_ln_allsum_double(dsum) / {hidden};
            double dvar = 0.0;
            for (int j = lane; j < {hidden}; j += 32) {{
                double d = static_cast<double>(xr[j]) - dmean;
                dvar += d * d;
            }}
            double dinv = 1.0 / sqrt(jt_ln_allsum_double(dvar) / {hidden} + eps);
            for (int j = lane; j < {hidden}; j += 32)
                yr[j] = out0_type((static_cast<double>(xr[j]) - dmean) * dinv {scale_d});
            return;
        }}
        #pragma unroll
        for (int i = 0; i < {per}; i++) {{
            int j = lane + i * 32;
            if (j < {hidden}) yr[j] = out0_type((cache[i] - mean) * inv_std {scale_f});
        }}
    }}
    """


@functools.lru_cache(maxsize=256)
def _warp_rows_launch(hidden, pointers):
    return f"""
    double eps = data["eps"];
    long long rows = in0->num / {hidden};
    kernel_warp_rows<<<(unsigned)((rows + 7) / 8), 256>>>({pointers}, rows, eps);
    """


def _ln_vec_threads(hidden, elem_per_vec):
    if hidden % elem_per_vec != 0:
        return 0, 128, 4
    vec_n = hidden // elem_per_vec
    threads = 32
    while threads < min(vec_n, 256):
        threads *= 2
    return vec_n, threads, threads // 32


@functools.lru_cache(maxsize=256)
def _block_rows_source_scalar(hidden, offset_literal, scale_literal):
    # Formatted once per configuration: the source is a pure function
    # of these, and a BERT-base forward built it 25 times.
    vec2_n, vec2_t, vec2_w = _ln_vec_threads(hidden, 8)
    vec4_n, vec4_t, vec4_w = _ln_vec_threads(hidden, 4)
    return f"""
            __device__ __forceinline__ float warp_sum(float value) {{
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += __shfl_down_sync(0xffffffff, value, offset);
                return value;
            }}
            __device__ __forceinline__ double warp_sum_double(double value) {{
                for (int offset = 16; offset > 0; offset >>= 1)
                    value += __shfl_down_sync(0xffffffff, value, offset);
                return value;
            }}
            __global__ __launch_bounds__({vec2_t}) static void kernel_vec2b(
                    const in0_type* __restrict__ x, out0_type* __restrict__ y, double eps) {{
                if (sizeof(in0_type) == 2 && sizeof(out0_type) == 2 && {vec2_n} > 0) {{
                    constexpr int kVecN = {vec2_n};
                    constexpr int kThreads = {vec2_t};
                    constexpr int kWarps = {vec2_w};
                    constexpr int kVecPer = (kVecN + kThreads - 1) / kThreads;
                    constexpr bool kVecCache = kVecPer <= 4;
                    long long row = blockIdx.x;
                    int tid = threadIdx.x;
                    int lane = tid & 31;
                    int warp = tid >> 5;
                    __shared__ float warp_buf[kWarps];
                    __shared__ float mean_shared, inv_std_shared;
                    __shared__ double warp_double_buf[kWarps];
                    __shared__ double mean_double_shared, inv_std_double_shared;
                    __shared__ int use_double;
                    const uint4* x8 = reinterpret_cast<const uint4*>(x + row * {hidden}LL);
                    uint4* y8 = reinterpret_cast<uint4*>(y + row * {hidden}LL);
                    uint4 cache[kVecCache ? (kVecPer > 0 ? kVecPer : 1) : 1];
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
                    sum = warp_sum(sum);
                    if (lane == 0) warp_buf[warp] = sum;
                    __syncthreads();
                    if (warp == 0) {{
                        float total = lane < kWarps ? warp_buf[lane] : 0.0f;
                        total = warp_sum(total);
                        if (lane == 0) {{
                            use_double = !isfinite(total);
                            if (!use_double) mean_shared = total / {hidden}.0f;
                        }}
                    }}
                    __syncthreads();
                    if (!use_double) {{
                        float mean = mean_shared;
                        float var = 0.0f;
                        if (kVecCache) {{
                            #pragma unroll
                            for (int i = 0; i < kVecPer; ++i) {{
                                int v = tid + i * kThreads;
                                if (v < kVecN) {{
                                    uint4 raw = cache[i];
                                    const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
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
                        var = warp_sum(var);
                        if (lane == 0) warp_buf[warp] = var;
                        __syncthreads();
                        if (warp == 0) {{
                            float total = lane < kWarps ? warp_buf[lane] : 0.0f;
                            total = warp_sum(total);
                            if (lane == 0) {{
                                use_double = !isfinite(total);
                                if (!use_double)
                                    inv_std_shared = rsqrtf(total / {hidden}.0f + static_cast<float>(eps));
                            }}
                        }}
                        __syncthreads();
                    }}
                    if (use_double) {{
                        double double_sum = 0.0;
                        for (int j = tid; j < {hidden}; j += kThreads)
                            double_sum += static_cast<double>(x[row * {hidden}LL + j]);
                        double_sum = warp_sum_double(double_sum);
                        if (lane == 0) warp_double_buf[warp] = double_sum;
                        __syncthreads();
                        if (warp == 0) {{
                            double total = lane < kWarps ? warp_double_buf[lane] : 0.0;
                            total = warp_sum_double(total);
                            if (lane == 0) mean_double_shared = total / {hidden}.0;
                        }}
                        __syncthreads();
                        double dm = mean_double_shared, dvar = 0.0;
                        for (int j = tid; j < {hidden}; j += kThreads) {{
                            double d = static_cast<double>(x[row * {hidden}LL + j]) - dm;
                            dvar += d * d;
                        }}
                        dvar = warp_sum_double(dvar);
                        if (lane == 0) warp_double_buf[warp] = dvar;
                        __syncthreads();
                        if (warp == 0) {{
                            double total = lane < kWarps ? warp_double_buf[lane] : 0.0;
                            total = warp_sum_double(total);
                            if (lane == 0) inv_std_double_shared = 1.0 / sqrt(total / {hidden}.0 + eps);
                        }}
                        __syncthreads();
                        double dinv = inv_std_double_shared;
                        for (int j = tid; j < {hidden}; j += kThreads)
                            y[row * {hidden}LL + j] = out0_type(
                                (static_cast<double>(x[row * {hidden}LL + j]) - dm)
                                * dinv * {scale_literal} + {offset_literal});
                    }} else {{
                        float mean = mean_shared, inv_std = inv_std_shared;
                        if (kVecCache) {{
                            #pragma unroll
                            for (int i = 0; i < kVecPer; ++i) {{
                                int v = tid + i * kThreads;
                                if (v < kVecN) {{
                                    uint4 raw = cache[i];
                                    const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                                    uint4 outraw;
                                    out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                                    #pragma unroll
                                    for (int k = 0; k < 8; ++k)
                                        op[k] = out0_type((static_cast<float>(vp[k]) - mean) * inv_std * {scale_literal} + {offset_literal});
                                    y8[v] = outraw;
                                }}
                            }}
                        }} else {{
                            for (int v = tid; v < kVecN; v += kThreads) {{
                                uint4 raw = x8[v];
                                const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                                uint4 outraw;
                                out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                                #pragma unroll
                                for (int k = 0; k < 8; ++k)
                                    op[k] = out0_type((static_cast<float>(vp[k]) - mean) * inv_std * {scale_literal} + {offset_literal});
                                y8[v] = outraw;
                            }}
                        }}
                    }}
                }}
            }}
            __global__ __launch_bounds__({vec4_t}) static void kernel_vec4b(
                    const in0_type* __restrict__ x, out0_type* __restrict__ y, double eps) {{
                if (sizeof(in0_type) == 4 && sizeof(out0_type) == 4 && {vec4_n} > 0) {{
                    constexpr int kVecN = {vec4_n};
                    constexpr int kThreads = {vec4_t};
                    constexpr int kWarps = {vec4_w};
                    constexpr int kVecPer = (kVecN + kThreads - 1) / kThreads;
                    constexpr bool kVecCache = kVecPer <= 4;
                    long long row = blockIdx.x;
                    int tid = threadIdx.x;
                    int lane = tid & 31;
                    int warp = tid >> 5;
                    __shared__ float warp_buf[kWarps];
                    __shared__ float mean_shared, inv_std_shared;
                    __shared__ double warp_double_buf[kWarps];
                    __shared__ double mean_double_shared, inv_std_double_shared;
                    __shared__ int use_double;
                    const float4* x4 = reinterpret_cast<const float4*>(x + row * {hidden}LL);
                    float4* y4 = reinterpret_cast<float4*>(y + row * {hidden}LL);
                    float4 cache[kVecCache ? (kVecPer > 0 ? kVecPer : 1) : 1];
                    float sum = 0.0f;
                    if (kVecCache) {{
                        #pragma unroll
                        for (int i = 0; i < kVecPer; ++i) {{
                            int v = tid + i * kThreads;
                            if (v < kVecN) {{
                                float4 val = x4[v];
                                cache[i] = val;
                                sum += val.x + val.y + val.z + val.w;
                            }}
                        }}
                    }} else {{
                        for (int v = tid; v < kVecN; v += kThreads) {{
                            float4 val = x4[v];
                            sum += val.x + val.y + val.z + val.w;
                        }}
                    }}
                    sum = warp_sum(sum);
                    if (lane == 0) warp_buf[warp] = sum;
                    __syncthreads();
                    if (warp == 0) {{
                        float total = lane < kWarps ? warp_buf[lane] : 0.0f;
                        total = warp_sum(total);
                        if (lane == 0) {{
                            use_double = !isfinite(total);
                            if (!use_double) mean_shared = total / {hidden}.0f;
                        }}
                    }}
                    __syncthreads();
                    if (!use_double) {{
                        float mean = mean_shared;
                        float var = 0.0f;
                        if (kVecCache) {{
                            #pragma unroll
                            for (int i = 0; i < kVecPer; ++i) {{
                                int v = tid + i * kThreads;
                                if (v < kVecN) {{
                                    float4 val = cache[i];
                                    float dx = val.x - mean, dy = val.y - mean, dz = val.z - mean, dw = val.w - mean;
                                    var += dx * dx + dy * dy + dz * dz + dw * dw;
                                }}
                            }}
                        }} else {{
                            for (int v = tid; v < kVecN; v += kThreads) {{
                                float4 val = x4[v];
                                float dx = val.x - mean, dy = val.y - mean, dz = val.z - mean, dw = val.w - mean;
                                var += dx * dx + dy * dy + dz * dz + dw * dw;
                            }}
                        }}
                        var = warp_sum(var);
                        if (lane == 0) warp_buf[warp] = var;
                        __syncthreads();
                        if (warp == 0) {{
                            float total = lane < kWarps ? warp_buf[lane] : 0.0f;
                            total = warp_sum(total);
                            if (lane == 0) {{
                                use_double = !isfinite(total);
                                if (!use_double)
                                    inv_std_shared = rsqrtf(total / {hidden}.0f + static_cast<float>(eps));
                            }}
                        }}
                        __syncthreads();
                    }}
                    if (use_double) {{
                        double double_sum = 0.0;
                        for (int j = tid; j < {hidden}; j += kThreads)
                            double_sum += static_cast<double>(x[row * {hidden}LL + j]);
                        double_sum = warp_sum_double(double_sum);
                        if (lane == 0) warp_double_buf[warp] = double_sum;
                        __syncthreads();
                        if (warp == 0) {{
                            double total = lane < kWarps ? warp_double_buf[lane] : 0.0;
                            total = warp_sum_double(total);
                            if (lane == 0) mean_double_shared = total / {hidden}.0;
                        }}
                        __syncthreads();
                        double dm = mean_double_shared, dvar = 0.0;
                        for (int j = tid; j < {hidden}; j += kThreads) {{
                            double d = static_cast<double>(x[row * {hidden}LL + j]) - dm;
                            dvar += d * d;
                        }}
                        dvar = warp_sum_double(dvar);
                        if (lane == 0) warp_double_buf[warp] = dvar;
                        __syncthreads();
                        if (warp == 0) {{
                            double total = lane < kWarps ? warp_double_buf[lane] : 0.0;
                            total = warp_sum_double(total);
                            if (lane == 0) inv_std_double_shared = 1.0 / sqrt(total / {hidden}.0 + eps);
                        }}
                        __syncthreads();
                        double dinv = inv_std_double_shared;
                        for (int j = tid; j < {hidden}; j += kThreads)
                            y[row * {hidden}LL + j] = out0_type(
                                (static_cast<double>(x[row * {hidden}LL + j]) - dm)
                                * dinv * {scale_literal} + {offset_literal});
                    }} else {{
                        float mean = mean_shared, inv_std = inv_std_shared;
                        if (kVecCache) {{
                            #pragma unroll
                            for (int i = 0; i < kVecPer; ++i) {{
                                int v = tid + i * kThreads;
                                if (v < kVecN) {{
                                    float4 val = cache[i];
                                    float4 outv;
                                    outv.x = (val.x - mean) * inv_std * {scale_literal} + {offset_literal};
                                    outv.y = (val.y - mean) * inv_std * {scale_literal} + {offset_literal};
                                    outv.z = (val.z - mean) * inv_std * {scale_literal} + {offset_literal};
                                    outv.w = (val.w - mean) * inv_std * {scale_literal} + {offset_literal};
                                    y4[v] = outv;
                                }}
                            }}
                        }} else {{
                            for (int v = tid; v < kVecN; v += kThreads) {{
                                float4 val = x4[v];
                                float4 outv;
                                outv.x = (val.x - mean) * inv_std * {scale_literal} + {offset_literal};
                                outv.y = (val.y - mean) * inv_std * {scale_literal} + {offset_literal};
                                outv.z = (val.z - mean) * inv_std * {scale_literal} + {offset_literal};
                                outv.w = (val.w - mean) * inv_std * {scale_literal} + {offset_literal};
                                y4[v] = outv;
                            }}
                        }}
                    }}
                }}
            }}
            __global__ static void kernel(
                    in0_type* x, out0_type* y, int hidden, double eps) {{
                int row = blockIdx.x;
                int tid = threadIdx.x;
                int lane = tid & 31;
                int warp = tid >> 5;
                __shared__ float warp_buf[4];
                __shared__ float mean_shared;
                __shared__ float inv_std_shared;
                __shared__ double warp_double_buf[4];
                __shared__ double mean_double_shared;
                __shared__ double inv_std_double_shared;
                __shared__ int use_double;
                // Row kept in registers; see the affine kernel below.
                constexpr int kPer = ({hidden} + 127) / 128;
                constexpr bool kCache = kPer <= 8;
                float cache[kCache ? kPer : 1];
                float sum = 0.0f;
                if (kCache) {{
                    int i = 0;
                    for (int j = tid; j < hidden; j += blockDim.x, ++i) {{
                        cache[i] = static_cast<float>(x[row * hidden + j]);
                        sum += cache[i];
                    }}
                }} else {{
                    for (int j = tid; j < hidden; j += blockDim.x)
                        sum += static_cast<float>(x[row * hidden + j]);
                }}
                sum = warp_sum(sum);
                if (lane == 0) warp_buf[warp] = sum;
                __syncthreads();
                if (warp == 0) {{
                    float total = lane < 4 ? warp_buf[lane] : 0.0f;
                    total = warp_sum(total);
                    if (lane == 0) {{
                        use_double = !isfinite(total);
                        if (!use_double) mean_shared = total / hidden;
                    }}
                }}
                __syncthreads();
                if (!use_double) {{
                    float mean = mean_shared;
                    float var = 0.0f;
                    if (kCache) {{
                        int i = 0;
                        for (int j = tid; j < hidden; j += blockDim.x, ++i) {{
                            float d = cache[i] - mean;
                            var += d * d;
                        }}
                    }} else {{
                        for (int j = tid; j < hidden; j += blockDim.x) {{
                            float d = static_cast<float>(
                                x[row * hidden + j]) - mean;
                            var += d * d;
                        }}
                    }}
                    var = warp_sum(var);
                    if (lane == 0) warp_buf[warp] = var;
                    __syncthreads();
                    if (warp == 0) {{
                        float total = lane < 4 ? warp_buf[lane] : 0.0f;
                        total = warp_sum(total);
                        if (lane == 0) {{
                            use_double = !isfinite(total);
                            if (!use_double)
                                inv_std_shared = rsqrtf(
                                    total / hidden + static_cast<float>(eps));
                        }}
                    }}
                    __syncthreads();
                }}
                if (use_double) {{
                    double double_sum = 0.0;
                    for (int j = tid; j < hidden; j += blockDim.x)
                        double_sum += static_cast<double>(
                            x[row * hidden + j]);
                    double_sum = warp_sum_double(double_sum);
                    if (lane == 0) warp_double_buf[warp] = double_sum;
                    __syncthreads();
                    if (warp == 0) {{
                        double total = lane < 4 ? warp_double_buf[lane] : 0.0;
                        total = warp_sum_double(total);
                        if (lane == 0)
                            mean_double_shared = total / hidden;
                    }}
                    __syncthreads();
                    double double_mean = mean_double_shared;
                    double double_var = 0.0;
                    for (int j = tid; j < hidden; j += blockDim.x) {{
                        double d = static_cast<double>(
                            x[row * hidden + j]) - double_mean;
                        double_var += d * d;
                    }}
                    double_var = warp_sum_double(double_var);
                    if (lane == 0) warp_double_buf[warp] = double_var;
                    __syncthreads();
                    if (warp == 0) {{
                        double total = lane < 4 ? warp_double_buf[lane] : 0.0;
                        total = warp_sum_double(total);
                        if (lane == 0)
                            inv_std_double_shared = 1.0 / sqrt(
                                total / hidden + eps);
                    }}
                    __syncthreads();
                    double double_inv_std = inv_std_double_shared;
                    for (int j = tid; j < hidden; j += blockDim.x)
                        y[row * hidden + j] = out0_type(
                            (static_cast<double>(x[row * hidden + j])
                             - double_mean)
                            * double_inv_std * {scale_literal}
                            + {offset_literal});
                }} else {{
                    float mean = mean_shared;
                    float inv_std = inv_std_shared;
                    int i = 0;
                    for (int j = tid; j < hidden; j += blockDim.x, ++i) {{
                        float xv = kCache ? cache[i]
                                          : static_cast<float>(x[row * hidden + j]);
                        y[row * hidden + j] = out0_type(
                            (xv - mean) * inv_std * {scale_literal}
                            + {offset_literal});
                    }}
                }}
            }}
            double eps = data["eps"];
            long long rows = in0->num / {hidden};
            if (sizeof(in0_type) == 2 && sizeof(out0_type) == 2 && {vec2_n} > 0
                    && ((((size_t)in0_p | (size_t)out0_p) & 15u) == 0)) {{
                kernel_vec2b<<<(unsigned)rows, {vec2_t}>>>(in0_p, out0_p, eps);
            }} else if (sizeof(in0_type) == 4 && sizeof(out0_type) == 4 && {vec4_n} > 0
                    && ((((size_t)in0_p | (size_t)out0_p) & 15u) == 0)) {{
                kernel_vec4b<<<(unsigned)rows, {vec4_t}>>>(in0_p, out0_p, eps);
            }} else {{
                kernel<<<(unsigned)rows, 128>>>(in0_p, out0_p, {hidden}, eps);
            }}
            """


@functools.lru_cache(maxsize=256)
def _block_rows_source(hidden, has_bias=True, offset_literal="0.0f", offset_value=0.0):
    # Formatted once per configuration: the source is a pure function
    # of these, and a BERT-base forward built it 25 times.
    bias_param = ", in2_type* bias" if has_bias else ""
    bias_arg = "in2_p, " if has_bias else ""
    offset_d = "static_cast<double>(bias[j])" if has_bias else repr(offset_value)
    offset_f = "static_cast<float>(bias[j])" if has_bias else offset_literal
    vec2_n, vec2_t, vec2_w = _ln_vec_threads(hidden, 8)
    vec4_n, vec4_t, vec4_w = _ln_vec_threads(hidden, 4)
    bias_vec2_load = (
        "bool b_vec = (sizeof(in2_type) == 2) && ((((size_t)bias) & 15u) == 0);"
        " const uint4* b8 = b_vec ? reinterpret_cast<const uint4*>(bias) : nullptr;"
        if has_bias else ""
    )
    bias_vec2_val = (
        "uint4 braw = b_vec ? b8[v] : make_uint4(0,0,0,0);"
        " const in2_type* bp = reinterpret_cast<const in2_type*>(&braw);"
        if has_bias else ""
    )
    bias_vec2_k = (
        "(b_vec ? static_cast<float>(bp[k]) : static_cast<float>(bias[v * 8 + k]))"
        if has_bias else offset_literal
    )
    bias_vec4_load = (
        "bool b_vec = (sizeof(in2_type) == 4) && ((((size_t)bias) & 15u) == 0);"
        " const float4* b4 = b_vec ? reinterpret_cast<const float4*>(bias) : nullptr;"
        if has_bias else ""
    )
    bias_vec4_val = (
        "float4 bv = b_vec ? b4[v] : make_float4("
        "static_cast<float>(bias[v*4]), static_cast<float>(bias[v*4+1]),"
        " static_cast<float>(bias[v*4+2]), static_cast<float>(bias[v*4+3]));"
        if has_bias else f"float4 bv = make_float4({offset_literal}, {offset_literal}, {offset_literal}, {offset_literal});"
    )
    reductions = "".join(f"""
        __device__ __forceinline__ {dtype} warp_sum_{dtype}({dtype} value) {{
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            return value;
        }}
        __device__ __forceinline__ {dtype} block_sum(
                {dtype} value, {dtype}* scratch, int lane, int warp) {{
            value = warp_sum_{dtype}(value);
            if (lane == 0) scratch[warp] = value;
            __syncthreads();
            return (scratch[0] + scratch[2]) + (scratch[1] + scratch[3]);
        }}
        __device__ __forceinline__ {dtype} block_sum_n(
                {dtype} value, {dtype}* scratch, int lane, int warp, int kWarps) {{
            value = warp_sum_{dtype}(value);
            if (kWarps == 1) {{
                return __shfl_sync(0xffffffff, value, 0);
            }}
            if (lane == 0) scratch[warp] = value;
            __syncthreads();
            if (warp == 0) {{
                {dtype} total = lane < kWarps ? scratch[lane] : ({dtype})0;
                total = warp_sum_{dtype}(total);
                if (lane == 0) scratch[0] = total;
            }}
            __syncthreads();
            return scratch[0];
        }}
    """ for dtype in ("float", "double"))
    return f"""
        {reductions}
        __global__ __launch_bounds__({vec2_t}) static void kernel_vec2b(
                const in0_type* __restrict__ x, const in1_type* __restrict__ weight{bias_param},
                out0_type* __restrict__ y, double eps) {{
            if (sizeof(in0_type) == 2 && sizeof(out0_type) == 2 && {vec2_n} > 0) {{
                constexpr int kVecN = {vec2_n};
                constexpr int kThreads = {vec2_t};
                constexpr int kWarps = {vec2_w};
                constexpr int kVecPer = (kVecN + kThreads - 1) / kThreads;
                constexpr bool kVecCache = kVecPer <= 4;
                long long row = blockIdx.x;
                int tid = threadIdx.x;
                int lane = tid & 31;
                int warp = tid >> 5;
                __shared__ float sum_scratch[kWarps], variance_scratch[kWarps];
                __shared__ double double_sum_scratch[kWarps], double_variance_scratch[kWarps];
                const uint4* x8 = reinterpret_cast<const uint4*>(x + row * {hidden}LL);
                uint4* y8 = reinterpret_cast<uint4*>(y + row * {hidden}LL);
                uint4 cache[kVecCache ? (kVecPer > 0 ? kVecPer : 1) : 1];
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
                sum = block_sum_n(sum, sum_scratch, lane, warp, kWarps);
                float mean = sum / {hidden}.0f;
                float inv_std = 0.0f;
                bool use_double = !isfinite(sum);
                if (!use_double) {{
                    float var = 0.0f;
                    if (kVecCache) {{
                        #pragma unroll
                        for (int i = 0; i < kVecPer; ++i) {{
                            int v = tid + i * kThreads;
                            if (v < kVecN) {{
                                uint4 raw = cache[i];
                                const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
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
                    var = block_sum_n(var, variance_scratch, lane, warp, kWarps);
                    use_double = !isfinite(var);
                    inv_std = rsqrtf(var / {hidden}.0f + static_cast<float>(eps));
                }}
                if (use_double) {{
                    double dsum = 0.0;
                    for (int j = tid; j < {hidden}; j += kThreads)
                        dsum += static_cast<double>(x[row * {hidden}LL + j]);
                    double dmean = block_sum_n(dsum, double_sum_scratch, lane, warp, kWarps) / {hidden}.0;
                    double dvar = 0.0;
                    for (int j = tid; j < {hidden}; j += kThreads) {{
                        double d = static_cast<double>(x[row * {hidden}LL + j]) - dmean;
                        dvar += d * d;
                    }}
                    dvar = block_sum_n(dvar, double_variance_scratch, lane, warp, kWarps);
                    double dinv = 1.0 / sqrt(dvar / {hidden}.0 + eps);
                    for (int j = tid; j < {hidden}; j += kThreads) {{
                        double scale = static_cast<double>(weight[j]);
                        double offset = {offset_d};
                        y[row * {hidden}LL + j] = out0_type(
                            (static_cast<double>(x[row * {hidden}LL + j]) - dmean) * dinv * scale + offset);
                    }}
                }} else {{
                    bool w_vec = (sizeof(in1_type) == 2) && ((((size_t)weight) & 15u) == 0);
                    const uint4* w8 = w_vec ? reinterpret_cast<const uint4*>(weight) : nullptr;
                    {bias_vec2_load}
                    if (kVecCache) {{
                        #pragma unroll
                        for (int i = 0; i < kVecPer; ++i) {{
                            int v = tid + i * kThreads;
                            if (v < kVecN) {{
                                uint4 raw = cache[i];
                                const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                                uint4 wraw = w_vec ? w8[v] : make_uint4(0,0,0,0);
                                const in1_type* wp = reinterpret_cast<const in1_type*>(&wraw);
                                {bias_vec2_val}
                                uint4 outraw;
                                out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                                #pragma unroll
                                for (int k = 0; k < 8; ++k) {{
                                    float sc = w_vec ? static_cast<float>(wp[k]) : static_cast<float>(weight[v * 8 + k]);
                                    float of = {bias_vec2_k};
                                    op[k] = out0_type((static_cast<float>(vp[k]) - mean) * inv_std * sc + of);
                                }}
                                y8[v] = outraw;
                            }}
                        }}
                    }} else {{
                        for (int v = tid; v < kVecN; v += kThreads) {{
                            uint4 raw = x8[v];
                            const in0_type* vp = reinterpret_cast<const in0_type*>(&raw);
                            uint4 wraw = w_vec ? w8[v] : make_uint4(0,0,0,0);
                            const in1_type* wp = reinterpret_cast<const in1_type*>(&wraw);
                            {bias_vec2_val}
                            uint4 outraw;
                            out0_type* op = reinterpret_cast<out0_type*>(&outraw);
                            #pragma unroll
                            for (int k = 0; k < 8; ++k) {{
                                float sc = w_vec ? static_cast<float>(wp[k]) : static_cast<float>(weight[v * 8 + k]);
                                float of = {bias_vec2_k};
                                op[k] = out0_type((static_cast<float>(vp[k]) - mean) * inv_std * sc + of);
                            }}
                            y8[v] = outraw;
                        }}
                    }}
                }}
            }}
        }}
        __global__ __launch_bounds__({vec4_t}) static void kernel_vec4b(
                const in0_type* __restrict__ x, const in1_type* __restrict__ weight{bias_param},
                out0_type* __restrict__ y, double eps) {{
            if (sizeof(in0_type) == 4 && sizeof(out0_type) == 4 && {vec4_n} > 0) {{
                constexpr int kVecN = {vec4_n};
                constexpr int kThreads = {vec4_t};
                constexpr int kWarps = {vec4_w};
                constexpr int kVecPer = (kVecN + kThreads - 1) / kThreads;
                constexpr bool kVecCache = kVecPer <= 4;
                long long row = blockIdx.x;
                int tid = threadIdx.x;
                int lane = tid & 31;
                int warp = tid >> 5;
                __shared__ float sum_scratch[kWarps], variance_scratch[kWarps];
                __shared__ double double_sum_scratch[kWarps], double_variance_scratch[kWarps];
                const float4* x4 = reinterpret_cast<const float4*>(x + row * {hidden}LL);
                float4* y4 = reinterpret_cast<float4*>(y + row * {hidden}LL);
                float4 cache[kVecCache ? (kVecPer > 0 ? kVecPer : 1) : 1];
                float sum = 0.0f;
                if (kVecCache) {{
                    #pragma unroll
                    for (int i = 0; i < kVecPer; ++i) {{
                        int v = tid + i * kThreads;
                        if (v < kVecN) {{
                            float4 val = x4[v];
                            cache[i] = val;
                            sum += val.x + val.y + val.z + val.w;
                        }}
                    }}
                }} else {{
                    for (int v = tid; v < kVecN; v += kThreads) {{
                        float4 val = x4[v];
                        sum += val.x + val.y + val.z + val.w;
                    }}
                }}
                sum = block_sum_n(sum, sum_scratch, lane, warp, kWarps);
                float mean = sum / {hidden}.0f;
                float inv_std = 0.0f;
                bool use_double = !isfinite(sum);
                if (!use_double) {{
                    float var = 0.0f;
                    if (kVecCache) {{
                        #pragma unroll
                        for (int i = 0; i < kVecPer; ++i) {{
                            int v = tid + i * kThreads;
                            if (v < kVecN) {{
                                float4 val = cache[i];
                                float dx = val.x - mean, dy = val.y - mean, dz = val.z - mean, dw = val.w - mean;
                                var += dx * dx + dy * dy + dz * dz + dw * dw;
                            }}
                        }}
                    }} else {{
                        for (int v = tid; v < kVecN; v += kThreads) {{
                            float4 val = x4[v];
                            float dx = val.x - mean, dy = val.y - mean, dz = val.z - mean, dw = val.w - mean;
                            var += dx * dx + dy * dy + dz * dz + dw * dw;
                        }}
                    }}
                    var = block_sum_n(var, variance_scratch, lane, warp, kWarps);
                    use_double = !isfinite(var);
                    inv_std = rsqrtf(var / {hidden}.0f + static_cast<float>(eps));
                }}
                if (use_double) {{
                    double dsum = 0.0;
                    for (int j = tid; j < {hidden}; j += kThreads)
                        dsum += static_cast<double>(x[row * {hidden}LL + j]);
                    double dmean = block_sum_n(dsum, double_sum_scratch, lane, warp, kWarps) / {hidden}.0;
                    double dvar = 0.0;
                    for (int j = tid; j < {hidden}; j += kThreads) {{
                        double d = static_cast<double>(x[row * {hidden}LL + j]) - dmean;
                        dvar += d * d;
                    }}
                    dvar = block_sum_n(dvar, double_variance_scratch, lane, warp, kWarps);
                    double dinv = 1.0 / sqrt(dvar / {hidden}.0 + eps);
                    for (int j = tid; j < {hidden}; j += kThreads) {{
                        double scale = static_cast<double>(weight[j]);
                        double offset = {offset_d};
                        y[row * {hidden}LL + j] = out0_type(
                            (static_cast<double>(x[row * {hidden}LL + j]) - dmean) * dinv * scale + offset);
                    }}
                }} else {{
                    bool w_vec = (sizeof(in1_type) == 4) && ((((size_t)weight) & 15u) == 0);
                    const float4* w4 = w_vec ? reinterpret_cast<const float4*>(weight) : nullptr;
                    {bias_vec4_load}
                    if (kVecCache) {{
                        #pragma unroll
                        for (int i = 0; i < kVecPer; ++i) {{
                            int v = tid + i * kThreads;
                            if (v < kVecN) {{
                                float4 val = cache[i];
                                float4 wv = w_vec ? w4[v] : make_float4(
                                    static_cast<float>(weight[v*4]), static_cast<float>(weight[v*4+1]),
                                    static_cast<float>(weight[v*4+2]), static_cast<float>(weight[v*4+3]));
                                {bias_vec4_val}
                                float4 outv;
                                outv.x = (val.x - mean) * inv_std * wv.x + bv.x;
                                outv.y = (val.y - mean) * inv_std * wv.y + bv.y;
                                outv.z = (val.z - mean) * inv_std * wv.z + bv.z;
                                outv.w = (val.w - mean) * inv_std * wv.w + bv.w;
                                y4[v] = outv;
                            }}
                        }}
                    }} else {{
                        for (int v = tid; v < kVecN; v += kThreads) {{
                            float4 val = x4[v];
                            float4 wv = w_vec ? w4[v] : make_float4(
                                static_cast<float>(weight[v*4]), static_cast<float>(weight[v*4+1]),
                                static_cast<float>(weight[v*4+2]), static_cast<float>(weight[v*4+3]));
                            {bias_vec4_val}
                            float4 outv;
                            outv.x = (val.x - mean) * inv_std * wv.x + bv.x;
                            outv.y = (val.y - mean) * inv_std * wv.y + bv.y;
                            outv.z = (val.z - mean) * inv_std * wv.z + bv.z;
                            outv.w = (val.w - mean) * inv_std * wv.w + bv.w;
                            y4[v] = outv;
                        }}
                    }}
                }}
            }}
        }}
        __global__ static void kernel(
                in0_type* x, in1_type* weight{bias_param},
                out0_type* y, double eps) {{
            constexpr int hidden = {hidden};
            long long row = blockIdx.x;
            int tid = threadIdx.x;
            int lane = tid & 31;
            int warp = tid >> 5;
            __shared__ float sum_scratch[4], variance_scratch[4];
            __shared__ double double_sum_scratch[4], double_variance_scratch[4];
            // The row is read once and kept in registers: the mean pass, the
            // variance pass and the write all want the same values, and
            // re-reading them made this kernel move 4x the row where torch's
            // Welford moves 2x -- measured 44.1 us against its 29.7 at
            // b8 s256 d512, which is that ratio. Capped at 8 values per thread
            // so a wide row falls back to re-reading rather than spilling.
            constexpr int kPer = ({hidden} + 127) / 128;
            constexpr bool kCache = kPer <= 8;
            float cache[kCache ? kPer : 1];
            float sum = 0.0f;
            if (kCache) {{
                #pragma unroll
                for (int part = 0; part < kPer; ++part) {{
                    int column = tid + part * 128;
                    if (column < hidden) {{
                        cache[part] = static_cast<float>(x[row * hidden + column]);
                        sum += cache[part];
                    }}
                }}
            }} else {{
                for (int j = tid; j < hidden; j += 128)
                    sum += static_cast<float>(x[row * hidden + j]);
            }}
            sum = block_sum(sum, sum_scratch, lane, warp);
            float mean = sum / hidden;
            float inv_std = 0.0f;
            bool use_double = !isfinite(sum);
            if (!use_double) {{
                float var = 0.0f;
                if (kCache) {{
                    #pragma unroll
                    for (int part = 0; part < kPer; ++part) {{
                        if (tid + part * 128 < hidden) {{
                            float d = cache[part] - mean;
                            var += d * d;
                        }}
                    }}
                }} else {{
                    for (int j = tid; j < hidden; j += 128) {{
                        float d = static_cast<float>(
                            x[row * hidden + j]) - mean;
                        var += d * d;
                    }}
                }}
                var = block_sum(var, variance_scratch, lane, warp);
                use_double = !isfinite(var);
                inv_std = rsqrtf(var / hidden + static_cast<float>(eps));
            }}
            if (use_double) {{
                double double_sum = 0.0;
                for (int j = tid; j < hidden; j += 128)
                    double_sum += static_cast<double>(
                        x[row * hidden + j]);
                double double_mean = block_sum(double_sum, double_sum_scratch, lane, warp) / hidden;
                double double_var = 0.0;
                for (int j = tid; j < hidden; j += 128) {{
                    double d = static_cast<double>(
                        x[row * hidden + j]) - double_mean;
                    double_var += d * d;
                }}
                double_var = block_sum(double_var, double_variance_scratch, lane, warp);
                double double_inv_std = 1.0 / sqrt(double_var / hidden + eps);
                for (int j = tid; j < hidden; j += 128) {{
                    double scale = static_cast<double>(weight[j]);
                    double offset = {offset_d};
                    y[row * hidden + j] = out0_type(
                        (static_cast<double>(x[row * hidden + j])
                         - double_mean)
                        * double_inv_std * scale + offset);
                }}
            }} else {{
                const int parts = kCache ? kPer : (hidden - tid + 127) / 128;
                #pragma unroll
                for (int part = 0; part < parts; ++part) {{
                    int j = tid + part * 128;
                    if (j >= hidden) break;
                    float scale = static_cast<float>(weight[j]);
                    float offset = {offset_f};
                    float xv = kCache ? cache[part]
                                      : static_cast<float>(x[row * hidden + j]);
                    y[row * hidden + j] = out0_type(
                        (xv - mean) * inv_std * scale + offset);
                }}
            }}
        }}
        double eps = data["eps"];
        long long rows = in0->num / {hidden};
        if (sizeof(in0_type) == 2 && sizeof(out0_type) == 2 && {vec2_n} > 0
                && ((((size_t)in0_p | (size_t)out0_p) & 15u) == 0)) {{
            kernel_vec2b<<<(unsigned)rows, {vec2_t}>>>(
                in0_p, in1_p, {bias_arg}out0_p, eps);
        }} else if (sizeof(in0_type) == 4 && sizeof(out0_type) == 4 && {vec4_n} > 0
                && ((((size_t)in0_p | (size_t)out0_p) & 15u) == 0)) {{
            kernel_vec4b<<<(unsigned)rows, {vec4_t}>>>(
                in0_p, in1_p, {bias_arg}out0_p, eps);
        }} else {{
            kernel<<<(unsigned)rows, 128>>>(
                in0_p, in1_p, {bias_arg}out0_p, eps);
        }}
        """


def _nchw_physical(x):
    """Return the dense ``[N, C, H, W]`` source of a ``permute(0, 2, 3, 1)`` view, or None."""
    if not isinstance(x, jt.Var) or len(x.shape) != 4:
        return None
    src = None
    if tuple(x._transpose_view_axes()) == (0, 2, 3, 1):
        src = x._transpose_view_source()
    elif tuple(x._producer_transpose_axes()) == (0, 2, 3, 1):
        src = x._input(0)
    if not isinstance(src, jt.Var) or len(src.shape) != 4:
        return None
    if int(src.shape[1]) != int(x.shape[3]):
        return None
    return src


@functools.lru_cache(maxsize=256)
def _nchw_block_rows_source(hidden):
    """Fused NCHW -> NHWC affine LayerNorm using one 128-thread block per spatial row."""
    return f"""
        __device__ __forceinline__ float warp_sum(float value) {{
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            return value;
        }}
        __device__ __forceinline__ double warp_sum_double(double value) {{
            for (int offset = 16; offset > 0; offset >>= 1)
                value += __shfl_down_sync(0xffffffff, value, offset);
            return value;
        }}
        __global__ static void kernel_nchw_row(
                const in0_type* __restrict__ x,
                const in1_type* __restrict__ weight,
                const in2_type* __restrict__ bias,
                out0_type* __restrict__ y,
                long long spatial, double eps) {{
            long long row = blockIdx.x;
            long long n = row / spatial;
            long long hw = row - n * spatial;
            long long base = n * {hidden}LL * spatial + hw;
            int tid = threadIdx.x;
            int lane = tid & 31;
            int warp = tid >> 5;
            __shared__ float warp_buf[4];
            __shared__ float mean_shared;
            __shared__ float inv_std_shared;
            __shared__ double warp_double_buf[4];
            __shared__ double mean_double_shared;
            __shared__ double inv_std_double_shared;
            __shared__ int use_double;
            constexpr int kPer = ({hidden} + 127) / 128;
            constexpr bool kCache = kPer <= 8;
            float cache[kCache ? kPer : 1];
            float sum = 0.0f;
            if (kCache) {{
                int i = 0;
                for (int j = tid; j < {hidden}; j += blockDim.x, ++i) {{
                    cache[i] = static_cast<float>(x[base + (long long)j * spatial]);
                    sum += cache[i];
                }}
            }} else {{
                for (int j = tid; j < {hidden}; j += blockDim.x)
                    sum += static_cast<float>(x[base + (long long)j * spatial]);
            }}
            sum = warp_sum(sum);
            if (lane == 0) warp_buf[warp] = sum;
            __syncthreads();
            if (warp == 0) {{
                float total = lane < 4 ? warp_buf[lane] : 0.0f;
                total = warp_sum(total);
                if (lane == 0) {{
                    use_double = !isfinite(total);
                    if (!use_double) mean_shared = total / {hidden};
                }}
            }}
            __syncthreads();
            if (!use_double) {{
                float mean = mean_shared;
                float var = 0.0f;
                if (kCache) {{
                    int i = 0;
                    for (int j = tid; j < {hidden}; j += blockDim.x, ++i) {{
                        float d = cache[i] - mean;
                        var += d * d;
                    }}
                }} else {{
                    for (int j = tid; j < {hidden}; j += blockDim.x) {{
                        float d = static_cast<float>(x[base + (long long)j * spatial]) - mean;
                        var += d * d;
                    }}
                }}
                var = warp_sum(var);
                if (lane == 0) warp_buf[warp] = var;
                __syncthreads();
                if (warp == 0) {{
                    float total = lane < 4 ? warp_buf[lane] : 0.0f;
                    total = warp_sum(total);
                    if (lane == 0) {{
                        use_double = !isfinite(total);
                        if (!use_double)
                            inv_std_shared = rsqrtf(total / {hidden} + static_cast<float>(eps));
                    }}
                }}
                __syncthreads();
            }}
            if (use_double) {{
                double double_sum = 0.0;
                for (int j = tid; j < {hidden}; j += blockDim.x)
                    double_sum += static_cast<double>(x[base + (long long)j * spatial]);
                double_sum = warp_sum_double(double_sum);
                if (lane == 0) warp_double_buf[warp] = double_sum;
                __syncthreads();
                if (warp == 0) {{
                    double total = lane < 4 ? warp_double_buf[lane] : 0.0;
                    total = warp_sum_double(total);
                    if (lane == 0) mean_double_shared = total / {hidden};
                }}
                __syncthreads();
                double double_mean = mean_double_shared;
                double double_var = 0.0;
                for (int j = tid; j < {hidden}; j += blockDim.x) {{
                    double d = static_cast<double>(x[base + (long long)j * spatial]) - double_mean;
                    double_var += d * d;
                }}
                double_var = warp_sum_double(double_var);
                if (lane == 0) warp_double_buf[warp] = double_var;
                __syncthreads();
                if (warp == 0) {{
                    double total = lane < 4 ? warp_double_buf[lane] : 0.0;
                    total = warp_sum_double(total);
                    if (lane == 0) inv_std_double_shared = 1.0 / sqrt(total / {hidden} + eps);
                }}
                __syncthreads();
                double double_inv_std = inv_std_double_shared;
                for (int j = tid; j < {hidden}; j += blockDim.x) {{
                    double scale = static_cast<double>(weight[j]);
                    double offset = static_cast<double>(bias[j]);
                    y[row * {hidden} + j] = out0_type(
                        (static_cast<double>(x[base + (long long)j * spatial]) - double_mean)
                        * double_inv_std * scale + offset);
                }}
            }} else {{
                float mean = mean_shared;
                float inv_std = inv_std_shared;
                int i = 0;
                for (int j = tid; j < {hidden}; j += blockDim.x, ++i) {{
                    float scale = static_cast<float>(weight[j]);
                    float offset = static_cast<float>(bias[j]);
                    float xv = kCache ? cache[i] : static_cast<float>(x[base + (long long)j * spatial]);
                    y[row * {hidden} + j] = out0_type((xv - mean) * inv_std * scale + offset);
                }}
            }}
        }}
        double eps = data["eps"];
        long long spatial = (long long)in0->shape[2] * in0->shape[3];
        long long rows = (long long)in0->shape[0] * spatial;
        if (rows > 0)
            kernel_nchw_row<<<(unsigned)rows, 128>>>(in0_p, in1_p, in2_p, out0_p, spatial, eps);
    """


def _offer_narrow_layer_norm(y, x, normalized_shape, weight, bias, eps):
    if not jt.flags.no_grad or jt.flags.amp_reg or x._producer_unary() != "float32":
        return y
    inputs = x._inputs()
    if len(inputs) != 1 or _jittor_dtype_name(inputs[0].dtype) not in ("float16", "bfloat16"):
        return y
    narrow = inputs[0]
    operands = tuple(value for value in (narrow, weight, bias) if isinstance(value, jt.Var))
    versions = tuple(value._saved_storage_version() for value in operands)

    def rebuild(dtype):
        if (dtype != _jittor_dtype_name(narrow.dtype) or not jt.flags.no_grad or jt.flags.amp_reg
                or tuple(value._saved_storage_version() for value in operands) != versions):
            return None
        return _layer_norm_no_grad_cuda(narrow, normalized_shape, weight, bias, eps)

    y._jt_narrow_layer_norm = (y._saved_storage_version(), rebuild)
    return y


@optional_kernel("nn.layer_norm.inference", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("float16", "bfloat16", "float32"),
                 supports=_supports_layer_norm_inference)
def _layer_norm_no_grad_cuda(
        x, normalized_shape, weight, bias, eps, *, allow_bfloat16=True):
    hidden = int(normalized_shape[0])
    has_weight = isinstance(weight, jt.Var)
    has_bias = isinstance(bias, jt.Var)
    eps_value = float(eps)
    if not has_weight and not has_bias:
        scale_value = float(weight)
        offset_value = float(bias)
        scale_literal = f"{scale_value:.9e}f"
        offset_literal = f"{offset_value:.9e}f"
        if _warp_rows(x, hidden):
            affine = ("", f"* {scale_literal} + {offset_literal}",
                      f"* {scale_value!r} + {offset_value!r}")
            y = jt.code(
                x.shape, x.dtype, [x],
                cuda_src=_warp_rows_source(hidden, affine)
                + _warp_rows_launch(hidden, "in0_p, out0_p"),
                data={"eps": eps_value})
            if scale_value == 1.0 and offset_value == 0.0:
                y._jt_plain_ln = (x, eps_value)
            return y
        y = jt.code(
            x.shape,
            x.dtype,
            [x],
            cuda_src=_block_rows_source_scalar(hidden, offset_literal, scale_literal),
            data={"eps": eps_value},
        )
        if scale_value == 1.0 and offset_value == 0.0:
            y._jt_plain_ln = (x, eps_value)
        return y
    if has_weight and not has_bias:
        y = jt.code(x.shape, x.dtype, [x, weight],
                    cuda_src=_weight_only_source(hidden, _warp_rows(x, hidden), float(bias)),
                    data={"eps": eps_value})
        return _offer_narrow_layer_norm(
            _stop_grad_outputs(y), x, normalized_shape, weight, bias, eps)
    nchw_src = _nchw_physical(x)
    if nchw_src is not None:
        spatial = int(nchw_src.shape[2]) * int(nchw_src.shape[3])
        # On large FP32 feature maps the block-row kernel gathers each
        # channel from a different memory segment. A tiled contiguous copy
        # and the vectorized dense kernel make coalesced accesses in both
        # passes and consume less device time.
        prefer_dense = (
            spatial >= 32 and x.numel() >= (1 << 20)
            and _jittor_dtype_name(x.dtype) == "float32"
            and dispatch_context(x).backend == "cuda"
        )
        if spatial > 0 and not prefer_dense:
            y = jt.code(x.shape, x.dtype, [nchw_src, weight, bias],
                        cuda_src=_nchw_block_rows_source(hidden), data={"eps": eps_value})
            return _offer_narrow_layer_norm(
                _stop_grad_outputs(y), x, normalized_shape, weight, bias, eps)
    y = jt.code(x.shape, x.dtype, [x, weight, bias],
                cuda_src=_affine_source(hidden, _warp_rows(x, hidden)),
                data={"eps": eps_value})
    return _offer_narrow_layer_norm(
        _stop_grad_outputs(y), x, normalized_shape, weight, bias, eps)


def _weight_only_source(hidden, warp, offset_value=0.0):
    offset_literal = f"{offset_value:.9e}f"
    if warp:
        affine = ("in1_type* weight, ",
                  f"* static_cast<float>(weight[j]) + {offset_literal}",
                  f"* static_cast<double>(weight[j]) + {offset_value!r}")
        return (_warp_rows_source(hidden, affine)
                + _warp_rows_launch(hidden, "in0_p, in1_p, out0_p"))
    return _block_rows_source(hidden, has_bias=False,
                              offset_literal=offset_literal, offset_value=offset_value)


@functools.lru_cache(maxsize=256)
def _affine_source_cached(hidden, warp):
    if warp:
        affine = ("in1_type* weight, in2_type* bias, ",
                  "* static_cast<float>(weight[j]) + static_cast<float>(bias[j])",
                  "* static_cast<double>(weight[j]) + static_cast<double>(bias[j])")
        return (_warp_rows_source(hidden, affine)
                + _warp_rows_launch(hidden, "in0_p, in1_p, in2_p, out0_p"))
    return _block_rows_source(hidden)


def _affine_source(hidden, warp, _legacy_warp=None):
    """The kernel for a Var weight and bias: a warp a row, or a block.

    Also what the native module call builds a `LayerNorm`'s inference
    operator from (`src/bindings/pyjt/py_module_call.cc`).
    """
    if _legacy_warp is not None:
        warp = _legacy_warp
    return _affine_source_cached(int(hidden), bool(warp))
