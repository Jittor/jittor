"""CUDA inference fast path for :func:`jittor.nn.layer_norm`."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

import functools
import os

import jittor as jt
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor._runtime.dispatch import native_rule, optional_kernel


@native_rule("layer_norm_inference")
def _supports_layer_norm_inference(
        x, normalized_shape, weight, bias, eps, *, allow_bfloat16=False):
    if _output_requires_grad(x, weight, bias):
        return False
    input_dtype = _jittor_dtype_name(x.dtype)
    supported_dtypes = ("float16", "float32")
    if allow_bfloat16:
        supported_dtypes += ("bfloat16",)
    if len(normalized_shape) != 1 or _jittor_dtype_name(input_dtype) not in supported_dtypes:
        return False
    hidden = int(normalized_shape[0])
    var_affine = isinstance(weight, jt.Var) and isinstance(bias, jt.Var)
    scalar_affine = not isinstance(weight, jt.Var) and not isinstance(bias, jt.Var)
    if not var_affine:
        if not scalar_affine or os.environ.get("JITTOR_LAYERNORM_SCALAR_FAST", "1") == "0":
            return False
    if int(x.shape[-1]) != hidden:
        return False
    if var_affine and (int(weight.numel()) != hidden or int(bias.numel()) != hidden):
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


@functools.lru_cache(maxsize=256)
def _block_rows_source_scalar(hidden, offset_literal, scale_literal):
    # Formatted once per configuration: the source is a pure function
    # of these, and a BERT-base forward built it 25 times.
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
            int rows = in0->num / {hidden};
            kernel<<<rows, 128>>>(in0_p, out0_p, {hidden}, eps);
            """


@functools.lru_cache(maxsize=256)
def _block_rows_source(hidden):
    # Formatted once per configuration: the source is a pure function
    # of these, and a BERT-base forward built it 25 times.
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
        __global__ static void kernel(
                in0_type* x, in1_type* weight, in2_type* bias,
                out0_type* y, int hidden, double eps) {{
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
                for (int j = tid; j < hidden; j += blockDim.x) {{
                    double scale = static_cast<double>(weight[j]);
                    double offset = static_cast<double>(bias[j]);
                    y[row * hidden + j] = out0_type(
                        (static_cast<double>(x[row * hidden + j])
                         - double_mean)
                        * double_inv_std * scale + offset);
                }}
            }} else {{
                float mean = mean_shared;
                float inv_std = inv_std_shared;
                int i = 0;
                for (int j = tid; j < hidden; j += blockDim.x, ++i) {{
                    float scale = static_cast<float>(weight[j]);
                    float offset = static_cast<float>(bias[j]);
                    float xv = kCache ? cache[i]
                                      : static_cast<float>(x[row * hidden + j]);
                    y[row * hidden + j] = out0_type(
                        (xv - mean) * inv_std * scale + offset);
                }}
            }}
        }}
        double eps = data["eps"];
        int rows = in0->num / {hidden};
        kernel<<<rows, 128>>>(
            in0_p, in1_p, in2_p, out0_p, {hidden}, eps);
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
def _nchw_tile_source(hidden):
    """Fused NCHW -> NHWC affine LayerNorm using a 32-spatial warp tile.

    Each warp processes 32 consecutive spatial positions ``hw0..hw0+31`` with
    coalesced NCHW reads and coalesced NHWC writes via a 32x33 shared tile.
    """
    return f"""
        __global__ static void kernel_nchw_tile(
                const in0_type* __restrict__ x,
                const in1_type* __restrict__ weight,
                const in2_type* __restrict__ bias,
                out0_type* __restrict__ y,
                long long spatial, long long tiles_per_n, double eps) {{
            long long tile_id = blockIdx.x;
            long long n = tile_id / tiles_per_n;
            long long hw0 = (tile_id - n * tiles_per_n) * 32;
            int lane = threadIdx.x;
            long long hw = hw0 + lane;
            bool valid = hw < spatial;
            long long base = n * {hidden}LL * spatial;
            float sum = 0.0f;
            if (valid) {{
                for (int j = 0; j < {hidden}; ++j)
                    sum += static_cast<float>(x[base + (long long)j * spatial + hw]);
            }}
            bool use_double = valid && !isfinite(sum);
            float mean = sum / {hidden};
            float inv_std = 0.0f;
            if (valid && !use_double) {{
                float var = 0.0f;
                for (int j = 0; j < {hidden}; ++j) {{
                    float d = static_cast<float>(x[base + (long long)j * spatial + hw]) - mean;
                    var += d * d;
                }}
                use_double = !isfinite(var);
                if (!use_double)
                    inv_std = rsqrtf(var / {hidden} + static_cast<float>(eps));
            }}
            if (__any_sync(0xffffffffu, use_double)) {{
                if (valid) {{
                    double dsum = 0.0;
                    for (int j = 0; j < {hidden}; ++j)
                        dsum += static_cast<double>(x[base + (long long)j * spatial + hw]);
                    double dmean = dsum / {hidden};
                    double dvar = 0.0;
                    for (int j = 0; j < {hidden}; ++j) {{
                        double d = static_cast<double>(x[base + (long long)j * spatial + hw]) - dmean;
                        dvar += d * d;
                    }}
                    double dinv = 1.0 / sqrt(dvar / {hidden} + eps);
                    out0_type* yr = y + (n * spatial + hw) * {hidden};
                    for (int j = 0; j < {hidden}; ++j) {{
                        double sc = static_cast<double>(weight[j]);
                        double of = static_cast<double>(bias[j]);
                        yr[j] = out0_type((static_cast<double>(x[base + (long long)j * spatial + hw]) - dmean) * dinv * sc + of);
                    }}
                }}
                return;
            }}
            __shared__ float tile[32][33];
            for (int j0 = 0; j0 < {hidden}; j0 += 32) {{
                if (valid) {{
                    #pragma unroll 4
                    for (int k = 0; k < 32; ++k) {{
                        int j = j0 + k;
                        if (j < {hidden}) {{
                            float xv = static_cast<float>(x[base + (long long)j * spatial + hw]);
                            tile[lane][k] = (xv - mean) * inv_std * static_cast<float>(weight[j]) + static_cast<float>(bias[j]);
                        }}
                    }}
                }}
                __syncwarp();
                int j_out = j0 + lane;
                if (j_out < {hidden}) {{
                    #pragma unroll 4
                    for (int t = 0; t < 32; ++t) {{
                        long long hw_out = hw0 + t;
                        if (hw_out < spatial)
                            y[(n * spatial + hw_out) * {hidden} + j_out] = out0_type(tile[t][lane]);
                    }}
                }}
                __syncwarp();
            }}
        }}
        double eps = data["eps"];
        long long n = in0->shape[0];
        long long spatial = (long long)in0->shape[2] * in0->shape[3];
        long long tiles_per_n = (spatial + 31) / 32;
        long long grid = n * tiles_per_n;
        if (grid > 0)
            kernel_nchw_tile<<<(unsigned)grid, 32>>>(in0_p, in1_p, in2_p, out0_p, spatial, tiles_per_n, eps);
    """


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


@optional_kernel("nn.layer_norm.inference", ("cuda", "rocm_legacy", "corex_legacy"),
                 dtypes=("float16", "bfloat16", "float32"),
                 supports=_supports_layer_norm_inference)
def _layer_norm_no_grad_cuda(
        x, normalized_shape, weight, bias, eps, *, allow_bfloat16=False):
    hidden = int(normalized_shape[0])
    scalar_affine = not isinstance(weight, jt.Var) and not isinstance(bias, jt.Var)
    eps_value = float(eps)
    if scalar_affine:
        scale_value = float(weight)
        offset_value = float(bias)
        scale_literal = f"{scale_value:.9e}f"
        offset_literal = f"{offset_value:.9e}f"
        if _warp_rows(x, hidden):
            affine = ("", f"* {scale_literal} + {offset_literal}",
                      f"* {scale_value!r} + {offset_value!r}")
            return jt.code(
                x.shape, x.dtype, [x],
                cuda_src=_warp_rows_source(hidden, affine)
                + _warp_rows_launch(hidden, "in0_p, out0_p"),
                data={"eps": eps_value})
        y = jt.code(
            x.shape,
            x.dtype,
            [x],
            cuda_src=_block_rows_source_scalar(hidden, offset_literal, scale_literal),
            data={"eps": eps_value},
        )
        return y
    nchw_src = _nchw_physical(x)
    if nchw_src is not None:
        spatial = int(nchw_src.shape[2]) * int(nchw_src.shape[3])
        if spatial > 0:
            src_code = (_nchw_tile_source(hidden) if spatial >= 256
                        else _nchw_block_rows_source(hidden))
            y = jt.code(x.shape, x.dtype, [nchw_src, weight, bias],
                        cuda_src=src_code, data={"eps": eps_value})
            return _stop_grad_outputs(y)
    y = jt.code(x.shape, x.dtype, [x, weight, bias],
                cuda_src=_affine_source(hidden, _warp_rows(x, hidden)),
                data={"eps": eps_value})
    return _stop_grad_outputs(y)


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
