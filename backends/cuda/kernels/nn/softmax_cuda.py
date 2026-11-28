"""CUDA softmax kernels owned by the neural-network backend domain."""

import jittor as jt
from jittor import nn
from functools import lru_cache
from jittor._runtime.backend_libraries import library_resource
from jittor._runtime.dispatch import optional_kernel, select_kernel


_MASKED_ROW_SOFTMAX = r"""
__device__ __forceinline__ static float score_reduce(float value, bool maximum) {
    __shared__ float groups[4];
    __shared__ float result;
    const int lane = threadIdx.x & 31;
    const int group = threadIdx.x >> 5;
    for (int offset = 16; offset; offset /= 2) {
        const float other = __shfl_down_sync(0xffffffff, value, offset);
        value = maximum ? fmaxf(value, other) : value + other;
    }
    if (lane == 0) groups[group] = value;
    __syncthreads();
    if (threadIdx.x == 0)
        result = maximum
            ? fmaxf(fmaxf(groups[0], groups[1]), fmaxf(groups[2], groups[3]))
            : (groups[0] + groups[1]) + (groups[2] + groups[3]);
    __syncthreads();
    return result;
}

__device__ __forceinline__ static long long score_row_offset(
        int batch, int head, int query, int4 dims) {
    long long offset = dims.x == 1 ? 0 : batch;
    offset = offset * dims.y + (dims.y == 1 ? 0 : head);
    offset = offset * dims.z + (dims.z == 1 ? 0 : query);
    return offset * dims.w;
}

__global__ static void masked_row_softmax(
        const float* scores, const bool* mask, const float* bias, float* probabilities,
        int heads, int queries, int keys, int4 mask_shape, int4 bias_shape, float scale, float fill) {
    const int row = blockIdx.x;
    const int batch = row / (heads * queries);
    const int head = (row / queries) % heads;
    const int query = row % queries;
    const bool* mask_row = mask + score_row_offset(batch, head, query, mask_shape);
    const float* bias_row = bias + score_row_offset(batch, head, query, bias_shape);
    const long long base = (long long)row * keys;
    extern __shared__ float weights[];
    float maximum = -INFINITY;
    for (int k = threadIdx.x; k < keys; k += blockDim.x) {
        float score = mask_row[mask_shape.w == 1 ? 0 : k]
            ? fill : scores[base + k] * scale;
        score += bias_row[bias_shape.w == 1 ? 0 : k];
        weights[k] = score;
        maximum = fmaxf(maximum, score);
    }
    maximum = score_reduce(maximum, true);
    float sum = 0.f;
    for (int k = threadIdx.x; k < keys; k += blockDim.x) {
        const float weight = expf(weights[k] - maximum);
        weights[k] = weight;
        sum += weight;
    }
    sum = score_reduce(sum, false);
    for (int k = threadIdx.x; k < keys; k += blockDim.x)
        probabilities[base + k] = weights[k] / sum;
}
const auto broadcast_shape = [](Var* input) {
    int dims[4] = {1, 1, 1, 1};
    const int rank = input->shape.size();
    for (int i = 0; i < rank; ++i) dims[4 - rank + i] = input->shape[i];
    return make_int4(dims[0], dims[1], dims[2], dims[3]);
};
masked_row_softmax<<<in0->num / in0_shape3, 128, in0_shape3 * sizeof(float)>>>(
    in0_p, in1_p, in2_p, out0_p, in0_shape1, in0_shape2, in0_shape3,
    broadcast_shape(in1), broadcast_shape(in2), data.at("scale"), data.at("fill"));
CHECK(0 == cudaGetLastError());
"""
if hasattr(jt.core, "_bind_masked_row_softmax"):
    jt.core._bind_masked_row_softmax(_MASKED_ROW_SOFTMAX)


def _supports_softmax(a, log=False, zero_all_neg_inf=False, dim=-1):
    shape = a.shape
    return bool(shape) and (dim == -1 or dim == len(shape) - 1) and int(shape[-1]) > 0


def can_softmax_v1(a, dim):
    return select_kernel("nn.softmax", a, dim=dim) is not None


def _softmax_schedule(length):
    if length < 10000:
        return "register", (125 if length % 125 == 0 else 128)
    if length <= 16384:
        return "register", (500 if length % 500 == 0 else 512)
    if length <= 65536:
        return "register", (250 if length % 250 == 0 else 256)
    return "stream", 512


def _softmax_streaming_grad(x, grad_x, threads, log=False):
    return jt.code(
        x.shape,
        x.dtype,
        [x, grad_x],
        cuda_header=f'''
#include <{library_resource('cub', 'home')}cub/cub.cuh>
#include <type/fp16_compute.h>
''',
        cuda_src=f'''
__global__ void kernel(in0_type* x, in1_type* g, out0_type* dx, int len) {{
    typedef cub::BlockReduce<float, {threads}> BlockReduce;
    constexpr int need_log = {int(log)};
    __shared__ typename BlockReduce::TempStorage storage;
    __shared__ float row_sum;
    int base = blockIdx.x * len;

    float local = 0.0f;
    for (int col = threadIdx.x; col < len; col += blockDim.x)
        local += need_log
            ? float(g[base + col])
            : float(x[base + col]) * float(g[base + col]);
    float reduced = BlockReduce(storage).Sum(local);
    if (threadIdx.x == 0) row_sum = reduced;
    __syncthreads();

    for (int col = threadIdx.x; col < len; col += blockDim.x) {{
        float xv = float(x[base + col]);
        float gv = float(g[base + col]);
        dx[base + col] = need_log
            ? out0_type(gv - expf(xv) * row_sum)
            : out0_type(xv * (gv - row_sum));
    }}
}}
int len = in0->shape[in0->shape.size()-1];
int rows = in0->numel() / len;
kernel<<<rows, {threads}>>>(in0_p, in1_p, out0_p, len);
CHECK(0 == cudaGetLastError());
''')


@lru_cache(maxsize=8)
def _softmax_header(cub_home):
    return f'''
#include <{cub_home}cub/cub.cuh>
#include <type/fp16_compute.h>
#include <type_traits>
'''


@lru_cache(maxsize=256)
def _softmax_v1_cls(length, log=False, zero_all_neg_inf=False):
    # The generated Function class and CUDA source depend only on the reduced
    # length and log mode. Building both inside every softmax call added tens of
    # microseconds of Python graph-construction overhead even after the kernel
    # was compiled, which is material for transformer attention. Cache the
    # immutable class and its formatted sources while still creating a fresh
    # Function instance per call. Include paths remain resolved at execution.
    _, tnum = _softmax_schedule(length)
    per_thread = (length-1) // tnum + 1
    ILP = 1
    for ilp in [8,4,2]:
        if length % tnum == 0 and per_thread % ilp == 0:
            ILP = ilp
            per_thread //= ILP
            break
    for_loop = f"""
    #pragma unroll
    for (int i=0; i<{per_thread}; i++)
    """
    if length % tnum != 0:
        for_loop += f"if ((i*{tnum}+threadIdx.x)*{ILP} < len)\n"

    # Short rows need no communication between warps. Pack eight independent
    # warp reductions into one block when there are enough rows to fill
    # the GPU. Wider rows keep the block kernel: its extra
    # memory parallelism is faster once a warp holds too many columns.
    warp_source = f"""
__global__ void softmax_warp_rows(in0_type* x, out0_type* y, int len, int rows) {{
    long long row = static_cast<long long>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= rows) return;
    constexpr int need_log = {int(log)};
    constexpr int zero_neg_inf = {int(zero_all_neg_inf)};
    constexpr int per = {(length + 31) // 32};
    float values[per];
    float maximum = -INFINITY;
    #pragma unroll
    for (int i = 0; i < per; ++i) {{
        int col = lane + i * 32;
        values[i] = col < len ? float(x[row * len + col]) : -INFINITY;
        maximum = ::max(maximum, values[i]);
    }}
    for (int offset = 16; offset > 0; offset >>= 1)
        maximum = ::max(maximum, __shfl_xor_sync(0xffffffffu, maximum, offset));
    float total = 0.0f;
    #pragma unroll
    for (int i = 0; i < per; ++i) {{
        if (lane + i * 32 < len) {{
            if (zero_neg_inf && !need_log && maximum == -INFINITY) {{
                values[i] = 0.0f;
            }} else if (need_log) {{
                values[i] -= maximum;
                total += expf(values[i]);
            }} else {{
                values[i] = expf(values[i] - maximum);
                total += values[i];
            }}
        }}
    }}
    for (int offset = 16; offset > 0; offset >>= 1)
        total += __shfl_xor_sync(0xffffffffu, total, offset);
    #pragma unroll
    for (int i = 0; i < per; ++i) {{
        int col = lane + i * 32;
        if (col < len) {{
            float value = (zero_neg_inf && !need_log && maximum == -INFINITY) ? 0.0f :
                (need_log ? values[i] - logf(total) : values[i] / total);
            y[row * len + col] = value;
        }}
    }}
}}
""" if length <= 256 else ""
    launch = f"kernel<<<bnum, {tnum}>>>(in0_p, out0_p, len);"
    if warp_source:
        launch = f"""
constexpr bool narrow_half_rows = sizeof(in0_type) == 2 && {length} <= 128;
if (bnum >= (narrow_half_rows ? 1024 : 4096)
        && (narrow_half_rows || std::is_same<in0_type, float>::value)) {{
    softmax_warp_rows<<<(static_cast<unsigned>(bnum) + 7) / 8, 256>>>(in0_p, out0_p, len, bnum);
}} else {{
    {launch}
}}
"""

    execute_source = warp_source + f'''
__global__ void kernel(in0_type* x, out0_type* y, int len) {{
    typedef cub::BlockReduce<float, {tnum}> BlockReduce;
    constexpr int need_log = {int(log)};
    constexpr int zero_neg_inf = {int(zero_all_neg_inf)};
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int id = blockIdx.x * len;
    in0_type v[{per_thread}][{ILP}];
    {for_loop}
        vload<sizeof(in0_type)*{ILP}>(v[i], &x[id+(i*{tnum}+threadIdx.x)*{ILP}]);
    // v[i] = x[id+i*{tnum}+threadIdx.x];
    float v1 = -INFINITY;
    {for_loop}
        #pragma unroll
        for (int j=0; j<{ILP}; j++) {{
            v1 = ::max(v1, float(v[i][j]));
        }}
    __shared__ float vmax;
#if CUB_VERSION >= 300000
    auto tmp = BlockReduce(temp_storage).Reduce(v1, ::cuda::maximum<>());
#else
    auto tmp = BlockReduce(temp_storage).Reduce(v1, cub::Max());
#endif
    if (threadIdx.x == 0)
        vmax = tmp;
    __syncthreads();

    v1 = 0;
    {for_loop}
        #pragma unroll
        for (int j=0; j<{ILP}; j++) {{
            if (zero_neg_inf && !need_log && vmax == -INFINITY) {{
                v[i][j] = 0.0f;
            }} else if (need_log) {{
                v[i][j] = float(v[i][j]) - vmax;
                v1 += expf(float(v[i][j]));
            }} else {{
                v[i][j] = expf(float(v[i][j]) - vmax);
                v1 += float(v[i][j]);
            }}
        }}

    tmp = BlockReduce(temp_storage).Sum(v1);
    __shared__ float vsum;
    if (threadIdx.x == 0)
        vsum = tmp;
    __syncthreads();

    {for_loop}
        #pragma unroll
        for (int j=0; j<{ILP}; j++) {{
            if (zero_neg_inf && !need_log && vmax == -INFINITY)
                v[i][j] = 0.0f;
            else if (need_log)
                v[i][j] = v[i][j] - @expand_op(log,@in0_type,vsum);
            else
                v[i][j] = float(v[i][j])/vsum;
        }}
    {for_loop}
        vload<sizeof(in0_type)*{ILP}>(&y[id+(i*{tnum}+threadIdx.x)*{ILP}], v[i]);
}}
int len = in0->shape[in0->shape.size()-1];
int bnum = in0->numel() / len;
cudaGetLastError();
{launch}
CHECK(0 == cudaGetLastError());
'''
    grad_source = f"""
__global__ void kernel(in0_type* x, in1_type* y, out0_type* z, int len) {{
    int id = blockIdx.x * len;
    in0_type vx[{per_thread}][{ILP}];
    in0_type vy[{per_thread}][{ILP}];
    {for_loop} {{
        vload<sizeof(in0_type)*{ILP}>(vx[i], &x[id+(i*{tnum}+threadIdx.x)*{ILP}]);
        vload<sizeof(in0_type)*{ILP}>(vy[i], &y[id+(i*{tnum}+threadIdx.x)*{ILP}]);
    }}
    float v1 = 0;
    {for_loop} 
        #pragma unroll
        for (int j=0; j<{ILP}; j++)
            v1 += {"float(vy[i][j]);" if log else "float(vx[i][j]*vy[i][j]);"}

    typedef cub::BlockReduce<float, {tnum}> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    auto tmp = BlockReduce(temp_storage).Sum(v1);
    __shared__ float reduce_var;
    if (threadIdx.x == 0)
        reduce_var = tmp;
    __syncthreads();

    {for_loop}
        #pragma unroll
        for (int j=0; j<{ILP}; j++)
            vx[i][j] = {
                "vy[i][j] - in0_type(expf(vx[i][j]) * reduce_var);" if log 
                else "vx[i][j] * (vy[i][j] - in0_type(reduce_var));"
            }

    {for_loop}
        vload<sizeof(in0_type)*{ILP}>(&z[id+(i*{tnum}+threadIdx.x)*{ILP}],
            vx[i]);
}}
int len = in0->shape[in0->shape.size()-1];
int bnum = in0->numel() / len;
cudaGetLastError();
kernel<<<bnum, {tnum}>>>(in0_p, in1_p, out0_p, len);
CHECK(0 == cudaGetLastError());
"""

    def forward(x, inference):
        header = _softmax_header(library_resource("cub", "home"))
        if inference:
            return jt.core._cuda_inference_unary_code(
                x, execute_source, header, not log and not zero_all_neg_inf)
        return jt.code(x.shape, x.dtype, [x], cuda_header=header, cuda_src=execute_source)

    class CodeSoftmax(jt.Function):
        @staticmethod
        def inference(x):
            return forward(x, True)

        def execute(self, x):
            self.save_vars = forward(x, False)
            return self.save_vars

        def grad(self, grad_x):
            x = self.save_vars
            # Non-vectorized register kernels above roughly 48K columns spill
            # heavily in backward (for example, 255 registers at 65535). Keep
            # their faster register-resident forward, but use the bounded
            # streaming reduction for backward. Aligned ILP paths such as
            # 65536 remain faster with the register kernel.
            # The streaming kernel loads the saved output and cotangent
            # using their own types. The vectorized loads assume one type.
            if grad_x.dtype != x.dtype or (length > 49152 and ILP == 1):
                return _softmax_streaming_grad(x, grad_x, 512, log)
            return jt.code(x.shape, x.dtype, [x, grad_x], cuda_header=f'''
#include <{library_resource('cub', 'home')}cub/cub.cuh>
#include <type/fp16_compute.h>
''', cuda_src=grad_source)
    return CodeSoftmax


# Wide contiguous FP32 log-softmax follows the CUDA reduction order used by
# ATen: float4 chunks, a scalar tail, then down-shuffles across lanes and warps.
_WIDE_FP32_LOG = r"""
__global__ void wide_log_softmax(const float* x, float* y, int width) {
    __shared__ float partial[32];
    __shared__ float maximum;
    __shared__ float log_sum;
    int tid = threadIdx.x;
    int lane = tid % 32;
    int warp = tid / 32;
    x += int64_t(blockIdx.x) * width;
    y += int64_t(blockIdx.x) * width;
    float high = -INFINITY;
    for (int j = tid; j < width; j += blockDim.x) high = fmaxf(high, x[j]);
    for (int d = 16; d; d >>= 1)
        high = fmaxf(high, __shfl_down_sync(0xffffffff, high, d));
    if (lane == 0) partial[warp] = high;
    __syncthreads();
    if (warp == 0) {
        high = partial[lane];
        for (int d = 16; d; d >>= 1)
            high = fmaxf(high, __shfl_down_sync(0xffffffff, high, d));
        if (lane == 0) maximum = high;
    }
    __syncthreads();
    const int shift = (uintptr_t(x) % 16) / sizeof(float);
    const int prefix = shift ? blockDim.x - shift : 0;
    float sum = 0.f;
    if (shift && tid >= shift) sum = expf(x[tid - shift] - maximum);
    int end = width - (width - prefix) % (4 * blockDim.x);
    for (int j = prefix + 4 * tid; j < end; j += 4 * blockDim.x) {
        float4 v = *reinterpret_cast<const float4*>(x + j);
        sum += expf(v.x - maximum);
        sum += expf(v.y - maximum);
        sum += expf(v.z - maximum);
        sum += expf(v.w - maximum);
    }
    for (int j = end + tid; j < width; j += blockDim.x) sum += expf(x[j] - maximum);
    for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (lane == 0) partial[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = partial[lane];
        for (int d = 16; d; d >>= 1) sum += __shfl_down_sync(0xffffffff, sum, d);
        if (lane == 0) log_sum = logf(sum);
    }
    __syncthreads();
    for (int j = tid; j < width; j += blockDim.x) y[j] = (x[j] - maximum) - log_sum;
}
int width = in0->shape[in0->shape.size()-1];
int rows = in0->num / width;
if (rows) wide_log_softmax<<<rows, 1024>>>(in0_p, out0_p, width);
CHECK(0 == cudaGetLastError());
"""


_ROW_FLOOR_INPUT = r"""
__device__ __forceinline__ static float softmax_floor_input(float value, float floor) {
    return value < floor ? -CUDART_INF_F : value;
}
"""


# Large vocabularies with only a few rows underutilize a one-block-per-row
# reduction. Compute per-tile maxima and sums, then scale independent tiles.
_TILED_FP32_SOFTMAX = r"""
__global__ static void tiled_softmax_stats(const float* input, float* stats, int width, int tiles) {
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage scratch;
    __shared__ float maximum;
    const int row = blockIdx.y;
    const int start = blockIdx.x * 4096;
    const int end = min(start + 4096, width);
    const int row_base = row * width;
    float partial_max = -CUDART_INF_F;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x)
        partial_max = fmaxf(partial_max, input[row_base + i]);
    #if CUB_VERSION >= 300000
    const float reduced_max = Reduce(scratch).Reduce(partial_max, cuda::maximum<>{});
    #else
    const float reduced_max = Reduce(scratch).Reduce(partial_max, cub::Max());
    #endif
    if (threadIdx.x == 0) maximum = reduced_max;
    __syncthreads();
    float partial_sum = 0.0f;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const float value = input[row_base + i];
        // A fully masked tile contributes zero; a NaN still propagates.
        partial_sum += (maximum == -CUDART_INF_F && value == maximum) ? 0.0f : expf(value - maximum);
    }
    const float reduced_sum = Reduce(scratch).Sum(partial_sum);
    if (threadIdx.x == 0) {
        int index = 2 * (row * tiles + blockIdx.x);
        stats[index] = maximum;
        stats[index + 1] = reduced_sum;
    }
}
__global__ static void tiled_softmax_out(const float* input, const float* stats, float* output,
                                         int width, int tiles) {
    using Reduce = cub::BlockReduce<float, 256>;
    __shared__ typename Reduce::TempStorage scratch;
    __shared__ float maximum, total;
    const int row = blockIdx.y;
    const int base = 2 * row * tiles;
    const int start = blockIdx.x * 4096;
    const int end = min(start + 4096, width);
    float partial_max = -CUDART_INF_F;
    for (int i = threadIdx.x; i < tiles; i += blockDim.x)
        partial_max = fmaxf(partial_max, stats[base + 2*i]);
    #if CUB_VERSION >= 300000
    const float reduced_max = Reduce(scratch).Reduce(partial_max, cuda::maximum<>{});
    #else
    const float reduced_max = Reduce(scratch).Reduce(partial_max, cub::Max());
    #endif
    if (threadIdx.x == 0) maximum = reduced_max;
    __syncthreads();
    float partial_sum = 0.0f;
    for (int i = threadIdx.x; i < tiles; i += blockDim.x) {
        const float tile_max = stats[base + 2*i];
        const float tile_sum = stats[base + 2*i + 1];
        partial_sum += tile_max == -CUDART_INF_F ? tile_sum : tile_sum * expf(tile_max - maximum);
    }
    const float reduced_sum = Reduce(scratch).Sum(partial_sum);
    if (threadIdx.x == 0) total = reduced_sum;
    __syncthreads();
    const float reciprocal = 1.0f / total;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const int index = row * width + i;
        output[index] = expf(input[index] - maximum) * reciprocal;
    }
}
const int width = in0_shape1;
const int tiles = (width + 4095) / 4096;
auto stream = reinterpret_cast<cudaStream_t>(
    backend_stream({BackendId::Cuda, current_device()}, BackendStreamKind::Compute).handle);
tiled_softmax_stats<<<dim3(tiles, in0_shape0), 256, 0, stream>>>(in0_p, out1_p, width, tiles);
tiled_softmax_out<<<dim3(tiles, in0_shape0), 256, 0, stream>>>(in0_p, out1_p, out0_p, width, tiles);
CHECK(cudaGetLastError() == cudaSuccess);
"""


@lru_cache(maxsize=2)
def _wide_tiled_softmax_source(masked):
    if not masked:
        return _TILED_FP32_SOFTMAX
    source = _TILED_FP32_SOFTMAX.replace("const float* input, ", "const float* input, const float* floors, ")
    source = source.replace("    const int row = blockIdx.y;",
                            "    const int row = blockIdx.y;\n    const float floor = floors[row];")
    source = source.replace("input[row_base + i]", "softmax_floor_input(input[row_base + i], floor)")
    source = source.replace("input[index]", "softmax_floor_input(input[index], floor)")
    source = source.replace("(in0_p, out1_p,", "(in0_p, in1_p, out1_p,")
    return source


def _wide_tiled_softmax(rows, floors=None):
    count, width = rows.shape
    tiles = (width + 4095) // 4096
    masked = floors is not None
    output, _ = jt.code(
        [rows.shape, (count, 2 * tiles)], [rows.dtype, rows.dtype],
        [rows, floors] if masked else [rows],
        cuda_header=f'''
#include <{library_resource('cub', 'home')}cub/cub.cuh>
#include <math_constants.h>
#include "runtime/device.h"
''' + (_ROW_FLOOR_INPUT if masked else ""),
        cuda_src=_wide_tiled_softmax_source(masked), backend="cuda")
    return output


@lru_cache(maxsize=8)
def _softmax_streaming_cls(threads, log=False, zero_all_neg_inf=False):
    # Large vocabularies cannot keep a complete row in registers without
    # spilling or exceeding the per-block register limit. Re-read logits for
    # max, exp-sum and output instead; this remains one launch and bounds
    # register use independently of vocabulary size.
    execute_source = f'''
__global__ void kernel(in0_type* x, out0_type* y, int len) {{
    typedef cub::BlockReduce<float, {threads}> BlockReduce;
    constexpr int need_log = {int(log)};
    constexpr int zero_neg_inf = {int(zero_all_neg_inf)};
    __shared__ typename BlockReduce::TempStorage storage;
    __shared__ float row_max;
    __shared__ float row_sum;
    int base = blockIdx.x * len;

    float local = -INFINITY;
    for (int col = threadIdx.x; col < len; col += blockDim.x)
        local = ::max(local, float(x[base + col]));
#if CUB_VERSION >= 300000
    float reduced = BlockReduce(storage).Reduce(local, ::cuda::maximum<>());
#else
    float reduced = BlockReduce(storage).Reduce(local, cub::Max());
#endif
    if (threadIdx.x == 0) row_max = reduced;
    __syncthreads();

    local = 0.0f;
    for (int col = threadIdx.x; col < len; col += blockDim.x)
        local += (zero_neg_inf && !need_log && row_max == -INFINITY)
            ? 0.0f : expf(float(x[base + col]) - row_max);
    reduced = BlockReduce(storage).Sum(local);
    if (threadIdx.x == 0) row_sum = reduced;
    __syncthreads();

    for (int col = threadIdx.x; col < len; col += blockDim.x) {{
        float shifted = float(x[base + col]) - row_max;
        y[base + col] = (zero_neg_inf && !need_log && row_max == -INFINITY)
            ? out0_type(0.0f) : need_log
            ? out0_type(shifted - logf(row_sum))
            : out0_type(expf(shifted) / row_sum);
    }}
}}
int len = in0->shape[in0->shape.size()-1];
int rows = in0->numel() / len;
kernel<<<rows, {threads}>>>(in0_p, out0_p, len);
CHECK(0 == cudaGetLastError());
'''

    class CodeSoftmaxStreaming(jt.Function):
        def execute(self, x):
            if (not log and not zero_all_neg_inf and str(x.dtype) in ("float32", "torch.float32")
                    and x.ndim == 2 and 0 < x.shape[0] <= 8 and x.shape[1] <= 262144
                    and x._storage_is_contiguous()):
                self.save_vars = _wide_tiled_softmax(x)
                return self.save_vars
            self.save_vars = jt.code(
                x.shape,
                x.dtype,
                [x],
                cuda_header=f'''
#include <{library_resource('cub', 'home')}cub/cub.cuh>
#include <type/fp16_compute.h>
''',
                cuda_src=(_WIDE_FP32_LOG if log and str(x.dtype) in ("float32", "torch.float32")
                          else execute_source))
            return self.save_vars

        def grad(self, grad_x):
            x = self.save_vars
            return _softmax_streaming_grad(x, grad_x, threads, log)
    return CodeSoftmaxStreaming


def _softmax_fp32_with_row_floor(rows, floors):
    """Apply a row floor before ordinary FP32 softmax, without a masked copy."""
    if (not jt.flags.use_cuda or not jt.flags.no_grad
            or rows.ndim != 2 or not 0 < rows.shape[0] <= 8
            or not 65536 < rows.shape[1] <= 262144
            or tuple(floors.shape) != (rows.shape[0], 1)
            or str(rows.dtype) not in ("float32", "torch.float32")
            or rows.dtype != floors.dtype
            or rows.placement_backend != 1 or floors.placement_backend != 1
            or rows.device_id != floors.device_id
            or not rows._storage_is_contiguous() or not floors._storage_is_contiguous()):
        return None
    return _wide_tiled_softmax(rows, floors)


@optional_kernel("nn.softmax", "cuda", dtypes=("float16", "bfloat16", "float32"),
                 supports=_supports_softmax)
def _softmax_v1(a, log=False, zero_all_neg_inf=False, dim=-1):
    length = int(a.shape[-1])
    kind, threads = _softmax_schedule(length)
    cls = (_softmax_v1_cls(length, bool(log), bool(zero_all_neg_inf))
           if kind == "register" else
           _softmax_streaming_cls(threads, bool(log), bool(zero_all_neg_inf)))
    no_grad = jt.flags.no_grad
    result = cls.inference(a) if kind == "register" and no_grad else cls()(a)
    if (not log and not zero_all_neg_inf and a.ndim == 4 and no_grad
            and 16 <= length <= 64 and a.numel() // length >= 1024):
        from .short_attention_cuda import record_softmax

        record_softmax(result, a)
    return result


def softmax_v1(a, log=False, zero_all_neg_inf=False):
    return _softmax_v1(a, log=log, zero_all_neg_inf=zero_all_neg_inf)


def _supports_torch_logsumexp(x):
    return (x.ndim == 2 and int(x.shape[0]) > 0 and int(x.shape[1]) > 0
            and x.numel() <= 2147483647 and x._storage_is_contiguous())


@optional_kernel("nn.torch_logsumexp_last", "cuda", dtypes=("float32",),
                 supports=_supports_torch_logsumexp)
def _torch_logsumexp_last(x):
    """Fixed-order row LSE with Torch's NaN and infinity behavior.

    This opt-in kernel is selected by the Torch compatibility frontend only;
    it does not alter native Jittor reductions or the cross-entropy kernel.
    """
    threads = 512
    return jt.code((int(x.shape[0]),), x.dtype, [x], cuda_header=f'''
#include <{library_resource('cub', 'home')}cub/cub.cuh>
#include <math.h>
''', cuda_src=f'''
__global__ void kernel(in0_type* x, out0_type* y, int len) {{
    typedef cub::BlockReduce<float, {threads}> FloatReduce;
    typedef cub::BlockReduce<int, {threads}> IntReduce;
    __shared__ typename FloatReduce::TempStorage float_storage;
    __shared__ typename IntReduce::TempStorage int_storage;
    __shared__ float row_max;
    __shared__ int row_nan;
    const int base = blockIdx.x * len;
    float local_max = -INFINITY;
    int local_nan = 0;
    for (int col = threadIdx.x; col < len; col += blockDim.x) {{
        const float v = float(x[base + col]);
        if (isnan(v)) local_nan = 1;
        else local_max = fmaxf(local_max, v);
    }}
#if CUB_VERSION >= 300000
    const float maximum = FloatReduce(float_storage).Reduce(local_max, ::cuda::maximum<>());
    const int has_nan = IntReduce(int_storage).Reduce(local_nan, ::cuda::maximum<>());
#else
    const float maximum = FloatReduce(float_storage).Reduce(local_max, cub::Max());
    const int has_nan = IntReduce(int_storage).Reduce(local_nan, cub::Max());
#endif
    if (threadIdx.x == 0) {{ row_max = maximum; row_nan = has_nan; }}
    __syncthreads();
    if (row_nan) {{
        if (threadIdx.x == 0) y[blockIdx.x] = nanf("");
        return;
    }}
    if (isinf(row_max)) {{
        if (threadIdx.x == 0) y[blockIdx.x] = row_max;
        return;
    }}
    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < len; col += blockDim.x)
        local_sum += expf(float(x[base + col]) - row_max);
    const float total = FloatReduce(float_storage).Sum(local_sum);
    if (threadIdx.x == 0) y[blockIdx.x] = row_max + logf(total);
}}
int rows = in0->shape[0];
int len = in0->shape[1];
kernel<<<rows, {threads}>>>(in0_p, out0_p, len);
CHECK(0 == cudaGetLastError());
''')
