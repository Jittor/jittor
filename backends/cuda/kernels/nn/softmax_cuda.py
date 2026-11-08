"""CUDA softmax kernels owned by the neural-network backend domain."""

import jittor as jt
from jittor import nn
from functools import lru_cache
from jittor._runtime.backend_libraries import library_resource
from jittor._runtime.dispatch import optional_kernel, select_kernel

def _supports_softmax(a, log=False, zero_all_neg_inf=False, dim=-1):
    if not a.shape:
        return False
    if dim != -1 and dim != len(a.shape)-1:
        return False
    if int(a.shape[-1]) <= 0:
        return False
    return True


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


@lru_cache(maxsize=32)
def _softmax_warp_fp32_cls(length):
    """Keep a short FP32 row in one warp with a fixed reduction tree.

    Accumulate strided columns per lane, then combine descending XOR partners.
    Unlike a block-wide reduction, this retains the same rounded FP32 values
    when the probabilities subsequently cross a low-precision cast boundary.
    """
    iterations = (length + 31) // 32

    class WarpSoftmax(jt.Function):
        def execute(self, x):
            self.output = jt.code(x.shape, x.dtype, [x], cuda_src=f'''
__global__ void kernel(const in0_type* x, out0_type* y, int rows) {{
    int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= rows) return;
    int lane = threadIdx.x;
    float values[{iterations}], maximum = -INFINITY;
    #pragma unroll
    for (int i = 0; i < {iterations}; ++i) {{
        int column = lane + i * 32;
        values[i] = column < {length} ? x[row * {length} + column] : -INFINITY;
        maximum = maximum < values[i] ? values[i] : maximum;
    }}
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {{
        float other = __shfl_xor_sync(0xffffffff, maximum, offset);
        maximum = maximum < other ? other : maximum;
    }}
    float total = 0.f;
    #pragma unroll
    for (int i = 0; i < {iterations}; ++i) {{
        values[i] = expf(values[i] - maximum);
        total += values[i];
    }}
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        total += __shfl_xor_sync(0xffffffff, total, offset);
    #pragma unroll
    for (int i = 0; i < {iterations}; ++i) {{
        int column = lane + i * 32;
        if (column < {length}) y[row * {length} + column] = values[i] / total;
    }}
}}
int rows = in0->numel() / {length};
kernel<<<(rows + 3) / 4, dim3(32, 4)>>>(in0_p, out0_p, rows);
CHECK(0 == cudaGetLastError());
''')
            return self.output

        def grad(self, grad_output):
            output = self.output
            return jt.code(output.shape, output.dtype, [output, grad_output], cuda_src=f'''
__global__ void kernel(const in0_type* y, const in1_type* g, out0_type* dx, int rows) {{
    int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= rows) return;
    int lane = threadIdx.x;
    float probabilities[{iterations}], gradients[{iterations}], total = 0.f;
    #pragma unroll
    for (int i = 0; i < {iterations}; ++i) {{
        int column = lane + i * 32;
        probabilities[i] = column < {length} ? y[row * {length} + column] : 0.f;
        gradients[i] = column < {length} ? g[row * {length} + column] : 0.f;
        total += probabilities[i] * gradients[i];
    }}
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        total += __shfl_xor_sync(0xffffffff, total, offset);
    #pragma unroll
    for (int i = 0; i < {iterations}; ++i) {{
        int column = lane + i * 32;
        if (column < {length})
            dx[row * {length} + column] = probabilities[i] * (gradients[i] - total);
    }}
}}
int rows = in0->numel() / {length};
kernel<<<(rows + 3) / 4, dim3(32, 4)>>>(in0_p, in1_p, out0_p, rows);
CHECK(0 == cudaGetLastError());
''')

    return WarpSoftmax


@lru_cache(maxsize=256)
def _softmax_v1_cls(length, log=False, zero_all_neg_inf=False):
    # The generated Function class and CUDA source depend only on the reduced
    # length and log mode. Building both inside every softmax call added tens of
    # microseconds of Python graph-construction overhead even after the kernel
    # was compiled, which is material for transformer attention. Cache the
    # immutable class while still creating a fresh Function instance per call.
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

    class CodeSoftmax(jt.Function):
        def execute(self, x):
            self.save_vars = jt.code(x.shape, x.dtype, [x], cuda_header=f'''
#include <{library_resource('cub', 'home')}cub/cub.cuh>
#include <type/fp16_compute.h>
''', cuda_src=f'''
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
kernel<<<bnum, {tnum}>>>(in0_p, out0_p, len);
CHECK(0 == cudaGetLastError());
''')
            return self.save_vars

        def grad(self, grad_x):
            x = self.save_vars
            # Non-vectorized register kernels above roughly 48K columns spill
            # heavily in backward (for example, 255 registers at 65535). Keep
            # their faster register-resident forward, but use the bounded
            # streaming reduction for backward. Aligned ILP paths such as
            # 65536 remain faster with the register kernel.
            if length > 49152 and ILP == 1:
                return _softmax_streaming_grad(x, grad_x, 512, log)
            return jt.code(x.shape, x.dtype, [x, grad_x], cuda_header=f'''
#include <{library_resource('cub', 'home')}cub/cub.cuh>
#include <type/fp16_compute.h>
''', 
                cuda_src=f"""
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
""")
    return CodeSoftmax


@lru_cache(maxsize=8)
def _softmax_streaming_cls(threads, log=False, zero_all_neg_inf=False):
    # Large vocabularies cannot keep a complete row in registers without
    # spilling or exceeding the per-block register limit. Re-read logits for
    # max, exp-sum and output instead; this remains one launch and bounds
    # register use independently of vocabulary size.
    class CodeSoftmaxStreaming(jt.Function):
        def execute(self, x):
            self.save_vars = jt.code(
                x.shape,
                x.dtype,
                [x],
                cuda_header=f'''
#include <{library_resource('cub', 'home')}cub/cub.cuh>
#include <type/fp16_compute.h>
''',
                cuda_src=f'''
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
''')
            return self.save_vars

        def grad(self, grad_x):
            x = self.save_vars
            return _softmax_streaming_grad(x, grad_x, threads, log)
    return CodeSoftmaxStreaming


@optional_kernel("nn.softmax", "cuda", dtypes=("float16", "bfloat16", "float32"),
                 supports=_supports_softmax)
def _softmax_v1(a, log=False, zero_all_neg_inf=False, dim=-1):
    length = int(a.shape[-1])
    from jittor._core.dtypes import dtype_name
    if length <= 1024 and not log and not zero_all_neg_inf and dtype_name(a.dtype) == "float32":
        return _softmax_warp_fp32_cls(length)()(a)
    kind, threads = _softmax_schedule(length)
    cls = (_softmax_v1_cls(length, bool(log), bool(zero_all_neg_inf))
           if kind == "register" else
           _softmax_streaming_cls(threads, bool(log), bool(zero_all_neg_inf)))
    return cls()(a)


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
