"""CUDA softmax kernels owned by the neural-network backend domain."""

import jittor as jt
from jittor import nn
from functools import lru_cache

def can_softmax_v1(a, dim):
    if not jt.flags.use_cuda:
        return False
    if dim != -1 and dim != len(a.shape)-1:
        return False
    if int(a.shape[-1]) <= 0:
        return False
    if str(a.dtype) not in ("float16", "bfloat16", "float32"):
        return False
    return True


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
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
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
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
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
    auto tmp = BlockReduce(temp_storage).Reduce(v1, cub::Max());
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
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
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
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
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
    float reduced = BlockReduce(storage).Reduce(local, cub::Max());
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


def softmax_v1(a, log=False, zero_all_neg_inf=False):
    assert can_softmax_v1(a, -1)
    length = int(a.shape[-1])
    kind, threads = _softmax_schedule(length)
    cls = (_softmax_v1_cls(length, bool(log), bool(zero_all_neg_inf))
           if kind == "register" else
           _softmax_streaming_cls(threads, bool(log), bool(zero_all_neg_inf)))
    return cls()(a)
