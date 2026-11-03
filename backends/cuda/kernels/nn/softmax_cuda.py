"""CUDA softmax kernels owned by the neural-network backend domain."""

import jittor as jt
from jittor import nn as nn
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
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


def _streaming_row_reduction(threads, log=False, zero_all_neg_inf=False):
    """One block owns one row; shared by streaming softmax and CE's small LSE."""
    return f'''
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
    __syncthreads();'''[1:]


@lru_cache(maxsize=128)
def _softmax_warp_cls(length, log=False, zero_all_neg_inf=False):
    """Short rows use a fixed butterfly tree and FP32 arithmetic until output."""
    width = min(32, 1 << (length - 1).bit_length())
    iterations = ((1 << (length - 1).bit_length()) + width - 1) // width
    header = '#include <type/fp16_compute.h>\n'
    geometry = f'''
    constexpr int width = {width}, iterations = {iterations}, length = {length};
    int lane = threadIdx.x % width;
    int row = (blockIdx.x * blockDim.x + threadIdx.x) / width;
    if (row >= rows) return;
    int base = row * length;
'''
    launch = f'''
int rows = in0->numel() / {length};
if (rows) kernel<<<(rows * {width} + 127) / 128, 128>>>(in0_p, out0_p, rows);
CHECK(0 == cudaGetLastError());
'''

    class WarpSoftmaxVJP(jt.Function):
        def _run_call(self, y, seed):
            self.y, self.seed = (jt.ops.reshape(v, v.shape) for v in (y, seed))
            return super()._run_call(y, seed)

        def execute(self, y, seed):
            # The softmax VJP first rounds output*seed to the tensor dtype;
            # then its reduction and subtraction stay in FP32. Avoid the
            # algebraically factorized form, which changes narrow gradients.
            return jt.code(y.shape, y.dtype, [y, seed], cuda_header=header,
                cuda_src=f'''
__global__ void kernel(in0_type* y, in1_type* g, out0_type* dx, int rows) {{
{geometry}
    float out[iterations], product[iterations];
    float sum = 0.0f;
    #pragma unroll
    for (int i=0; i<iterations; ++i) {{
        int col = lane + i * width;
        out[i] = col < length ? float(y[base+col]) : 0.0f;
        product[i] = col < length ? ({int(log)} ? float(g[base+col])
            : float(in0_type(out[i] * float(g[base+col])))) : 0.0f;
        sum += product[i];
    }}
    #pragma unroll
    for (int offset=width/2; offset; offset/=2)
        sum += __shfl_xor_sync(__activemask(), sum, offset, width);
    #pragma unroll
    for (int i=0; i<iterations; ++i) {{
        int col = lane + i * width;
        if (col < length) dx[base+col] = out0_type(__fmaf_rn(
            -({int(log)} ? expf(out[i]) : out[i]), sum, product[i]));
    }}
}}
int rows = in0->numel() / {length};
if (rows) kernel<<<(rows * {width} + 127) / 128, 128>>>(in0_p, in1_p, out0_p, rows);
CHECK(0 == cudaGetLastError());
''')

        def grad(self, direction):
            with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
                y, g, d = self.y.float32(), self.seed.float32(), direction.float32()
                if log:
                    probability = y.exp()
                    dy = -d * probability * g.sum([-1], keepdims=True)
                    dg = d - (d * probability).sum([-1], keepdims=True)
                else:
                    dy = (d * g - d * (y * g).sum([-1], keepdims=True)
                          - g * (d * y).sum([-1], keepdims=True))
                    dg = y * (d - (d * y).sum([-1], keepdims=True))
                return dy.cast(self.y.dtype), dg.cast(self.seed.dtype)

    class WarpSoftmax(jt.Function):
        def _run_call(self, value):
            self.original = jt.ops.reshape(value, value.shape)
            return super()._run_call(value)

        def execute(self, value):
            self.output = jt.code(value.shape, value.dtype, [value],
                cuda_header=header, cuda_src=f'''
__global__ void kernel(in0_type* x, out0_type* y, int rows) {{
{geometry}
    float v[iterations];
    float maximum = -INFINITY;
    #pragma unroll
    for (int i=0; i<iterations; ++i) {{
        int col = lane + i * width;
        v[i] = col < length ? float(x[base+col]) : -INFINITY;
        maximum = maximum > v[i] ? maximum : v[i];
    }}
    #pragma unroll
    for (int offset=width/2; offset; offset/=2) {{
        float other = __shfl_xor_sync(__activemask(), maximum, offset, width);
        maximum = maximum < other ? other : maximum;
    }}
    float sum = 0.0f;
    #pragma unroll
    for (int i=0; i<iterations; ++i) {{
        float e = ({int(zero_all_neg_inf and not log)} && maximum == -INFINITY)
            ? 0.0f : expf(v[i] - maximum);
        sum += e;
        if (!{int(log)}) v[i] = e;
    }}
    #pragma unroll
    for (int offset=width/2; offset; offset/=2)
        sum += __shfl_xor_sync(__activemask(), sum, offset, width);
    float logsum = {int(log)} ? logf(sum) : 0.0f;
    #pragma unroll
    for (int i=0; i<iterations; ++i) {{
        int col = lane + i * width;
        if (col < length) y[base+col] = out0_type(
            ({int(zero_all_neg_inf and not log)} && maximum == -INFINITY) ? 0.0f :
            {int(log)} ? (v[i] - maximum) - logsum : v[i] / sum);
    }}
}}
{launch}''')
            return self.output

        def grad(self, seed):
            # Rebuild on the original edge so higher derivatives remain
            # connected through the closed-form VJP.
            return WarpSoftmaxVJP()(WarpSoftmax()(self.original), seed)
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
            narrow = _jittor_dtype_name(x.dtype) in ("float16", "bfloat16")
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
                ("in0_type(float(vy[i][j]) - expf(float(vx[i][j])) * reduce_var);" if narrow
                 else "vy[i][j] - in0_type(expf(vx[i][j]) * reduce_var);") if log
                else ("in0_type(float(in0_type(vx[i][j] * vy[i][j])) - float(vx[i][j]) * reduce_var);" if narrow
                      else "vx[i][j] * (vy[i][j] - in0_type(reduce_var));")
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
{_streaming_row_reduction(threads, log, zero_all_neg_inf)}

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
    if length <= 1024 and a.numel() <= 2147483647 and a._storage_is_contiguous():
        return _softmax_warp_cls(length, bool(log), bool(zero_all_neg_inf))()(a)
    kind, threads = _softmax_schedule(length)
    cls = (_softmax_v1_cls(length, bool(log), bool(zero_all_neg_inf))
           if kind == "register" else
           _softmax_streaming_cls(threads, bool(log), bool(zero_all_neg_inf)))
    return cls()(a)


def softmax_v1(a, log=False, zero_all_neg_inf=False):
    return _softmax_v1(a, log=log, zero_all_neg_inf=zero_all_neg_inf)


def _supports_cross_entropy_lse(x):
    # A backend candidate is selected from actual tensor residency, not the
    # ambient use_cuda flag. Keep portable FP64/strided/empty behavior intact.
    # The shared streaming body has int row offsets; larger inputs retain the
    # generic index-width-aware path rather than overflow that offset.
    return (x.ndim == 2 and int(x.shape[0]) > 0 and int(x.shape[1]) > 0
            and x.numel() <= 2147483647 and x._storage_is_contiguous())


@optional_kernel("nn.cross_entropy_lse", "cuda", dtypes=("float32",),
                 supports=_supports_cross_entropy_lse)
def _cross_entropy_lse(x):
    """Return only [rows] logsumexp, without a vocabulary-sized temporary.

    The generic exp-sum may distribute one row over blocks and atomically add
    their partials in a varying order. CE saves that result for backward, so
    even equal rounded losses could have different gradients. One CTA and a
    fixed CUB reduction tree per row preserve bounded memory and repeatability.
    """
    threads = 512
    return jt.code((int(x.shape[0]),), x.dtype, [x], cuda_header=f'''
#include <{library_resource('cub', 'home')}cub/cub.cuh>
#include <type/fp16_compute.h>
''', cuda_src=f'''
__global__ void kernel(in0_type* x, out0_type* y, int len) {{
{_streaming_row_reduction(threads, log=True)}
    if (threadIdx.x == 0) y[blockIdx.x] = row_max + logf(row_sum);
}}
int rows = in0->shape[0];
int len = in0->shape[1];
kernel<<<rows, {threads}>>>(in0_p, out0_p, len);
CHECK(0 == cudaGetLastError());
''')


@optional_kernel("nn.torch_logsumexp_last", "cuda", dtypes=("float32",),
                 supports=_supports_cross_entropy_lse)
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
    const float maximum = FloatReduce(float_storage).Reduce(local_max, cub::Max());
    const int has_nan = IntReduce(int_storage).Reduce(local_nan, cub::Max());
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


@optional_kernel("nn.cross_entropy_row_statistics", "cuda", dtypes=("float32",),
                 supports=lambda x: _supports_cross_entropy_lse(x) and int(x.shape[1]) > 1024)
def _cross_entropy_row_statistics(x):
 """Keep maximum and log-sum separate, using bounded per-row storage."""
 return jt.code((x.shape[0],2),'float32',[x],cuda_header='#include <cfloat>\n#include <cstdint>\n', cuda_src=r'''
__device__ float ce_tree_reduce(float v,float* shared,bool ismax){
 int lane=threadIdx.x%32,warp=threadIdx.x/32;
 #pragma unroll
 for(int o=16;o;o/=2){float b=__shfl_down_sync(0xffffffff,v,o);v=ismax?(v<b?b:v):v+b;}
 __syncthreads();if(lane==0)shared[warp]=v;__syncthreads();
 v=threadIdx.x<32?shared[lane]:(ismax?-INFINITY:0.f);
 if(warp==0){
 #pragma unroll
 for(int o=16;o;o/=2){float b=__shfl_down_sync(0xffffffff,v,o);v=ismax?(v<b?b:v):v+b;}}
 if(threadIdx.x==0)shared[0]=v;__syncthreads();return shared[0];
}
__global__ void kernel(in0_type* x,out0_type* y,int n){
 __shared__ float shared[32];int row=blockIdx.x;int tid=threadIdx.x;int shift=(reinterpret_cast<uintptr_t>(x+row*n)%16)/sizeof(float);
 int size=n,base=row*n;float top=-FLT_MAX;
 if(shift){if(tid>=shift&&tid<n+shift)top=fmaxf(top,x[base+tid-shift]);base+=1024-shift;size-=1024-shift;}
 int tail=size%(4*1024);
 for(int offset=tid;offset*4<size-tail;offset+=1024){
  #pragma unroll
  for(int j=0;j<4;j++)top=fmaxf(top,x[base+offset*4+j]);
 }
 for(int col=size-tail+tid;col<size;col+=1024)top=fmaxf(top,x[base+col]);
 top=ce_tree_reduce(top,shared,true);float sum=0.f;
 if(shift&&tid>=shift&&tid<n+shift)sum+=expf(x[row*n+tid-shift]-top);
 for(int offset=tid;offset*4<size-tail;offset+=1024){
  #pragma unroll
  for(int j=0;j<4;j++)sum+=expf(x[base+offset*4+j]-top);
 }
 for(int col=size-tail+tid;col<size;col+=1024)sum+=expf(x[base+col]-top);
 sum=ce_tree_reduce(sum,shared,false);if(tid==0){y[row*2]=top;y[row*2+1]=logf(sum);}
}
kernel<<<in0->shape[0],1024>>>(in0_p,out0_p,in0->shape[1]);CHECK(0==cudaGetLastError());
''')


def _cross_entropy_row_gradient(x, st, safe, sc):
 """Apply the log-softmax/NLL VJP with one final output rounding."""
 return jt.code(x.shape,x.dtype,[x,st,safe,sc],cuda_src=r'''
__global__ void kernel(in0_type* x,in1_type* stats,in2_type* target,in3_type* scales,out0_type* y,int n,int total){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;int row=i/n;int col=i%n;
 float logp=(x[i]-stats[row*2])-stats[row*2+1];float s=scales[row];
 y[i]=fmaf(-expf(logp),-s,col==target[row]?-s:0.f);
}
kernel<<<(in0->numel()+255)/256,256>>>(in0_p,in1_p,in2_p,in3_p,out0_p,in0->shape[1],in0->numel());CHECK(0==cudaGetLastError());
''')
