"""CUDA softmax kernels owned by the neural-network backend domain."""

import jittor as jt
from jittor import nn
from functools import lru_cache
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
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


def _supports_warp_softmax_f32(a):
    """Bounded dense last-axis input; all other routes keep their old owner."""
    if _jittor_dtype_name(a.dtype) != "float32" or not a.shape:
        return False
    length = int(a.shape[-1])
    total = int(a.numel())
    return (32 <= length <= 1024 and 0 < total <= ((1 << 31) - 1) // 4
            and a._storage_is_contiguous())


@lru_cache(maxsize=32)
def _softmax_warp_f32_cls(length):
    """FP32 persistent-warp fold, with a separately rounded backward product.

    The lane tree follows the installed Torch2.8 PersistentSoftmax.cuh.
    No broad reassociation or fast-math compiler switch is needed.
    """
    if type(length) is not int or not 32 <= length <= 1024:
        raise ValueError("bounded FP32 softmax length required")
    power = 1 << (length - 1).bit_length()
    iterations = power // 32
    batch = 2 if power <= 128 else 1

    class WarpSoftmaxF32(jt.Function):
        def execute(self, x):
            source = r"""
            __global__ static void warp_softmax_f32(
                const float* input, float* output, int rows) {
                const int lane = threadIdx.x;
                const int first = (blockIdx.x * 4 + threadIdx.y) * BATCH;
                #pragma unroll
                for (int b = 0; b < BATCH; ++b) {
                    const int row = first + b;
                    float value[ITERATIONS];
                    #pragma unroll
                    for (int i = 0; i < ITERATIONS; ++i) {
                        const int column = lane + i * 32;
                        value[i] = row < rows && column < LENGTH
                            ? input[row * LENGTH + column] : -INFINITY;
                    }
                    float maximum = value[0];
                    #pragma unroll
                    for (int i = 0; i < ITERATIONS; ++i)
                        maximum = maximum > value[i] ? maximum : value[i];
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        float other = __shfl_xor_sync(0xffffffffu, maximum, offset);
                        maximum = maximum < other ? other : maximum;
                    }
                    float total = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < ITERATIONS; ++i) {
                        value[i] = expf(value[i] - maximum);
                        total = total + value[i];
                    }
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        total = total + __shfl_xor_sync(0xffffffffu, total, offset);
                    if (row < rows) {
                        #pragma unroll
                        for (int i = 0; i < ITERATIONS; ++i) {
                            const int column = lane + i * 32;
                            if (column < LENGTH)
                                output[row * LENGTH + column] = value[i] / total;
                        }
                    }
                }
            }
            const int rows = in0->num / LENGTH;
            warp_softmax_f32<<<(rows + 4 * BATCH - 1) / (4 * BATCH), dim3(32, 4)>>>(
                in0_p, out0_p, rows);
            CHECK(0 == cudaGetLastError());
            """
            source = source.replace("ITERATIONS", str(iterations)).replace(
                "BATCH", str(batch)).replace("LENGTH", str(length))
            self.output = jt.code(x.shape, x.dtype, [x],
                                  cuda_header="#include <math.h>", cuda_src=source)
            return self.output

        def grad(self, gradient):
            if gradient is None:
                return None
            source = r"""
            __global__ static void warp_softmax_f32_grad(
                const float* output, const float* gradient, float* result, int rows) {
                const int lane = threadIdx.x;
                const int first = (blockIdx.x * 4 + threadIdx.y) * BATCH;
                #pragma unroll
                for (int b = 0; b < BATCH; ++b) {
                    const int row = first + b;
                    float y[ITERATIONS], gy[ITERATIONS];
                    float total = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < ITERATIONS; ++i) {
                        const int column = lane + i * 32;
                        const bool valid = row < rows && column < LENGTH;
                        y[i] = valid ? output[row * LENGTH + column] : 0.0f;
                        const float dy = valid ? gradient[row * LENGTH + column] : 0.0f;
                        gy[i] = __fmul_rn(dy, y[i]);
                        total = total + gy[i];
                    }
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        total = total + __shfl_xor_sync(0xffffffffu, total, offset);
                    if (row < rows) {
                        #pragma unroll
                        for (int i = 0; i < ITERATIONS; ++i) {
                            const int column = lane + i * 32;
                            if (column < LENGTH)
                                result[row * LENGTH + column] = __fmaf_rn(-y[i], total, gy[i]);
                        }
                    }
                }
            }
            const int rows = in0->num / LENGTH;
            warp_softmax_f32_grad<<<(rows + 4 * BATCH - 1) / (4 * BATCH), dim3(32, 4)>>>(
                in0_p, in1_p, out0_p, rows);
            CHECK(0 == cudaGetLastError());
            """
            source = source.replace("ITERATIONS", str(iterations)).replace(
                "BATCH", str(batch)).replace("LENGTH", str(length))
            return jt.code(self.output.shape, self.output.dtype, [self.output, gradient],
                           cuda_header="#include <math.h>", cuda_src=source)

    return WarpSoftmaxF32


def _supports_large_softmax_f32(x):
    if not x.shape or _jittor_dtype_name(x.dtype) != 'float32':
        return False
    columns=int(x.shape[-1]);total=int(x.numel())
    return (131072 <= columns <= 262144 and columns % 4 == 0
            and 0 < total <= ((1 << 31)-1)//4
            and 1 <= total//columns <= 64 and x._storage_is_contiguous()
            and x._storage_offset() % 4 == 0)


@lru_cache(maxsize=1)
def _softmax_large_f32_cls():
    class LargeSoftmaxF32(jt.Function):
        def execute(self,x):
            if not _supports_large_softmax_f32(x):
                raise ValueError('bounded dense aligned FP32 CUDA row required')
            self.output=jt.code(x.shape,x.dtype,[x],cuda_header='#include <float.h>\n#include <math.h>',cuda_src=r'''
            __global__ static void large_softmax_f32(
                const float* input,float* output,int columns) {
                const int t=threadIdx.x,lane=t&31,warp=t>>5;
                __shared__ float partial[32];
                __shared__ float maximum,total;
                input+=blockIdx.x*columns;output+=blockIdx.x*columns;
                float local=-FLT_MAX;
                const int vectors_end=columns-columns%4096;
                for(int i=t*4;i<vectors_end;i+=4096) {
                    const float4 v=*reinterpret_cast<const float4*>(input+i);
                    local=local < v.x ? v.x : local;
                    local=local < v.y ? v.y : local;
                    local=local < v.z ? v.z : local;
                    local=local < v.w ? v.w : local;
                }
                for(int i=vectors_end+t;i<columns;i+=1024)
                    local=local < input[i] ? input[i] : local;
                for(int shift=16;shift;shift>>=1) {
                    const float other=__shfl_down_sync(0xffffffffu,local,shift);
                    local=local < other ? other : local;
                }
                if(lane==0)partial[warp]=local;
                __syncthreads();
                local=t<32?partial[lane]:-FLT_MAX;
                if(warp==0) {
                    for(int shift=16;shift;shift>>=1) {
                        const float other=__shfl_down_sync(0xffffffffu,local,shift);
                        local=local < other ? other : local;
                    }
                    if(lane==0)maximum=local;
                }
                __syncthreads();
                local=0.0f;
                for(int i=t*4;i<vectors_end;i+=4096) {
                    const float4 v=*reinterpret_cast<const float4*>(input+i);
                    local+=expf(v.x-maximum);local+=expf(v.y-maximum);
                    local+=expf(v.z-maximum);local+=expf(v.w-maximum);
                }
                for(int i=vectors_end+t;i<columns;i+=1024)
                    local+=expf(input[i]-maximum);
                for(int shift=16;shift;shift>>=1)
                    local+=__shfl_down_sync(0xffffffffu,local,shift);
                __syncthreads();
                if(lane==0)partial[warp]=local;
                __syncthreads();
                local=t<32?partial[lane]:0.0f;
                if(warp==0) {
                    for(int shift=16;shift;shift>>=1)
                        local+=__shfl_down_sync(0xffffffffu,local,shift);
                    if(lane==0)total=local;
                }
                __syncthreads();
                for(int i=t;i<columns;i+=1024)
                    output[i]=expf(input[i]-maximum)/total;
            }
            const int columns=in0->shape[in0->shape.size()-1];
            large_softmax_f32<<<in0->num/columns,1024>>>(in0_p,out0_p,columns);
            CHECK(0 == cudaGetLastError());
            ''')
            return self.output

        def grad(self,dy):
            if dy is None:
                return None
            # The native reference rounds dy*output in an independent launch.
            # A separate buffer also prevents an unintended fused dot-product.
            scaled=jt.code(self.output.shape,self.output.dtype,[dy.contiguous(),self.output],cuda_src=r'''
            __global__ static void prescale(const float* dy,const float* y,float* gy,int n) {
                const int i=blockIdx.x*256+threadIdx.x;
                if(i<n)gy[i]=__fmul_rn(dy[i],y[i]);
            }
            prescale<<<(in0->num+255)/256,256>>>(in0_p,in1_p,out0_p,in0->num);
            CHECK(0 == cudaGetLastError());
            ''')
            return jt.code(self.output.shape,self.output.dtype,[self.output,scaled],cuda_src=r'''
            __global__ static void large_softmax_f32_vjp(
                const float* y,const float* gy,float* dx,int columns) {
                const int t=threadIdx.x;
                __shared__ float partial[512];
                y+=blockIdx.x*columns;gy+=blockIdx.x*columns;dx+=blockIdx.x*columns;
                float value=0.0f;
                const int vectors_end=columns-columns%2048;
                for(int i=t*4;i<vectors_end;i+=2048) {
                    const float4 v=*reinterpret_cast<const float4*>(gy+i);
                    value+=v.x;value+=v.y;value+=v.z;value+=v.w;
                }
                for(int i=vectors_end+t;i<columns;i+=512)value+=gy[i];
                partial[t]=value;
                __syncthreads();
                // Large-row backward uses SoftMax.cu blockReduce, not its
                // forward warp-shuffle blockReduceWarp reduction tree.
                if(t<16) {
                    float subtotal=0.0f;
                    for(int j=0;j<32;++j)subtotal+=partial[t*32+j];
                    __syncwarp(0x0000ffffu);
                    partial[t]=subtotal;
                }
                __syncthreads();
                if(t==0) {
                    float total=0.0f;
                    for(int j=0;j<16;++j)total+=partial[j];
                    partial[0]=total;
                }
                __syncthreads();
                for(int i=t;i<columns;i+=512)
                    dx[i]=fmaf(-y[i],partial[0],gy[i]);
            }
            const int columns=in0->shape[in0->shape.size()-1];
            large_softmax_f32_vjp<<<in0->num/columns,512>>>(in0_p,in1_p,out0_p,columns);
            CHECK(0 == cudaGetLastError());
            ''')
    return LargeSoftmaxF32



@optional_kernel("nn.softmax", "cuda", dtypes=("float16", "bfloat16", "float32"),
                 supports=_supports_softmax)
def _softmax_v1(a, log=False, zero_all_neg_inf=False, dim=-1):
    length = int(a.shape[-1])
    if not bool(log) and not bool(zero_all_neg_inf) and _supports_warp_softmax_f32(a):
        return _softmax_warp_f32_cls(length)()(a)
    if not bool(log) and not bool(zero_all_neg_inf) and _supports_large_softmax_f32(a):
        return _softmax_large_f32_cls().apply(a)
    kind, threads = _softmax_schedule(length)
    cls = (_softmax_v1_cls(length, bool(log), bool(zero_all_neg_inf))
           if kind == "register" else
           _softmax_streaming_cls(threads, bool(log), bool(zero_all_neg_inf)))
    return cls()(a)


def softmax_v1(a, log=False, zero_all_neg_inf=False):
    return _softmax_v1(a, log=log, zero_all_neg_inf=zero_all_neg_inf)
