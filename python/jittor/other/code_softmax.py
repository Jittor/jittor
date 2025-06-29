import jittor as jt
from jittor import nn
import numpy as np

def can_softmax_v1(a, dim):
    if not jt.flags.use_cuda:
        return False
    if dim != -1 and dim != len(a.shape)-1:
        return False
    if a.shape[-1] > 10000 and np.prod(a.shape[:-1]) < 64:
        return False
    return True

def softmax_v1(a, log=False):
    assert can_softmax_v1(a, -1)
    length = a.shape[-1]

    if length < 65536:
        tnum = 250 if length % 250 == 0 else 256
    else:
        tnum = 125 if length % 125 == 0 else 128

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
#include <helper_cuda.h>
''', cuda_src=f'''
__global__ void kernel(in0_type* x, out0_type* y, int len) {{
    typedef cub::BlockReduce<float, {tnum}> BlockReduce;
    constexpr int need_log = {int(log)};
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int id = blockIdx.x * len;
    in0_type v[{per_thread}][{ILP}];
    {for_loop}
        vload<sizeof(in0_type)*{ILP}>(v[i], &x[id+(i*{tnum}+threadIdx.x)*{ILP}]);
    // v[i] = x[id+i*{tnum}+threadIdx.x];
    float v1 = -1e30;
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
            if (need_log) {{
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
            if (need_log)
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
getLastCudaError("Failed to run CodeSoftmax forward");
''')
            return self.save_vars

        def grad(self, grad_x):
            x = self.save_vars
            return jt.code(x.shape, x.dtype, [x, grad_x], cuda_header=f'''
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
#include <type/fp16_compute.h>
#include <helper_cuda.h>
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
getLastCudaError("Failed to run CodeSoftmax backward");
""")
    return CodeSoftmax()(a)