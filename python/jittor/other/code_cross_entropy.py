import jittor as jt
from jittor import nn
import numpy as np

def can_cross_entropy(a, dim):
    if not jt.flags.use_cuda:
        return False
    if dim != -1 and dim != len(a.shape)-1:
        return False
    if a.shape[-1] > 10000 and np.prod(a.shape[:-1]) < 64:
        return False
    return True

def cross_entropy(output, target):
    assert can_cross_entropy(output, -1)
    length = output.shape[-1]

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

    class CodeCrossEntropy(jt.Function):
        def execute(self, x, target):
            self.save_vars = [x, target]
            cross_entropy = jt.code(target.shape, x.dtype, [x, target], cuda_header=f'''
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
#include <type/fp16_compute.h>
#include <helper_cuda.h>
''', cuda_src=f'''
__global__ void kernel(in0_type* x, in1_type* target, out0_type* y, int len) {{
    typedef cub::BlockReduce<float, {tnum}> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int id = blockIdx.x * len;
    in0_type v[{per_thread}][{ILP}];
    {for_loop}
        vload<sizeof(in0_type)*{ILP}>(v[i], &x[id+(i*{tnum}+threadIdx.x)*{ILP}]);

    float v1 = -1e30;
    {for_loop}
        #pragma unroll
        for (int j=0; j<{ILP}; j++) {{
            v1 = max(v1, float(v[i][j]));
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
            v1 += expf(float(float(v[i][j]) - vmax));
        }}

    auto vsum = BlockReduce(temp_storage).Sum(v1);
    if (threadIdx.x == 0)
        y[blockIdx.x] = -float(x[id+target[blockIdx.x]] - vmax) + @expand_op(log,@in0_type,vsum);
}}
int len = in0->shape[in0->shape.size()-1];
int bnum = in0->numel() / len;
cudaGetLastError();
kernel<<<bnum, {tnum}>>>(in0_p, in1_p, out0_p, len);
getLastCudaError("Failed to run CodeCrossEntropy forward");
''')
            return cross_entropy

        def grad(self, grad):
            x, target = self.save_vars
            return jt.code(x.shape, x.dtype, [x, target, grad], cuda_header=f'''
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
#include <type/fp16_compute.h>
#include <helper_cuda.h>
''', cuda_src=f'''
__global__ void kernel(in0_type* x, in1_type* target, in2_type* grad, out0_type* y, int len) {{
    typedef cub::BlockReduce<float, {tnum}> BlockReduce;
    constexpr int need_log = false;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int id = blockIdx.x * len;
    in0_type v[{per_thread}][{ILP}];
    {for_loop}
        vload<sizeof(in0_type)*{ILP}>(v[i], &x[id+(i*{tnum}+threadIdx.x)*{ILP}]);
    float v1 = -1e30;
    {for_loop}
        #pragma unroll
        for (int j=0; j<{ILP}; j++) {{
            v1 = max(v1, float(v[i][j]));
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
            v[i][j] *= grad[blockIdx.x];
        }}
    {for_loop}
        vload<sizeof(in0_type)*{ILP}>(&y[id+(i*{tnum}+threadIdx.x)*{ILP}], v[i]);
    
    if (threadIdx.x == 0)
        y[id + target[blockIdx.x]] -= grad[blockIdx.x];
}}
int len = in0->shape[in0->shape.size()-1];
int bnum = in0->numel() / len;
cudaGetLastError();
kernel<<<bnum, {tnum}>>>(in0_p, in1_p, in2_p, out0_p, len);
getLastCudaError("Failed to run CodeCrossEntropy backward");
''')
    return CodeCrossEntropy()(output, target)
