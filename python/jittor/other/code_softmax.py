import jittor as jt
from jittor import nn

def can_softmax_v1(a, dim):
    if not jt.flags.use_cuda:
        return False
    if dim != -1 and dim != len(a.shape)-1:
        return False
    if a.shape[len(a.shape)-1] > 10000:
        return False
    return True

def softmax_v1(a):
    assert can_softmax_v1(a, -1)
    length = a.shape[-1]
    # tnum = 1024
    tnum = 500 if length % 500 == 0 else 512
    # tnum = 250
    per_thread = (length-1) // tnum + 1
    for_loop = f"""
    #pragma unroll
    for (int i=0; i<{per_thread}; i++)
    """
    if length % tnum == 0:
        for_loop += f"if (i*{tnum}+threadIdx.x < len)\n"

    return jt.code(a.shape, a.dtype, [a], cuda_header=f'''
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
''', cuda_src=f'''
__global__ void kernel(in0_type* x, out0_type* y, int len) {{
    typedef cub::BlockReduce<float, {tnum}> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int id = blockIdx.x * len;
    in0_type v[{per_thread}];
    {for_loop} v[i] = x[id+i*{tnum}+threadIdx.x];
    float v1 = v[0];
    {for_loop} v1 = max(v1, v[i]);
    __shared__ float vmax;
    auto tmp = BlockReduce(temp_storage).Reduce(v1, cub::Max());
    if (threadIdx.x == 0)
        vmax = tmp;
    __syncthreads();

    v1 = 0;
    {for_loop} {{
        v[i] = expf(v[i] - vmax);
        v1 += v[i];
    }}

    tmp = BlockReduce(temp_storage).Sum(v1);
    __shared__ float vsum;
    if (threadIdx.x == 0)
        vsum = tmp;
    __syncthreads();

    {for_loop} y[id+i*{tnum}+threadIdx.x] = v[i] / vsum;
}}
int len = in0->shape[in0->shape.size()-1];
int bnum = in0->numel() / len;
cudaGetLastError();
kernel<<<bnum, {tnum}>>>(in0_p, out0_p, len);
CHECK(0 == cudaGetLastError());
''', cuda_grad_src=[f"""
__global__ void kernel(pout0_type* x, dout_type* y, out0_type* z, int len) {{
    int id = blockIdx.x * len;
    in0_type vx[{per_thread}];
    in0_type vy[{per_thread}];
    {for_loop} {{
        vx[i] = x[id+i*{tnum}+threadIdx.x];
        vy[i] = y[id+i*{tnum}+threadIdx.x];
    }}
    float v1 = 0;
    {for_loop} v1 += vx[i]*vy[i];


    typedef cub::BlockReduce<float, {tnum}> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    auto tmp = BlockReduce(temp_storage).Sum(v1);
    __shared__ float reduce_var;
    if (threadIdx.x == 0)
        reduce_var = tmp;
    __syncthreads();

    {for_loop}
        z[id+i*{tnum}+threadIdx.x] = vx[i] * (vy[i] - reduce_var);
}}
int len = in0->shape[in0->shape.size()-1];
int bnum = in0->numel() / len;
cudaGetLastError();
kernel<<<bnum, {tnum}>>>(pout0_p, dout_p, out0_p, len);
CHECK(0 == cudaGetLastError());
"""])