import jittor as jt
from jittor import nn
import numpy as np


def cross_entropy(output, target):
    tnum = min(512, output.shape[-1])

    class CodeCrossEntropy(jt.Function):
        def execute(self, x, target):
            self.save_vars = [x, target]
            cross_entropy = jt.code(target.shape, x.dtype, [x, target], cuda_header=f'''
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
#include <type/fp16_compute.h>
#include <helper_cuda.h>
''', cuda_src=f'''
__global__ void kernel(in0_type* x, in1_type* target, out0_type* y, size_t len) {{
    typedef cub::BlockReduce<float, {tnum}> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t id = blockIdx.x * len;

    float v1 = -1e30;
    for (size_t i = threadIdx.x; i < len; i += blockDim.x)
        v1 = ::max(v1, float(x[id + i]));

    __shared__ float vmax;
    auto tmp = BlockReduce(temp_storage).Reduce(v1, cub::Max());
    if (threadIdx.x == 0)
        vmax = tmp;
    __syncthreads();

    v1 = 0;
    for (size_t i = threadIdx.x; i < len; i += blockDim.x)
        v1 += expf(float(float(x[id + i]) - vmax));

    auto vsum = BlockReduce(temp_storage).Sum(v1);
    if (threadIdx.x == 0)
        y[blockIdx.x] = -float(x[id+target[blockIdx.x]]) + vmax + float(@expand_op(log,@in0_type,vsum));
}}
size_t len = in0->shape[in0->shape.size()-1];
size_t bnum = in0->numel() / len;
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
__global__ void kernel(in0_type* x, in1_type* target, in2_type* grad, out0_type* y, size_t len) {{
    typedef cub::BlockReduce<float, {tnum}> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    size_t id = blockIdx.x * len;

    float v1 = -1e30;
    for (size_t i = threadIdx.x; i < len; i += blockDim.x)
        v1 = ::max(v1, float(x[id + i]));
    __shared__ float vmax;
    auto tmp = BlockReduce(temp_storage).Reduce(v1, cub::Max());
    if (threadIdx.x == 0)
        vmax = tmp;
    __syncthreads();

    v1 = 0;
    for (size_t i = threadIdx.x; i < len; i += blockDim.x) {{
        y[id + i] = expf(float(x[id + i]) - vmax);
        v1 += float(y[id + i]);
    }}

    tmp = BlockReduce(temp_storage).Sum(v1);
    __shared__ float vsum;
    if (threadIdx.x == 0)
        vsum = tmp;
    __syncthreads();

    for (size_t i = threadIdx.x; i < len; i += blockDim.x)
        y[id + i] = float(y[id + i]) / vsum * float(grad[blockIdx.x]);
    __syncthreads();

    if (threadIdx.x == 0)
        y[id + target[blockIdx.x]] -= grad[blockIdx.x];
}}
size_t len = in0->shape[in0->shape.size()-1];
size_t bnum = in0->numel() / len;
cudaGetLastError();
kernel<<<bnum, {tnum}>>>(in0_p, in1_p, in2_p, out0_p, len);
getLastCudaError("Failed to run CodeCrossEntropy backward");
''')
    return CodeCrossEntropy()(output, target)
