/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#include "helper_cuda.h"
#include "fp16_dev.h"

#define BLOCK_SIZE 128
template <class value_type>
__global__ void float2half_rn_kernel(int size, const value_type *buffIn, half1 *buffOut)
{
    const int idx = BLOCK_SIZE*blockIdx.x+threadIdx.x;
    if (idx >= size) {
        return;
    }
#if CUDART_VERSION < 9000
    half1 val;
    val.x = __float2half_rn(float(buffIn[idx]));
#else
    half1 val = __float2half_rn(float(buffIn[idx]));
#endif
    buffOut[idx] = val;
}

template <class value_type>
void gpu_float2half_rn(int size, const value_type *buffIn, half1 *buffOut)
{
    int grid_size = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float2half_rn_kernel<value_type><<<grid_size, BLOCK_SIZE>>> (size, buffIn, buffOut);
    checkCudaErrors(cudaDeviceSynchronize());
}

template void gpu_float2half_rn<float> (int, const float*, half1*);
template void gpu_float2half_rn<double> (int, const double*, half1*);

