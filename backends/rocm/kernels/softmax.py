"""Native HIP FP32 last-axis inference softmax."""

from functools import lru_cache

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs


def supported(x, log=False, zero_all_neg_inf=False, dim=-1):
    if not isinstance(x, jt.Var) or not x.ndim or not isinstance(dim, int):
        return False
    if dim not in (-1, x.ndim - 1) or log or zero_all_neg_inf:
        return False
    if dtype_name(x.dtype) != 'float32' or jt.flags.amp_reg or _output_requires_grad(x):
        return False
    width = x.shape[-1]
    return 0 < width <= 8192 and 0 <= x.numel() // width <= 2147483647


# HIP supports 32-lane shuffle subgroups on both wave32 and wave64 devices.
_HEADER = r'''
#include <hip/hip_runtime.h>
__device__ static float hip_softmax_group_max(float value) {
    for (int offset=16; offset; offset>>=1)
        value=fmaxf(value,__shfl_down(value,offset,32));
    return value;
}
__device__ static float hip_softmax_group_sum(float value) {
    for (int offset=16; offset; offset>>=1)
        value+=__shfl_down(value,offset,32);
    return value;
}
'''


@lru_cache(maxsize=128)
def _small_source(items):
    # Eight independent short rows per block avoid a block-wide reduction.
    return r'''
 __global__ static void hip_softmax_small(const float* x,float* y,int width,long long rows) {
    const int lane=threadIdx.x%%32;
    const long long row=(long long)blockIdx.x*8+threadIdx.x/32;
    if(row>=rows)return;
    const long long base=row*width;
    float values[%(items)d];
    float maxval=-INFINITY;
    #pragma unroll
    for(int i=0;i<%(items)d;i++) {
        const int col=lane+i*32;
        float v=col<width?x[base+col]:-INFINITY;
        values[i]=v;maxval=fmaxf(maxval,v);
    }
    const float maximum=__shfl(hip_softmax_group_max(maxval),0,32);
    float total=0.f;
    #pragma unroll
    for(int i=0;i<%(items)d;i++) {values[i]=expf(values[i]-maximum);total+=values[i];}
    const float denom=__shfl(hip_softmax_group_sum(total),0,32);
    #pragma unroll
    for(int i=0;i<%(items)d;i++) {
        const int col=lane+i*32;
        if(col<width)y[base+col]=values[i]/denom;
    }
 }
 const int width=in0->shape[in0->shape.size()-1];
 const long long rows=in0->num/width;
 if(rows) {hip_softmax_small<<<(rows+7)/8,256>>>(in0_p,out0_p,width,rows);CHECK(hipGetLastError()==hipSuccess);}
 ''' % {'items': items}


@lru_cache(maxsize=128)
def _source(items):
    # Long rows use all 256 threads and reduce each 32-lane subgroup first.
    return r'''
 __global__ static void hip_softmax(const float* x,float* y,int width) {
    const long long base=(long long)blockIdx.x*width;
    const int tid=threadIdx.x,lane=tid%%32,group=tid/32;
    __shared__ float maxima[8],sums[8];
    float values[%(items)d];
    float maxval=-INFINITY;
    #pragma unroll
    for(int i=0;i<%(items)d;i++) {
        const int col=tid+i*256;
        float v=col<width?x[base+col]:-INFINITY;
        values[i]=v;maxval=fmaxf(maxval,v);
    }
    maxval=hip_softmax_group_max(maxval);
    if(lane==0)maxima[group]=maxval;
    __syncthreads();
    if(group==0) {
        float m=hip_softmax_group_max(lane<8?maxima[lane]:-INFINITY);
        if(lane==0)maxima[0]=m;
    }
    __syncthreads();
    const float maximum=maxima[0];
    float total=0.f;
    #pragma unroll
    for(int i=0;i<%(items)d;i++) {values[i]=expf(values[i]-maximum);total+=values[i];}
    total=hip_softmax_group_sum(total);
    if(lane==0)sums[group]=total;
    __syncthreads();
    if(group==0) {
        float s=hip_softmax_group_sum(lane<8?sums[lane]:0.f);
        if(lane==0)sums[0]=s;
    }
    __syncthreads();
    const float denom=sums[0];
    #pragma unroll
    for(int i=0;i<%(items)d;i++) {
        const int col=tid+i*256;
        if(col<width)y[base+col]=values[i]/denom;
    }
 }
 const int width=in0->shape[in0->shape.size()-1];
 const long long rows=in0->num/width;
 if(rows) {hip_softmax<<<rows,256>>>(in0_p,out0_p,width);CHECK(hipGetLastError()==hipSuccess);}
 ''' % {'items': items}


def softmax(x, log=False, zero_all_neg_inf=False, dim=-1):
    if not supported(x, log=log, zero_all_neg_inf=zero_all_neg_inf, dim=dim):
        return None
    width = int(x.shape[-1])
    source = (_small_source((width + 31) // 32) if width <= 256
              else _source((width + 255) // 256))
    return _stop_grad_outputs(jt.code(x.shape, x.dtype, [x], cuda_header=_HEADER, cuda_src=source))
