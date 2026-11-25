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
    return 0 < width <= 4096 and 0 <= x.numel() // width <= 2147483647


@lru_cache(maxsize=128)
def _source(items):
 return r'''
 __global__ static void hip_softmax(const float* x,float* y,int width) {
  const long long base=(long long)blockIdx.x*width;
  const int tid=threadIdx.x;
  __shared__ float scratch[256];
  float values[%(items)d];
  float maxval=-INFINITY;
  #pragma unroll
  for(int i=0;i<%(items)d;i++) {
   const int col=tid+i*256;
   float v=col<width?x[base+col]:-INFINITY;
   values[i]=v;maxval=fmaxf(maxval,v);
  }
  scratch[tid]=maxval;__syncthreads();
  for(int stride=128;stride;stride>>=1) {
   if(tid<stride)scratch[tid]=fmaxf(scratch[tid],scratch[tid+stride]);
   __syncthreads();
  }
  const float maximum=scratch[0];__syncthreads();
  float total=0.f;
  #pragma unroll
  for(int i=0;i<%(items)d;i++) {values[i]=expf(values[i]-maximum);total+=values[i];}
  scratch[tid]=total;__syncthreads();
  for(int stride=128;stride;stride>>=1) {
   if(tid<stride)scratch[tid]+=scratch[tid+stride];
   __syncthreads();
  }
  const float denom=scratch[0];
  #pragma unroll
  for(int i=0;i<%(items)d;i++) {
   const int col=tid+i*256;
   if(col<width)y[base+col]=values[i]/denom;
  }
 }
 const int width=in0->shape[in0->shape.size()-1];
 const long long rows=in0->num/width;
 if(rows) {hip_softmax<<<rows,256>>>(in0_p,out0_p,width);CHECK(hipGetLastError()==hipSuccess);}
 ''' % {'items':items}


def softmax(x, log=False, zero_all_neg_inf=False, dim=-1):
    if not supported(x, log=log, zero_all_neg_inf=zero_all_neg_inf, dim=dim):
        return None
    return _stop_grad_outputs(jt.code(x.shape, x.dtype, [x],
        cuda_header='#include <hip/hip_runtime.h>',
        cuda_src=_source((int(x.shape[-1]) + 255) // 256)))
