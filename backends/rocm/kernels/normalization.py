"""Native HIP FP32 inference group and layer normalization."""
from functools import lru_cache
import math
import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
from jittor.nn.functional.activation import offer_activation


_HEADER = r'''#include <hip/hip_runtime.h>
    template<class T> __device__ static T hip_norm_sum(T value,T* scratch) {
        int tid=threadIdx.x;
        scratch[tid]=value;__syncthreads();
        for(int s=128;s;s>>=1) {
            if(tid<s)scratch[tid]+=scratch[tid+s];
            __syncthreads();
        }
        value=scratch[0];__syncthreads();return value;
    }
'''

def _supported(x, weight, bias, eps):
    values = (x, weight, bias)
    return (isinstance(x, jt.Var) and dtype_name(x.dtype) == 'float32'
            and not jt.flags.amp_reg and not _output_requires_grad(*values)
            and all(dtype_name(v.dtype) == 'float32' if isinstance(v, jt.Var)
                    else type(v) in (int, float) and math.isfinite(v)
                    for v in (weight, bias))
            and type(eps) in (int, float) and math.isfinite(eps) and eps > 0)


def supports_group_norm(x, num_groups, weight=1, bias=0, eps=1e-5):
    if not _supported(x, weight, bias, eps) or x.ndim < 2:
        return False
    if type(num_groups) is not int or num_groups <= 0 or x.shape[1] % num_groups:
        return False
    width = math.prod(x.shape[1:]) // num_groups
    return (0 < width <= 1048576 and x.shape[0] * num_groups <= 2147483647
            and all(not isinstance(v, jt.Var) or tuple(v.shape) == (x.shape[1],)
                    for v in (weight, bias)))


def supports_layer_norm(x, normalized_shape, weight=1, bias=0, eps=1e-5, **kwargs):
    if not _supported(x, weight, bias, eps):
        return False
    shape = tuple(normalized_shape)
    width = math.prod(shape)
    return (0 < len(shape) <= x.ndim and tuple(x.shape[-len(shape):]) == shape
            and 0 < width <= 1048576 and x.numel() // width <= 2147483647
            and all(not isinstance(v, jt.Var) or tuple(v.shape) == shape
                    for v in (weight, bias)))


@lru_cache(maxsize=128)
def _source(kind, extent, eps, scale, shift, act):
    # Runtime dimensions keep model shapes on the same compiled kernel.
    if kind == 'group':
        dimensions = f'''long long spatial=1;
        for (int d=2;d<in0->shape.size();d++) spatial*=in0->shape[d];
        long long channels=in0->shape[1],width=channels/{extent}*spatial;
        long long rows=in0->shape[0]*{extent};'''
        affine_index = '(base+j)/spatial%channels'
    else:
        dimensions = f'''long long width=1;
        for (int d=in0->shape.size()-{extent};d<in0->shape.size();d++) width*=in0->shape[d];
        long long rows=in0->num/width,spatial=1,channels=1;'''
        affine_index = 'j'
    args = ''
    pointers = 'in0_p,out0_p'
    affine = ''
    for name,value in [('scale', scale), ('shift', shift)]:
        if value == 'tensor':
            args += f',const float* {name}'
            pointers += ',in'+str(1 + (name=='shift' and scale=='tensor'))+'_p'
            affine += f'float {name}_value={name}[{affine_index}];'
        else:
            affine += f'float {name}_value={float(value):.9e}f;'
    apply = 'value = value * scale_value + shift_value;'
    if act == 'silu':
        apply += 'value = value / (1.f + expf(-value));'
    return r'''
    __global__ static void hip_norm(const float* x,float* y%(args)s,
            long long width,long long spatial,long long channels) {
        const long long base=(long long)blockIdx.x*width;
        const int tid=threadIdx.x;
        __shared__ float scratch[256];
        // Subtract an anchor before summing: a large common offset must not
        // round away a small variance, nor round the mean back onto the input.
        float anchor=x[base],sum=0.f;
        for(long long j=tid;j<width;j+=256)sum+=x[base+j]-anchor;
        float mean=hip_norm_sum(sum,scratch)/width;
        float variance=0.f;
        for(long long j=tid;j<width;j+=256) {
            float d=(x[base+j]-anchor)-mean;variance+=d*d;
        }
        variance=hip_norm_sum(variance,scratch)/width;
        if(isfinite(mean)&&isfinite(variance)) {
            float inverse=rsqrtf(variance+%(eps).9ef);
            for(long long j=tid;j<width;j+=256) {
                float value=((x[base+j]-anchor)-mean)*inverse;
                %(affine)s
                %(apply)s
                y[base+j]=value;
            }
        } else {
            // Scale exceptional rows before the reductions: even opposite
            // finite FP32 extremes must normalize without overflowing. This
            // uses FP32 throughout and needs no backend FP64 capability.
            float magnitude=0.f;
            for(long long j=tid;j<width;j+=256)magnitude=fmaxf(magnitude,fabsf(x[base+j]));
            scratch[tid]=magnitude;__syncthreads();
            for(int s=128;s;s>>=1) {
                if(tid<s)scratch[tid]=fmaxf(scratch[tid],scratch[tid+s]);
                __syncthreads();
            }
            magnitude=scratch[0];__syncthreads();
            float sum=0.f;
            for(long long j=tid;j<width;j+=256)sum+=x[base+j]/magnitude;
            float m=hip_norm_sum(sum,scratch)/width,v=0.f;
            for(long long j=tid;j<width;j+=256) {
                float d=x[base+j]/magnitude-m;v+=d*d;
            }
            float inverse=rsqrtf(hip_norm_sum(v,scratch)/width+(%(eps).9ef/magnitude)/magnitude);
            for(long long j=tid;j<width;j+=256) {
                float value=(x[base+j]/magnitude-m)*inverse;
                %(affine)s
                %(apply)s
                y[base+j]=value;
            }
        }
    }
    %(dimensions)s
    if(rows) {
        hip_norm<<<rows,256>>>(%(pointers)s,width,spatial,channels);
        CHECK(hipGetLastError()==hipSuccess);
    }
    ''' % dict(args=args,eps=eps,affine=affine,apply=apply,
               dimensions=dimensions,pointers=pointers)


def _normalize(x,weight,bias,eps,kind,extent,act=''):
    inputs=[x]
    spec=[]
    for value in (weight,bias):
        if isinstance(value,jt.Var):
            spec.append('tensor');inputs.append(value)
        else:spec.append(float(value))
    return _stop_grad_outputs(jt.code(x.shape,x.dtype,inputs,
        cuda_header=_HEADER,
        cuda_src=_source(kind,extent,float(eps),*spec,act)))


def group_norm(x,num_groups,weight=1,bias=0,eps=1e-5):
    if not supports_group_norm(x,num_groups,weight,bias,eps):return None
    def build(act):
        if act not in ('','silu'):return None
        return _normalize(x,weight,bias,eps,'group',num_groups,act)
    y=build('')
    offer_activation(y,build)
    return y


def layer_norm(x,normalized_shape,weight=1,bias=0,eps=1e-5,**kwargs):
    if not supports_layer_norm(x,normalized_shape,weight,bias,eps):return None
    return _normalize(x,weight,bias,eps,'layer',len(normalized_shape))
