"""CUDA launch for the shared incomplete-gamma implementation."""

import jittor as jt


_IGAMMA_KERNEL = '''__global__ void igamma_kernel(float* __restrict__ x,
                float* out,
                float alpha,
                int batch_shape)
{
    int tidx = threadIdx.x;
    int start = batch_shape / blockDim.x * tidx;
    int end = threadIdx.x == blockDim.x - 1 ? batch_shape : start + batch_shape / blockDim.x;
    float* bx = x+batch_shape*blockIdx.x;
    float* bout = out + batch_shape * blockIdx.x;
    for(int i=start;i<end;i++)
        bout[i] = calc_igamma(alpha, bx[i]);
}
'''


def igamma(alpha, x, shared_header):
    cuda_header = "#define C10_DEVICE __host__ __device__\n" + shared_header + _IGAMMA_KERNEL
    cuda_src = '''
        @alias(x, in0)
        @alias(px ,out0)
        int batch_size = x_stride0 == 1 ? 1 : x_shape0;
        int batch_shape = x_shape0 * x_stride0 / batch_size;
        float alpha = data["alpha"];
        igamma_kernel<<<batch_size, 16>>>(x_p, px_p, alpha, batch_shape);
    '''
    out = jt.code(x.shape, x.dtype, [x], cuda_header=cuda_header, cuda_src=cuda_src, data={"alpha": alpha})
    return out
