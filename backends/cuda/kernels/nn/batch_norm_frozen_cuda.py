"""cuDNN batch statistics for FP32 training with no autograd consumers."""

from pathlib import Path

import jittor as jt
from jittor._runtime.backend_libraries import get_library_ops
from jittor._runtime.dispatch import dispatch_context


_DESCRIPTOR = (Path(__file__).resolve().parents[2] /
               "libraries/cudnn/include/cudnn_descriptor.h")
_HEADER = '#include "' + str(_DESCRIPTOR) + '"\n' + r'''
__global__ static void jt_bn_biased_variance(float* variance, int channels,
                                           long long count) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel < channels)
        variance[channel] *= float(count - 1) / float(count);
}
'''


def _frozen_batch_norm_statistics(x, weight, bias, eps):
    """Return output and biased moments, or decline unsupported geometry.

    The caller has established that none of these inputs requires autograd.
    Running-buffer updates remain owned by the public normalization function.
    """
    count = int(x.shape[0]) * int(x.shape[2]) * int(x.shape[3])
    if (dispatch_context(x).backend != "cuda" or eps < 1e-5 or count <= 1
            or not x._storage_is_contiguous()
            or get_library_ops("cudnn", load=True) is None):
        return None
    channels = int(x.shape[1])
    output, mean, variance, saved_mean, invstd = jt.code(
        [x.shape, (channels,), (channels,), (channels,), (channels,)],
        ["float32"] * 5, [x, weight, bias], cuda_header=_HEADER,
        cuda_src=r'''
        jittor::CudnnTensorDescriptor input, normalization;
        checkCudaErrors(cudnnSetTensor4dDescriptor(
            input, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            in0_shape0, in0_shape1, in0_shape2, in0_shape3));
        checkCudaErrors(cudnnDeriveBNTensorDescriptor(
            normalization, input, CUDNN_BATCHNORM_SPATIAL));
        float alpha = 1.f, beta = 0.f;
        // Even a factor of one can read previous running moments. Scratch
        // may contain NaNs from a retired allocation, so initialize it.
        checkCudaErrors(cudaMemsetAsync(
            out1_p, 0, in0_shape1 * sizeof(float), cudaStreamPerThread));
        checkCudaErrors(cudaMemsetAsync(
            out2_p, 0, in0_shape1 * sizeof(float), cudaStreamPerThread));
        checkCudaErrors(cudnnBatchNormalizationForwardTraining(
            jittor::cudnn_bind_stream(), CUDNN_BATCHNORM_SPATIAL,
            &alpha, &beta, input, in0_p, input, out0_p, normalization,
            in1_p, in2_p, 1.0, out1_p, out2_p, EPSILON, out3_p, out4_p));
        jt_bn_biased_variance<<<(in0_shape1 + 255) / 256, 256>>>(
            out2_p, in0_shape1, (long long)in0_shape0 * in0_shape2 * in0_shape3);
        '''.replace("EPSILON", repr(float(eps))))
    return output.stop_grad(), mean.stop_grad(), variance.stop_grad()
