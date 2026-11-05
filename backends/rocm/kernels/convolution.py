"""FP32 ROCm convolution lowered to native hipBLAS GEMM."""

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor.nn.functional._amp import prefers_half


def supported(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
              *, _depthwise_fast_path=True):
    if (groups != 1 or x.ndim != 4 or weight.ndim != 4
            or dtype_name(x.dtype) != 'float32'
            or dtype_name(weight.dtype) != 'float32'
            or (bias is not None and dtype_name(bias.dtype) != 'float32')
            or not weight._storage_is_contiguous() or prefers_half()):
        return False
    from jittor.misc import _pair
    stride, padding, dilation = map(_pair, (stride, padding, dilation))
    if any(size > 2147483647 for sizes in (x.shape, weight.shape, stride, padding, dilation)
           for size in sizes):
        return False
    n, channels, height, width = x.shape
    kh, kw = weight.shape[-2:]
    oh = (height + 2*padding[0] - dilation[0]*(kh-1) - 1)//stride[0] + 1
    ow = (width + 2*padding[1] - dilation[1]*(kw-1) - 1)//stride[1] + 1
    inner, columns = channels*kh*kw, n*oh*ow
    # The portable contraction can fuse without an im2col allocation. Keep
    # it when this explicit workspace would exceed 256 MiB or BLAS limits.
    return (oh > 0 and ow > 0 and oh*ow <= 2147483647 and inner <= 2147483647
            and columns <= 2147483647 and inner*columns*4 <= (256 << 20))



def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
           *, _depthwise_fast_path=True):
    from jittor.misc import _pair
    stride, padding, dilation = map(_pair, (stride, padding, dilation))
    from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
    if not _output_requires_grad(x, weight, bias):
        from jittor._runtime.backend_libraries import get_library_ops
        return _stop_grad_outputs(get_library_ops('hipblas').hipblas_conv_inference(
            x, weight, [] if bias is None else [bias], stride, padding, dilation))
    n, channels, height, width = x.shape
    out_channels, _, kh, kw = weight.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    oh = (height + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (width + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    # Row-major GEMM: weights [Cout, Cin*Kh*Kw], columns [Cin*Kh*Kw, N*Oh*Ow].
    # Reindex owns both padding and its native differentiable scatter; backward
    # goes through these same operators rather than a separate convolution.
    columns = x.reindex((channels * kh * kw, n * oh * ow), [
        f'i1/{oh * ow}',
        f'i0/{kh * kw}',
        f'(i1/{ow})%{oh}*{sh}-{ph}+(i0/{kw})%{kh}*{dh}',
        f'i1%{ow}*{sw}-{pw}+i0%{kw}*{dw}',
    ])
    output = jt.matmul(weight.reshape((out_channels, channels * kh * kw)), columns)
    output = output.reshape((out_channels, n, oh, ow)).transpose((1, 0, 2, 3))
    output = output.contiguous()
    if bias is not None:
        output = output + bias.reshape((1, out_channels, 1, 1))
    return output
