"""ROCm convolution lowered to native hipBLAS GEMM."""

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor.nn.functional._amp import prefers_half


def supported(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
              *, _depthwise_fast_path=True):
    dtype = dtype_name(x.dtype)
    floating = ('float32', 'float16', 'bfloat16')
    amp = prefers_half()
    if (groups != 1 or x.ndim != 4 or weight.ndim != 4
            or dtype not in floating or dtype_name(weight.dtype) not in floating
            or (bias is not None and dtype_name(bias.dtype) not in floating)
            or not weight._storage_is_contiguous()):
        return False
    if amp:
        dtype = dtype_name(jt.binary_dtype_infer("multiply", x.dtype, weight.dtype))
    elif dtype_name(weight.dtype) != dtype or (bias is not None and dtype_name(bias.dtype) != dtype):
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
    if not (oh > 0 and ow > 0 and oh*ow <= 2147483647 and inner <= 2147483647
            and columns <= 2147483647):
        return False
    workspace_limit = 256 << 20
    if inner*columns*4 <= workspace_limit:
        return True
    # Inference can reuse a bounded im2col tile. The differentiable path
    # still requires the full columns tensor for its existing backward.
    from jittor._runtime.core_api import _output_requires_grad
    return dtype == "float32" and inner*4 <= workspace_limit and not _output_requires_grad(x, weight, bias)



def conv2d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
           *, _depthwise_fast_path=True):
    from jittor.misc import _pair
    stride, padding, dilation = map(_pair, (stride, padding, dilation))
    from jittor._runtime.core_api import _output_requires_grad, _stop_grad_outputs
    if prefers_half():
        dtype = jt.binary_dtype_infer("multiply", x.dtype, weight.dtype)
        x, weight = x.cast(dtype), weight.cast(dtype)
        if bias is not None:
            bias = bias.cast(dtype)
    if dtype_name(x.dtype) == "float32" and not _output_requires_grad(x, weight, bias):
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
    if n > 0 and _output_requires_grad(x, weight, bias):
        from jittor._runtime.backend_libraries import get_library_ops
        native = getattr(get_library_ops("hipblas"), "miopen_conv2d", None)
        if native is not None:
            # MIOpen plans forward and both adjoints at construction. Only a
            # genuine capability miss returns to the existing HIP GEMM graph.
            with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
                try:
                    output = native(x, weight, (n,out_channels,oh,ow),
                                    stride, padding, dilation, 0)
                except jt.JittorError as error:
                    if "MIOpen training convolution has no bounded solution" not in str(error):
                        raise
                else:
                    if bias is not None:
                        output = output + bias.reshape((1,out_channels,1,1))
                    return output
    # Row-major GEMM: weights [Cout, Cin*Kh*Kw], columns [Cin*Kh*Kw, N*Oh*Ow].
    # Reindex owns both padding and its native differentiable scatter; backward
    # goes through these same operators rather than a separate convolution.
    columns = x.reindex((channels * kh * kw, n * oh * ow), [
        f'i1/{oh * ow}',
        f'i0/{kh * kw}',
        f'(i1/{ow})%{oh}*{sh}-{ph}+(i0/{kw})%{kh}*{dh}',
        f'i1%{ow}*{sw}-{pw}+i0%{kw}*{dw}',
    ])
    dense_weight = weight.reshape((out_channels, channels * kh * kw))
    if (bias is not None and dtype_name(x.dtype) in ("float16", "bfloat16")
            and _output_requires_grad(x, weight, bias)):
        from jittor._runtime.backend_libraries import get_library_ops
        with jt.flag_scope(amp_reg=jt.amp_flags.disabled):
            output = get_library_ops("hipblas").hipblas_matmul(
                dense_weight, columns, False, False, True)
            output = output.reshape((out_channels, n, oh, ow)).transpose((1, 0, 2, 3)).contiguous()
            return (output + bias.float32().reshape((1, out_channels, 1, 1))).cast(x.dtype)
    output = jt.matmul(dense_weight, columns)
    output = output.reshape((out_channels, n, oh, ow)).transpose((1, 0, 2, 3))
    output = output.contiguous()
    if bias is not None:
        output = output + bias.reshape((1, out_channels, 1, 1))
    return output
