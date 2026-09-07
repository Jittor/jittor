"""Stateless core 2d pooling implementation."""

import math
import jittor as jt
from jittor._runtime.dispatch import try_dispatch
from jittor.backends.cuda.kernels.pooling.pool2d import pool2d_cuda_options
from .average import avg_pool2d
from . import _state


def _pool2d_parameters(kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op='maximum'):
    assert dilation == None
    assert return_indices == None or op == "maximum"
    p_return_indices = return_indices
    p_kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    p_op = op
    stride = stride if stride else kernel_size
    p_stride = stride if isinstance(stride, tuple) else (stride, stride)
    p_padding = padding if isinstance(padding, tuple) else (padding, padding)
    p_ceil_mode = ceil_mode
    # torch's count_include_pad selects the averaging *divisor*; it is not
    # conditional on the padding being non-zero. The old
    # ``count_include_pad and padding != 0`` read the raw argument, so
    # padding=(0,0) took a different branch than padding=0 (a tuple is
    # never == 0) and the same pooling produced different numbers. Same fix
    # as Pool3d. op="mean" no longer reads this at all -- it delegates to
    # jt.nn.avg_pool2d -- but MaxPool2d/MaxPool3d still snapshot it.
    p_count_include_pad = count_include_pad
    for item in p_kernel_size:
        if item <= 0:
            raise RuntimeError(f"kernel_size must be greater than zero, but got {item}")
    for item in p_stride:
        if item <= 0:
            raise RuntimeError(f"stride must be greater than zero, but got {item}")
    for item in p_padding:
        if item < 0:
            raise RuntimeError(f"padding must be non-negative, but got {item}")
    return {
        "ceil_mode": p_ceil_mode,
        "count_include_pad": p_count_include_pad,
        "kernel_size": p_kernel_size,
        "op": p_op,
        "padding": p_padding,
        "return_indices": p_return_indices,
        "stride": p_stride,
    }


def _pool2d(x, *, ceil_mode, count_include_pad, kernel_size, op, padding, return_indices, stride):
    if op == "mean":
        # One implementation of average pooling for the whole package; see
        # jittor/nn/functional/pooling.py. The kernels below stay for
        # maximum/minimum, whose semantics this class still owns.
        return avg_pool2d(
            x, kernel_size, stride, padding,
            ceil_mode, count_include_pad)
    N,C,H,W = x.shape
    # torch only requires the *padded* input to be at least the kernel size
    # (so the output has size >= 1). The original guard ignored padding, which
    # wrongly rejected e.g. SPPF's MaxPool2d(kernel=13, padding=6) on an 8x8
    # feature map (padded 20x20, valid in torch). Make the guard padding-aware;
    # for padding=0 this only additionally allows the H==kernel boundary, which
    # torch accepts (output 1), so no previously-passing case regresses.
    if (H + 2*padding[0] < kernel_size[0]
            or W + 2*padding[1] < kernel_size[1]):
        raise RuntimeError(f"size of var should be larger than kernel_size")
    if ceil_mode == False:
        h = (H+padding[0]*2-kernel_size[0])//stride[0]+1
        w = (W+padding[1]*2-kernel_size[1])//stride[1]+1
        use_code_op = op in ['maximum', 'minimum']
    else:
        h = (H+padding[0]*2-kernel_size[0] + stride[0] - 1)//stride[0]+1
        w = (W+padding[1]*2-kernel_size[1] + stride[1] - 1)//stride[1]+1
        use_code_op = op in ['maximum', 'minimum']

    fast = try_dispatch(
        "nn.pool2d", x, kernel_size, stride, padding,
        None, return_indices, ceil_mode, count_include_pad, op)
    if fast is not None:
        return fast
    if use_code_op and _state.pool_use_code_op:
        forward_body = f'''
                int k3 = i3*{stride[1]}-{padding[1]};
                int k2 = i2*{stride[0]}-{padding[0]};
                int k3_ = min(k3 + {kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {kernel_size[0]}, in0_shape2);
                k3 = max(0, k3);
                k2 = max(0, k2);
            '''
        if not return_indices:
            forward_body += f'''
                @out(i0, i1, i2, i3) = @expand_op(init_{op}, @out_type);
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        @out(i0, i1, i2, i3) = @expand_op({op}, @out_type, @out(i0, i1, i2, i3), @out_type, @in0(i0, i1, p, q), @in0_type);
                '''
        else:
            forward_body += f'''
                auto out_value = @expand_op(init_{op}, @out_type);
                int64 out_index = -1;
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)\x20
                        if (out_value < @in0(i0, i1, p, q)) {{
                            out_value = @in0(i0, i1, p, q);
                            out_index = (int64)p * in0_shape3 + q;
                        }}
                @out(i0, i1, i2, i3) = out_value;
                @out1(i0, i1, i2, i3) = out_index;
                '''
        backward_body = f'''
                int k3 = i3*{stride[1]}-{padding[1]};
                int k2 = i2*{stride[0]}-{padding[0]};
                int k3_ = min(k3 + {kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {kernel_size[0]}, in0_shape2);
                k3 = max(0, k3);
                k2 = max(0, k2);
                int bo=1;
                for (int p = k2; p < k2_ && bo; ++p)
                    for (int q = k3; q < k3_ && bo; ++q) {{
                        if (@pout(i0,i1,i2,i3) == @in0(i0,i1,p,q)) {{
                            atomicAdd(&@out(i0,i1,p,q), @dout(i0,i1,i2,i3)),
                            bo=0;
                        }}
                    }}
            '''
        if return_indices:
            # int64 like torch's return_indices, and the encoding
            # below is computed in int64 too: it is a *flat* offset
            # into one (n, c) plane, so `p * W + q` leaves int32 as
            # soon as a plane has 2**31 elements -- which is where
            # the index would matter most.
            return_shapes = [[N,C,h,w]] * 2
            return_dtypes = [x.dtype, 'int64']
        else:
            return_shapes = [N,C,h,w]
            return_dtypes = x.dtype
        out = jt.code(return_shapes, return_dtypes, [x],
            **pool2d_cuda_options(forward_body, backward_body),
            cpu_header='',
            cpu_src=f'''
                    using namespace std;
                    for (int i0=0; i0<out_shape0; i0++)
                    for (int i1=0; i1<out_shape1; i1++)
                    for (int i2=0; i2<out_shape2; i2++)
                    for (int i3=0; i3<out_shape3; i3++)
                        {{ {forward_body} }}
                ''',
            cpu_grad_src = [f'''
                    using namespace std;
                    std::memset(out_p, 0, out->size);
                    #define atomicAdd(a,b) (*a) += b

                    for (int i0=0; i0<pout_shape0; i0++)
                    for (int i1=0; i1<pout_shape1; i1++)
                    for (int i2=0; i2<pout_shape2; i2++)\x20
                    for (int i3=0; i3<pout_shape3; i3++)
                        {{ {backward_body} }}
                '''])
        return out
    else:
        # TODO: backward
        xx = x.reindex([N,C,h,w,kernel_size[0],kernel_size[1]], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{stride[0]}-{padding[0]}+i4", # Hid
            f"i3*{stride[1]}-{padding[1]}+i5", # Wid
        ])
        return xx.reduce(op, [4,5])
