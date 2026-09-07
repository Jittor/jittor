"""Stateless core 3d pooling implementation."""

import math
import jittor as jt
from jittor.backends.cuda.kernels.pooling.pool3d import pool3d_cuda_options
from .average import avg_pool3d, _pool_output_size
from . import _state


def _triple(x):
    if isinstance(x, tuple):
        assert len(x) == 3
        return x
    else:
        return (x,x,x)


def _pool3d_parameters(kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op='maximum'):
    assert dilation == None
    assert return_indices == None or op == "maximum"
    p_return_indices = return_indices
    p_kernel_size = _triple(kernel_size)
    p_op = op
    stride = stride if stride else kernel_size
    p_stride = _triple(stride)
    p_padding = _triple(padding)
    p_ceil_mode = ceil_mode
    # torch's count_include_pad selects the averaging *divisor*; it is not
    # conditional on the padding being non-zero. The old
    # ``count_include_pad and padding != 0`` read the raw argument, so
    # padding=(0,0,0) took a different branch than padding=0 (a tuple is
    # never == 0) and the same pooling produced different numbers.
    p_count_include_pad = count_include_pad
    if p_kernel_size[0] <= 0 or p_kernel_size[1] <= 0 or p_kernel_size[2] <= 0:
        raise RuntimeError(f"kernel_size must be greater than zero, but got {kernel_size}")
    if p_stride[0] <= 0 or p_stride[1] <= 0 or p_stride[2] <= 0:
        raise RuntimeError(f"stride must be greater than zero, but got {stride}")
    if p_padding[0] < 0 or p_padding[1] < 0 or p_padding[2] < 0:
        raise RuntimeError(f"padding must be non-negative, but got {padding}")
    return {
        "ceil_mode": p_ceil_mode,
        "count_include_pad": p_count_include_pad,
        "kernel_size": p_kernel_size,
        "op": p_op,
        "padding": p_padding,
        "return_indices": p_return_indices,
        "stride": p_stride,
    }


def _pool3d(x, *, ceil_mode, count_include_pad, kernel_size, op, padding, return_indices, stride):
    if op == "mean":
        # Same single implementation as Pool; see core_2d.Pool.execute.
        return avg_pool3d(
            x, kernel_size, stride, padding,
            ceil_mode, count_include_pad)
    N,C,D,H,W = x.shape
    if D <= kernel_size[0] or H <= kernel_size[1] or W <= kernel_size[2]:
        raise RuntimeError(f"size of var should be larger than kernel_size")
    d, h, w = (
        _pool_output_size(size, kernel, stride, padding, ceil_mode)
        for size, kernel, stride, padding in zip(
            (D, H, W), kernel_size, stride, padding
        )
    )
    use_code_op = op in ['maximum', 'minimum']

    if use_code_op and _state.pool_use_code_op:
        forward_body = f'''
                int k4 = i4*{stride[2]}-{padding[2]};
                int k3 = i3*{stride[1]}-{padding[1]};
                int k2 = i2*{stride[0]}-{padding[0]};
                int k4_ = min(k4 + {kernel_size[2]}, in0_shape4);
                int k3_ = min(k3 + {kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {kernel_size[0]}, in0_shape2);
                k4 = max(0, k4);
                k3 = max(0, k3);
                k2 = max(0, k2);
            '''
        if not return_indices:
            forward_body += f'''
                @out(i0, i1, i2, i3, i4) = @expand_op(init_{op}, @out_type);
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        for (int r = k4; r < k4_; ++r)
                            @out(i0, i1, i2, i3, i4) = @expand_op({op}, @out_type, @out(i0, i1, i2, i3, i4), @out_type, @in0(i0, i1, p, q, r), @in0_type);
                '''
        else:
            forward_body += f'''
                auto out_value = @expand_op(init_{op}, @out_type);
                int64 out_index = -1;
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        for (int r = k4; r < k4_; ++r)
                            if (out_value < @in0(i0, i1, p, q, r)) {{
                            out_value = @in0(i0, i1, p, q, r);
                            out_index = ((int64)p * in0_shape3 + q) * in0_shape4 + r;
                        }}
                @out(i0, i1, i2, i3, i4) = out_value;
                @out1(i0, i1, i2, i3, i4) = out_index;
                '''
        backward_body = f'''
                int k4 = i4*{stride[2]}-{padding[2]};
                int k3 = i3*{stride[1]}-{padding[1]};
                int k2 = i2*{stride[0]}-{padding[0]};
                int k4_ = min(k4 + {kernel_size[2]}, in0_shape4);
                int k3_ = min(k3 + {kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {kernel_size[0]}, in0_shape2);
                k4 = max(0, k4);
                k3 = max(0, k3);
                k2 = max(0, k2);
                int bo=1;
                for (int p = k2; p < k2_ && bo; ++p)
                    for (int q = k3; q < k3_ && bo; ++q)\x20
                        for (int r = k4; r < k4_ && bo; ++r) {{
                            if (@pout(i0,i1,i2,i3,i4) == @in0(i0,i1,p,q,r)) {{
                                atomicAdd(&@out(i0,i1,p,q,r), @dout(i0,i1,i2,i3,i4)),
                                bo=0;
                            }}
                        }}
            '''
        if return_indices:
            # int64 like torch's return_indices, and the encoding
            # below is computed in int64 too: it is a *flat* offset
            # into one (n, c) plane, so `(p * H + q) * W + r` leaves
            # int32 as soon as a plane has 2**31 elements -- which is
            # where the index would matter most.
            return_shapes = [[N,C,d,h,w]] * 2
            return_dtypes = [x.dtype, 'int64']
        else:
            return_shapes = [N,C,d,h,w]
            return_dtypes = x.dtype
        out = jt.code(return_shapes, return_dtypes, [x],
            **pool3d_cuda_options(forward_body, backward_body),
            cpu_header='',
            cpu_src=f'''
                    using namespace std;
                    for (int i0=0; i0<out_shape0; i0++)
                    for (int i1=0; i1<out_shape1; i1++)
                    for (int i2=0; i2<out_shape2; i2++)
                    for (int i3=0; i3<out_shape3; i3++)
                    for (int i4=0; i4<out_shape4; i4++)
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
                    for (int i4=0; i4<pout_shape4; i4++)
                        {{ {backward_body} }}
                '''])
        return out
    else:
        # TODO: backward
        xx = x.reindex([N,C,d,h,w,kernel_size[0],kernel_size[1],kernel_size[2]], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{stride[0]}-{padding[0]}+i5", # Did
            f"i3*{stride[1]}-{padding[1]}+i6", # Hid
            f"i4*{stride[2]}-{padding[2]}+i7", # Hid
        ])
        return xx.reduce(op, [5,6,7])
