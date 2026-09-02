"""Legacy 3D pooling core and generated CPU/CUDA kernels."""

import jittor as jt


def _triple(x):
    if isinstance(x, tuple):
        assert len(x) == 3
        return x
    else:
        return (x,x,x)


def _valid_element_counts(size, kernel, stride, padding, out_size):
    """Per-output-position count of real (non-padded) input elements, one axis.

    This is torch's ``count_include_pad=False`` divisor: the window
    ``[i*stride - padding, i*stride - padding + kernel)`` intersected with
    ``[0, size)``.  It depends only on the geometry, so it is a constant.
    """
    counts = []
    for index in range(out_size):
        start = index * stride - padding
        end = min(start + kernel, size)
        counts.append(max(end - max(start, 0), 1))
    return counts


class Pool3d(jt.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None or op == "maximum"
        self.return_indices = return_indices
        self.kernel_size = jt.pool._triple(kernel_size)
        self.op = op
        stride = stride if stride else kernel_size
        self.stride = jt.pool._triple(stride)
        self.padding = jt.pool._triple(padding)
        self.ceil_mode = ceil_mode
        # torch's count_include_pad selects the averaging *divisor*; it is not
        # conditional on the padding being non-zero. The old
        # ``count_include_pad and padding != 0`` read the raw argument, so
        # padding=(0,0,0) took a different branch than padding=0 (a tuple is
        # never == 0) and the same pooling produced different numbers.
        self.count_include_pad = count_include_pad
        if self.kernel_size[0] <= 0 or self.kernel_size[1] <= 0 or self.kernel_size[2] <= 0:
            raise RuntimeError(f"kernel_size must be greater than zero, but got {kernel_size}")
        if self.stride[0] <= 0 or self.stride[1] <= 0 or self.stride[2] <= 0:
            raise RuntimeError(f"stride must be greater than zero, but got {stride}")
        if self.padding[0] < 0 or self.padding[1] < 0 or self.padding[2] < 0:
            raise RuntimeError(f"padding must be non-negative, but got {padding}")

    def execute(self, x):
        N,C,D,H,W = x.shape
        if D <= self.kernel_size[0] or H <= self.kernel_size[1] or W <= self.kernel_size[2]:
            raise RuntimeError(f"size of var should be larger than kernel_size")
        if self.ceil_mode == False:
            d = (D+self.padding[0]*2-self.kernel_size[0])//self.stride[0]+1
            h = (H+self.padding[1]*2-self.kernel_size[1])//self.stride[1]+1
            w = (W+self.padding[2]*2-self.kernel_size[2])//self.stride[2]+1
            use_code_op = self.op in ['maximum', 'minimum']
            # some second order avg_pool is require, so we don't use code op here
        else:
            d = (D+self.padding[0]*2-self.kernel_size[0] + self.stride[0] - 1)//self.stride[0]+1
            h = (H+self.padding[1]*2-self.kernel_size[1] + self.stride[1] - 1)//self.stride[1]+1
            w = (W+self.padding[2]*2-self.kernel_size[2] + self.stride[2] - 1)//self.stride[2]+1
            use_code_op = self.op in ['maximum', 'minimum', 'mean']

        if use_code_op and jt.pool.pool_use_code_op:
            if self.op == 'mean':
                if self.count_include_pad:
                    # torch counts the padded positions but stops at the edge of
                    # the padded volume, so a ceil_mode window hanging past
                    # ``size + padding`` divides by less than the kernel volume.
                    # Written from i2/i3/i4 because k2/k3/k4 are clamped to >= 0
                    # by the time this runs.
                    starts = [f"(i{2+a}*{self.stride[a]}-{self.padding[a]})" for a in range(3)]
                    terms = [
                        f"(min({starts[a]}+{self.kernel_size[a]}, in0_shape{2+a}+{self.padding[a]}) - {starts[a]})"
                        for a in range(3)
                    ]
                    count = "int count = " + " * ".join(terms) + ";"
                else:
                    count = "int count = (k2_ - k2) * (k3_ - k3) * (k4_ - k4);"
                count += "float32 rcount = 1.0f / count;"
            else:
                count = ""
            forward_body = f'''
                int k4 = i4*{self.stride[2]}-{self.padding[2]};
                int k3 = i3*{self.stride[1]}-{self.padding[1]};
                int k2 = i2*{self.stride[0]}-{self.padding[0]};
                int k4_ = min(k4 + {self.kernel_size[2]}, in0_shape4);
                int k3_ = min(k3 + {self.kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size[0]}, in0_shape2);
                k4 = max(0, k4);
                k3 = max(0, k3);
                k2 = max(0, k2);
                {count}
            '''
            if not self.return_indices:
                forward_body += f'''
                @out(i0, i1, i2, i3, i4) = @expand_op(init_{self.op}, @out_type);
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        for (int r = k4; r < k4_; ++r)
                            @out(i0, i1, i2, i3, i4) = @expand_op({self.op}, @out_type, @out(i0, i1, i2, i3, i4), @out_type, @in0(i0, i1, p, q, r), @in0_type);
                '''
            else:
                forward_body += f'''
                auto out_value = @expand_op(init_{self.op}, @out_type);
                int out_index = -1;
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        for (int r = k4; r < k4_; ++r)
                            if (out_value < @in0(i0, i1, p, q, r)) {{
                            out_value = @in0(i0, i1, p, q, r);
                            out_index = p * in0_shape3 * in0_shape4 + q * in0_shape4 + r;
                        }}
                @out(i0, i1, i2, i3, i4) = out_value;
                @out1(i0, i1, i2, i3, i4) = out_index;
                '''
            backward_body = f'''
                int k4 = i4*{self.stride[2]}-{self.padding[2]};
                int k3 = i3*{self.stride[1]}-{self.padding[1]};
                int k2 = i2*{self.stride[0]}-{self.padding[0]};
                int k4_ = min(k4 + {self.kernel_size[2]}, in0_shape4);
                int k3_ = min(k3 + {self.kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size[0]}, in0_shape2);
                k4 = max(0, k4);
                k3 = max(0, k3);
                k2 = max(0, k2);
                {count}
                int bo=1;
                for (int p = k2; p < k2_ && bo; ++p)
                    for (int q = k3; q < k3_ && bo; ++q)\x20
                        for (int r = k4; r < k4_ && bo; ++r) {{
                            {"atomicAdd(&@out(i0,i1,p,q,r), @dout(i0,i1,i2,i3,i4)/count);"
                                if self.op == "mean" else
                            f"""if (@pout(i0,i1,i2,i3,i4) == @in0(i0,i1,p,q,r)) {{
                                atomicAdd(&@out(i0,i1,p,q,r), @dout(i0,i1,i2,i3,i4)),
                                bo=0;
                            }}"""}
                        }}
            '''
            if self.return_indices:
                return_shapes = [[N,C,d,h,w]] * 2
                return_dtypes = [x.dtype, 'int32']
            else:
                return_shapes = [N,C,d,h,w]
                return_dtypes = x.dtype
            out = jt.code(return_shapes, return_dtypes, [x],
                cuda_header="""
                    #include <misc/cuda_limits.h>
                """,
                cuda_src=f'''
                    __global__ static void kernel1(@ARGS_DEF) {{
                        @PRECALC
                        int p4 = threadIdx.x;
                        int s4 = blockDim.x;
                        int p3 = threadIdx.y;
                        int s3 = blockDim.y;
                        int p2 = threadIdx.z + blockIdx.x * blockDim.z;
                        int s2 = blockDim.z * gridDim.x;
                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i4 = p4; i4 < out_shape4; i4 += s4)
                        for (int i3 = p3; i3 < out_shape3; i3 += s3)
                        for (int i2 = p2; i2 < out_shape2; i2 += s2)
                            {{ {forward_body} }}
                    }}
                    int tx = std::min(1024, out_shape4);
                    int ty = std::min(1024 / tx, out_shape3);
                    int tz = std::min(1024 / tx / ty, out_shape2);
                    int bx = (out_shape2 - 1) / tz + 1;
                    int by = out_shape1;
                    int bz = out_shape0;
                    dim3 s1(bx, by, bz);
                    dim3 s2(tx, ty, tz);
                    kernel1<<<s1, s2>>>(@ARGS);
                ''',
                cuda_grad_src=[f'''
                    __global__ static void kernel3(@ARGS_DEF) {{
                        @PRECALC
                        int p4 = threadIdx.x;
                        int s4 = blockDim.x;
                        int p3 = threadIdx.y;
                        int s3 = blockDim.y;
                        int p2 = threadIdx.z + blockIdx.x * blockDim.z;
                        int s2 = blockDim.z * gridDim.x;
                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i4 = p4; i4 < pout_shape4; i4 += s4)
                        for (int i3 = p3; i3 < pout_shape3; i3 += s3)
                        for (int i2 = p2; i2 < pout_shape2; i2 += s2)
                                {{ {backward_body} }}
                    }}
                    cudaMemsetAsync(out_p, 0, out->size);
                    int tx = std::min(1024, pout_shape4);
                    int ty = std::min(1024 / tx, pout_shape3);
                    int tz = std::min(1024 / tx / ty, pout_shape2);
                    int bx = (pout_shape2 - 1) / tz + 1;
                    int by = pout_shape1;
                    int bz = pout_shape0;
                    dim3 s1(bx, by, bz);
                    dim3 s2(tx, ty, tz);
                    kernel3<<<s1, s2>>>(@ARGS);
                '''],
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
            xx = x.reindex([N,C,d,h,w,self.kernel_size[0],self.kernel_size[1],self.kernel_size[2]], [
                "i0", # Nid
                "i1", # Cid
                f"i2*{self.stride[0]}-{self.padding[0]}+i5", # Did
                f"i3*{self.stride[1]}-{self.padding[1]}+i6", # Hid
                f"i4*{self.stride[2]}-{self.padding[2]}+i7", # Hid
            ])
            if self.op == 'mean' and not self.count_include_pad:
                # reduce('mean') always divides by the full kernel volume, which
                # is torch's count_include_pad=True. For False the divisor is the
                # number of *real* input elements the window covers; the padded
                # reads are 0 (reindex's overflow_value), so summing and dividing
                # by that count is exact. The divisor separates per axis, so it
                # is three small constant vectors rather than a 3-D table.
                divisor = None
                for axis, (size, out) in enumerate(((D, d), (H, h), (W, w))):
                    counts = _valid_element_counts(
                        size, self.kernel_size[axis], self.stride[axis],
                        self.padding[axis], out)
                    shape = [1, 1, 1, 1, 1]
                    shape[2 + axis] = out
                    part = jt.array(counts, dtype=x.dtype).reshape(shape)
                    divisor = part if divisor is None else divisor * part
                return xx.reduce('add', [5,6,7]) / divisor
            return xx.reduce(self.op, [5,6,7])


_PUBLIC_SYMBOLS = (_triple, Pool3d)
