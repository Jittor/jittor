"""Legacy 2D pooling core and generated CPU/CUDA kernels."""

import jittor as jt


class Pool(jt.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None or op == "maximum"
        self.return_indices = return_indices
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.op = op
        stride = stride if stride else kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.ceil_mode = ceil_mode
        # torch's count_include_pad selects the averaging *divisor*; it is not
        # conditional on the padding being non-zero. The old
        # ``count_include_pad and padding != 0`` read the raw argument, so
        # padding=(0,0) took a different branch than padding=0 (a tuple is
        # never == 0) and the same pooling produced different numbers. Same fix
        # as Pool3d. op="mean" no longer reads this at all -- it delegates to
        # jt.nn.avg_pool2d -- but MaxPool2d/MaxPool3d still snapshot it.
        self.count_include_pad = count_include_pad
        for item in self.kernel_size:
            if item <= 0:
                raise RuntimeError(f"kernel_size must be greater than zero, but got {item}")
        for item in self.stride:
            if item <= 0:
                raise RuntimeError(f"stride must be greater than zero, but got {item}")
        for item in self.padding:
            if item < 0:
                raise RuntimeError(f"padding must be non-negative, but got {item}")

    def execute(self, x):
        if self.op == "mean":
            # One implementation of average pooling for the whole package; see
            # jittor/nn/functional/pooling.py. The kernels below stay for
            # maximum/minimum, whose semantics this class still owns.
            return jt.nn.avg_pool2d(
                x, self.kernel_size, self.stride, self.padding,
                self.ceil_mode, self.count_include_pad)
        N,C,H,W = x.shape
        # torch only requires the *padded* input to be at least the kernel size
        # (so the output has size >= 1). The original guard ignored padding, which
        # wrongly rejected e.g. SPPF's MaxPool2d(kernel=13, padding=6) on an 8x8
        # feature map (padded 20x20, valid in torch). Make the guard padding-aware;
        # for padding=0 this only additionally allows the H==kernel boundary, which
        # torch accepts (output 1), so no previously-passing case regresses.
        if (H + 2*self.padding[0] < self.kernel_size[0]
                or W + 2*self.padding[1] < self.kernel_size[1]):
            raise RuntimeError(f"size of var should be larger than kernel_size")
        if self.ceil_mode == False:
            h = (H+self.padding[0]*2-self.kernel_size[0])//self.stride[0]+1
            w = (W+self.padding[1]*2-self.kernel_size[1])//self.stride[1]+1
            use_code_op = self.op in ['maximum', 'minimum']
        else:
            h = (H+self.padding[0]*2-self.kernel_size[0] + self.stride[0] - 1)//self.stride[0]+1
            w = (W+self.padding[1]*2-self.kernel_size[1] + self.stride[1] - 1)//self.stride[1]+1
            use_code_op = self.op in ['maximum', 'minimum']

        if use_code_op and jt.pool.pool_use_code_op:
            forward_body = f'''
                int k3 = i3*{self.stride[1]}-{self.padding[1]};
                int k2 = i2*{self.stride[0]}-{self.padding[0]};
                int k3_ = min(k3 + {self.kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size[0]}, in0_shape2);
                k3 = max(0, k3);
                k2 = max(0, k2);
            '''
            if not self.return_indices:
                forward_body += f'''
                @out(i0, i1, i2, i3) = @expand_op(init_{self.op}, @out_type);
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        @out(i0, i1, i2, i3) = @expand_op({self.op}, @out_type, @out(i0, i1, i2, i3), @out_type, @in0(i0, i1, p, q), @in0_type);
                '''
            else:
                forward_body += f'''
                auto out_value = @expand_op(init_{self.op}, @out_type);
                int out_index = -1;
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)\x20
                        if (out_value < @in0(i0, i1, p, q)) {{
                            out_value = @in0(i0, i1, p, q);
                            out_index = p * in0_shape3 + q;
                        }}
                @out(i0, i1, i2, i3) = out_value;
                @out1(i0, i1, i2, i3) = out_index;
                '''
            backward_body = f'''
                int k3 = i3*{self.stride[1]}-{self.padding[1]};
                int k2 = i2*{self.stride[0]}-{self.padding[0]};
                int k3_ = min(k3 + {self.kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size[0]}, in0_shape2);
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
            if self.return_indices:
                return_shapes = [[N,C,h,w]] * 2
                return_dtypes = [x.dtype, 'int32']
            else:
                return_shapes = [N,C,h,w]
                return_dtypes = x.dtype
            out = jt.code(return_shapes, return_dtypes, [x],
                cuda_header="""
                    #include <misc/cuda_limits.h>
                """,
                cuda_src=f'''
                    __global__ static void kernel1(@ARGS_DEF) {{
                        @PRECALC
                        int p3 = threadIdx.x;
                        int s3 = blockDim.x;
                        int p2 = threadIdx.y + blockIdx.x * blockDim.y;
                        int s2 = blockDim.y * gridDim.x;
                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i3 = p3; i3 < out_shape3; i3 += s3)
                        for (int i2 = p2; i2 < out_shape2; i2 += s2)
                            {{ {forward_body} }}
                    }}
                    int tx = std::min(1024, out_shape3);
                    int ty = std::min(1024 / tx, out_shape2);
                    int bx = (out_shape2 - 1) / ty + 1;
                    int by = out_shape1;
                    int bz = out_shape0;
                    dim3 s1(bx, by, bz);
                    dim3 s2(tx, ty);
                    kernel1<<<s1, s2>>>(@ARGS);
                ''',
                cuda_grad_src=[f'''
                    __global__ static void kernel3(@ARGS_DEF) {{
                        @PRECALC
                        int p3 = threadIdx.x;
                        int s3 = blockDim.x;
                        int p2 = threadIdx.y + blockIdx.x * blockDim.y;
                        int s2 = blockDim.y * gridDim.x;
                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i3 = p3; i3 < pout_shape3; i3 += s3)
                            for (int i2 = p2; i2 < pout_shape2; i2 += s2)
                                {{ {backward_body} }}
                    }}
                    cudaMemsetAsync(out_p, 0, out->size);
                    int tx = std::min(1024, pout_shape3);
                    int ty = std::min(1024 / tx, pout_shape2);
                    int bx = (pout_shape2 - 1) / ty + 1;
                    int by = pout_shape1;
                    int bz = pout_shape0;
                    dim3 s1_(bx, by, bz);
                    dim3 s2_(tx, ty);
                    kernel3<<<s1_, s2_>>>(@ARGS);
                '''],
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
            xx = x.reindex([N,C,h,w,self.kernel_size[0],self.kernel_size[1]], [
                "i0", # Nid
                "i1", # Cid
                f"i2*{self.stride[0]}-{self.padding[0]}+i4", # Hid
                f"i3*{self.stride[1]}-{self.padding[1]}+i5", # Wid
            ])
            return xx.reduce(self.op, [4,5])


_PUBLIC_SYMBOLS = (Pool,)
