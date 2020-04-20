# ***************************************************************
# Copyright (c) 2020 Jittor. Authors:
#     Guowei Yang <471184555@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#
# All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from jittor import init, Module
import numpy as np
import math

class Pool(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, op="maximum"):
        assert dilation == None
        assert return_indices == None
        self.kernel_size = kernel_size
        self.op = op
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode

    def execute(self, x):
        N,C,H,W = x.shape
        if (self.ceil_mode == False):
            h = (H+self.padding*2-self.kernel_size)//self.stride+1
            w = (W+self.padding*2-self.kernel_size)//self.stride+1
        else:
            h = (H+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1
            w = (W+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1

        if (self.op == 'maximum' or self.op == 'minimum'):
            if (self.op == 'maximum'):
                op = 'max'
            else:
                op = 'min'
            out = jt.code([N,C,h,w], x.dtype, [x],
                cuda_src=f'''
                    __global__ static void kernel1(@ARGS_DEF) {{
                        @PRECALC
                        int p3 = threadIdx.x;
                        int s3 = blockDim.x;
                        int p2 = threadIdx.y + blockIdx.x * blockDim.y;
                        int s2 = blockDim.y * gridDim.x;
                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i3 = p3; i3 < outshape3; i3 += s3)
                            for (int i2 = p2; i2 < outshape2; i2 += s2) {{
                                int k3 = i3*{self.stride}-{self.padding};
                                int k2 = i2*{self.stride}-{self.padding};
                                int k3_ = min(k3 + {self.kernel_size}, in0shape3);
                                int k2_ = min(k2 + {self.kernel_size}, in0shape2);
                                k3 = max(0, k3);
                                k2 = max(0, k2);
                                @out(i0, i1, i2, i3) = @in0(i0, i1, k2, k3);
                                for (int p = k2; p < k2_; ++p)
                                    for (int q = k3; q < k3_; ++q)
                                        @out(i0, i1, i2, i3) = {op}(@out(i0, i1, i2, i3), @in0(i0, i1, p, q));
                            }}
                    }}
                    int tx = min(1024, outshape3);
                    int ty = min(1024 / tx, outshape2);
                    int bx = (outshape2 - 1) / ty + 1;
                    int by = outshape1;
                    int bz = outshape0;
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
                        for (int i3 = p3; i3 < poutshape3; i3 += s3)
                            for (int i2 = p2; i2 < poutshape2; i2 += s2) {{
                                int k3 = i3*{self.stride}-{self.padding};
                                int k2 = i2*{self.stride}-{self.padding};
                                int k3_ = min(k3 + {self.kernel_size}, in0shape3);
                                int k2_ = min(k2 + {self.kernel_size}, in0shape2);
                                k3 = max(0, k3);
                                k2 = max(0, k2);
                                int bo=1;
                                for (int p = k2; p < k2_ && bo; ++p)
                                    for (int q = k3; q < k3_ && bo; ++q) {{
                                        if (@pout(i0,i1,i2,i3) == @in0(i0,i1,p,q)) {{
                                            atomicAdd(&@out(i0,i1,p,q), @dout(i0,i1,i2,i3));
                                            bo=0;
                                        }}
                                    }}
                            }}
                    }}
                    cudaMemsetAsync(outp, 0, out->size);
                    int tx = min(1024, poutshape3);
                    int ty = min(1024 / tx, poutshape2);
                    int bx = (poutshape2 - 1) / ty + 1;
                    int by = poutshape1;
                    int bz = poutshape0;
                    dim3 s1_(bx, by, bz);
                    dim3 s2_(tx, ty);
                    kernel3<<<s1_, s2_>>>(@ARGS);
                '''],
                cpu_src=f'''
                    for (int i0=0; i0<outshape0; i0++)
                    for (int i1=0; i1<outshape1; i1++)
                    for (int i2=0; i2<outshape2; i2++)
                    for (int i3=0; i3<outshape3; i3++) {{
                        int k2 = i2*{self.stride}-{self.padding};
                        int k3 = i3*{self.stride}-{self.padding};
                        int k2_ = std::min(k2 + {self.kernel_size}, in0shape2);
                        int k3_ = std::min(k3 + {self.kernel_size}, in0shape3);
                        k2 = std::max(0, k2);
                        k3 = std::max(0, k3);
                        @out(i0, i1, i2, i3) = @in0(i0, i1, k2, k3);
                        for (int p = k2; p < k2_; ++p)
                            for (int q = k3; q < k3_; ++q)
                                @out(i0, i1, i2, i3) = std::{op}(@out(i0, i1, i2, i3), @in0(i0, i1, p, q));
                    }}
                ''',
                cpu_grad_src = [f'''
                    for (int i=0; i<outshape0; i++)
                    for (int j=0; j<outshape1; j++)
                    for (int k=0; k<outshape2; k++)
                    for (int l=0; l<outshape3; l++) @out(i,j,k,l) = 0;

                    for (int i0=0; i0<poutshape0; i0++)
                    for (int i1=0; i1<poutshape1; i1++)
                    for (int i2=0; i2<poutshape2; i2++) 
                    for (int i3=0; i3<poutshape3; i3++) {{
                        int k3 = i3*{self.stride}-{self.padding};
                        int k2 = i2*{self.stride}-{self.padding};
                        int k3_ = std::min(k3 + {self.kernel_size}, in0shape3);
                        int k2_ = std::min(k2 + {self.kernel_size}, in0shape2);
                        k3 = std::max(0, k3);
                        k2 = std::max(0, k2);
                        int bo=1;
                        for (int p = k2; p < k2_ && bo; ++p)
                            for (int q = k3; q < k3_ && bo; ++q) {{
                                if (@pout(i0,i1,i2,i3) == @in0(i0,i1,p,q)) {{
                                    @out(i0,i1,p,q) += @dout(i0,i1,i2,i3);
                                    bo=0;
                                }}
                            }}
                    }}
                '''])
            return out
        else:
            xx = x.reindex([N,C,h,w,self.kernel_size,self.kernel_size], [
                "i0", # Nid
                "i1", # Cid
                f"i2*{self.stride}-{self.padding}+i4", # Hid
                f"i3*{self.stride}-{self.padding}+i5", # Wid
            ])
            return xx.reduce(self.op, [4,5])


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size):
        self.output_size = output_size

    def execute(self, x):
        if isinstance(self.output_size, int):
            oh = self.output_size
            ow = self.output_size
        elif isinstance(self.output_size, tuple) or isinstance(self.output_size, list):
            oh = x.shape[2] if self.output_size[0] is None else self.output_size[0]
            ow = x.shape[3] if self.output_size[1] is None else self.output_size[1]
        else:
            raise TypeError(f"AdaptiveAvgPool2d only support int, typle or list input. Not support {type(self.output_size)} yet.")
        N,C,H,W = x.shape
        self.sh = math.floor(H / oh)
        self.sw = math.floor(W / ow)
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,h,w,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sh}+i4", # Hid
            f"i3*{self.sw}+i5", # Wid
        ])
        return xx.reduce("mean", [4,5])

def pool(x, kernel_size, op, padding=0, stride = 1):
    return Pool(kernel_size, stride, padding, op=op)(x)