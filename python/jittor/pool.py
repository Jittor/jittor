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
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None
        self.kernel_size = kernel_size
        self.op = op
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad and padding != 0

    def execute(self, x):
        N,C,H,W = x.shape
        if self.ceil_mode == False:
            h = (H+self.padding*2-self.kernel_size)//self.stride+1
            w = (W+self.padding*2-self.kernel_size)//self.stride+1
        else:
            h = (H+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1
            w = (W+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1

        if self.op in ['maximum', 'minimum', 'mean']:
            if self.op == 'mean':
                if self.count_include_pad:
                    count = f"int count = {self.kernel_size*self.kernel_size};"
                else:
                    count = "int count = (k2_ - k2) * (k3_ - k3);"
                count += "float32 rcount = 1.0f / count;"
            else:
                count = ""
            forward_body = f'''{{
                int k3 = i3*{self.stride}-{self.padding};
                int k2 = i2*{self.stride}-{self.padding};
                int k3_ = min(k3 + {self.kernel_size}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size}, in0_shape2);
                k3 = max(0, k3);
                k2 = max(0, k2);
                @out(i0, i1, i2, i3) = init_{self.op}(out_type);
                {count}
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        @out(i0, i1, i2, i3) = {self.op}(out_type, @out(i0, i1, i2, i3), @in0(i0, i1, p, q));
            }}'''
            backward_body = f'''{{
                int k3 = i3*{self.stride}-{self.padding};
                int k2 = i2*{self.stride}-{self.padding};
                int k3_ = min(k3 + {self.kernel_size}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size}, in0_shape2);
                k3 = max(0, k3);
                k2 = max(0, k2);
                {count}
                int bo=1;
                for (int p = k2; p < k2_ && bo; ++p)
                    for (int q = k3; q < k3_ && bo; ++q) {{
                        {"atomicAdd(&@out(i0,i1,p,q), @dout(i0,i1,i2,i3)/count);"
                            if self.op == "mean" else
                        f"""if (@pout(i0,i1,i2,i3) == @in0(i0,i1,p,q)) {{
                            atomicAdd(&@out(i0,i1,p,q), @dout(i0,i1,i2,i3)),
                            bo=0;
                        }}"""}
                    }}
            }}'''
            out = jt.code([N,C,h,w], x.dtype, [x],
                cuda_header="""
                    #include <ops/binary_op_defs.h>
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
                            {forward_body}
                    }}
                    int tx = min(1024, out_shape3);
                    int ty = min(1024 / tx, out_shape2);
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
                                {backward_body}
                    }}
                    cudaMemsetAsync(out_p, 0, out->size);
                    int tx = min(1024, pout_shape3);
                    int ty = min(1024 / tx, pout_shape2);
                    int bx = (pout_shape2 - 1) / ty + 1;
                    int by = pout_shape1;
                    int bz = pout_shape0;
                    dim3 s1_(bx, by, bz);
                    dim3 s2_(tx, ty);
                    kernel3<<<s1_, s2_>>>(@ARGS);
                '''],
                cpu_header='#include <ops/binary_op_defs.h>',
                cpu_src=f'''
                    using namespace std;
                    for (int i0=0; i0<out_shape0; i0++)
                    for (int i1=0; i1<out_shape1; i1++)
                    for (int i2=0; i2<out_shape2; i2++)
                    for (int i3=0; i3<out_shape3; i3++)
                        {forward_body}
                ''',
                cpu_grad_src = [f'''
                    using namespace std;
                    std::memset(out_p, 0, out->size);
                    #define atomicAdd(a,b) (*a) += b

                    for (int i0=0; i0<pout_shape0; i0++)
                    for (int i1=0; i1<pout_shape1; i1++)
                    for (int i2=0; i2<pout_shape2; i2++) 
                    for (int i3=0; i3<pout_shape3; i3++)
                        {backward_body}
                '''])
            return out
        else:
            # TODO: backward 
            xx = x.reindex([N,C,h,w,self.kernel_size,self.kernel_size], [
                "i0", # Nid
                "i1", # Cid
                f"i2*{self.stride}-{self.padding}+i4", # Hid
                f"i3*{self.stride}-{self.padding}+i5", # Wid
            ])
            return xx.reduce(self.op, [4,5])


class Pool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None
        self.kernel_size = kernel_size
        self.op = op
        self.stride = stride if stride else kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad and padding != 0

    def execute(self, x):
        N,C,H,W,D = x.shape
        if self.ceil_mode == False:
            h = (H+self.padding*2-self.kernel_size)//self.stride+1
            w = (W+self.padding*2-self.kernel_size)//self.stride+1
            d = (D+self.padding*2-self.kernel_size)//self.stride+1
        else:
            h = (H+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1
            w = (W+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1
            d = (D+self.padding*2-self.kernel_size + self.stride - 1)//self.stride+1

        if self.op in ['maximum', 'minimum', 'mean']:
            if self.op == 'mean':
                if self.count_include_pad:
                    count = f"int count = {self.kernel_size*self.kernel_size*self.kernel_size};"
                else:
                    count = "int count = (k2_ - k2) * (k3_ - k3)* (k4_ - k4);"
                count += "float32 rcount = 1.0f / count;"
            else:
                count = ""
            forward_body = f'''{{
                int k4 = i4*{self.stride}-{self.padding};
                int k3 = i3*{self.stride}-{self.padding};
                int k2 = i2*{self.stride}-{self.padding};
                int k4_ = min(k4 + {self.kernel_size}, in0_shape4);
                int k3_ = min(k3 + {self.kernel_size}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size}, in0_shape2);
                k4 = max(0, k4);
                k3 = max(0, k3);
                k2 = max(0, k2);
                @out(i0, i1, i2, i3, i4) = init_{self.op}(out_type);
                {count}
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        for (int r = k4; r < k4_; ++r)
                            @out(i0, i1, i2, i3, i4) = {self.op}(out_type, @out(i0, i1, i2, i3, i4), @in0(i0, i1, p, q, r));
            }}'''
            backward_body = f'''{{
                int k4 = i4*{self.stride}-{self.padding};
                int k3 = i3*{self.stride}-{self.padding};
                int k2 = i2*{self.stride}-{self.padding};
                int k4_ = min(k4 + {self.kernel_size}, in0_shape4);
                int k3_ = min(k3 + {self.kernel_size}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size}, in0_shape2);
                k4 = max(0, k4);
                k3 = max(0, k3);
                k2 = max(0, k2);
                {count}
                int bo=1;
                for (int p = k2; p < k2_ && bo; ++p)
                    for (int q = k3; q < k3_ && bo; ++q) 
                        for (int r = k4; r < k4_ && bo; ++r) {{
                        {"atomicAdd(&@out(i0,i1,p,q,r), @dout(i0,i1,i2,i3,i4)/count);"
                            if self.op == "mean" else
                        f"""if (@pout(i0,i1,i2,i3,i4) == @in0(i0,i1,p,q,r)) {{
                            atomicAdd(&@out(i0,i1,p,q,r), @dout(i0,i1,i2,i3,i4)),
                            bo=0;
                        }}"""}
                    }}
            }}'''
            
            out = jt.code([N,C,h,w,d], x.dtype, [x],
                cuda_header="""
                    #include <ops/binary_op_defs.h>
                    #include <misc/cuda_limits.h>
                """,
                cuda_src=f'''
                    __global__ static void kernel1(@ARGS_DEF) {{
                        @PRECALC
                        int res_x = (in0_shape4 - 1) / blockDim.x + 1;
                        int res_y = (in0_shape3 - 1) / blockDim.y + 1;
                        int res_z = (in0_shape2 - 1) / blockDim.z + 1;

                        int idx4 = blockIdx.x / (res_y * res_z);
                        int idx3 = (blockIdx.x - idx4 * res_y * res_z) / res_z;
                        int idx2 = blockIdx.x - idx4 * res_y * res_z - idx3 * res_y;


                        int p4 = threadIdx.x + idx4 * blockDim.x;
                        int s4 = blockDim.x * res_x;

                        int p3 = threadIdx.y + idx3 * blockDim.y;
                        int s3 = blockDim.y * res_y;

                        int p2 = threadIdx.z + idx2 * blockDim.z;
                        int s2 = blockDim.z * res_z;

                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i4 = p4; i4 < out_shape4; i4 += s4)
                        for (int i3 = p3; i3 < out_shape3; i3 += s3)
                        for (int i2 = p2; i2 < out_shape2; i2 += s2)
                            {forward_body}
                    }}

                    int tx = min(1024, out_shape4);
                    int ty = min(1024 / tx, out_shape3);
                    int tz = min(1024 / tx / ty, out_shape2);


                    int res_x = (out_shape4 - 1) / tx + 1;
                    int res_y = (out_shape3 - 1) / ty + 1;
                    int res_z = (out_shape2 - 1) / tz + 1;

                    

                    int bx = res_x * res_y * res_z;
                    int by = out_shape1;
                    int bz = out_shape0;

                    dim3 s1(bx, by, bz);
                    dim3 s2(tx, ty, tz);
                    kernel1<<<s1, s2>>>(@ARGS);
                ''',
                cuda_grad_src=[f'''
                    __global__ static void kernel3(@ARGS_DEF) {{
                        @PRECALC


                        int res_x = (in0_shape4 - 1) / blockDim.x + 1;
                        int res_y = (in0_shape3 - 1) / blockDim.y + 1;
                        int res_z = (in0_shape2 - 1) / blockDim.z + 1;

                        int idx4 = blockIdx.x / (res_y * res_z);
                        int idx3 = (blockIdx.x - idx4 * res_y * res_z) / res_z;
                        int idx2 = blockIdx.x - idx4 * res_y * res_z - idx3 * res_y;


                        int p4 = threadIdx.x + idx4 * blockDim.x;
                        int s4 = blockDim.x * res_x;

                        int p3 = threadIdx.y + idx3 * blockDim.y;
                        int s3 = blockDim.y * res_y;

                        int p2 = threadIdx.z + idx2 * blockDim.z;
                        int s2 = blockDim.z * res_z;

                        int i1 = blockIdx.y;
                        int i0 = blockIdx.z;
                        for (int i4 = p4; i4 < pout_shape4; i4 += s4)
                            for (int i3 = p3; i3 < pout_shape3; i3 += s3)
                                for (int i2 = p2; i2 < pout_shape2; i2 += s2)
                                    {backward_body}
                    }}
                    cudaMemsetAsync(out_p, 0, out->size);

                    int tx = min(1024, pout_shape4);
                    int ty = min(1024 / tx, pout_shape3);
                    int tz = min(1024 / tx / ty, pout_shape2);

                    int res_x = (pout_shape4 - 1) / tx + 1;
                    int res_y = (pout_shape3 - 1) / ty + 1;
                    int res_z = (pout_shape2 - 1) / tz + 1;

                    int bx = res_x * res_y * res_z;

                    int by = pout_shape1;

                    int bz = pout_shape0;
                    dim3 s1_(bx, by, bz);
                    dim3 s2_(tx, ty, tz);
                    kernel3<<<s1_, s2_>>>(@ARGS);
                '''],
                cpu_header='#include <ops/binary_op_defs.h>',
                cpu_src=f'''
                    using namespace std;
                    for (int i0=0; i0<out_shape0; i0++)
                    for (int i1=0; i1<out_shape1; i1++)
                    for (int i2=0; i2<out_shape2; i2++)
                    for (int i3=0; i3<out_shape3; i3++)
                    for (int i4=0; i4<out_shape4; i4++)
                        {forward_body}
                ''',
                cpu_grad_src = [f'''
                    using namespace std;
                    std::memset(out_p, 0, out->size);
                    #define atomicAdd(a,b) (*a) += b
                    for (int i0=0; i0<pout_shape0; i0++)
                    for (int i1=0; i1<pout_shape1; i1++)
                    for (int i2=0; i2<pout_shape2; i2++) 
                    for (int i3=0; i3<pout_shape3; i3++)
                    for (int i4=0; i4<pout_shape4; i4++)
                        {backward_body}
                '''])
            return out
        else:
            # TODO: backward 
            xx = x.reindex([N,C,h,w,d,self.kernel_size,self.kernel_size,self.kernel_size], [
                "i0", # Nid
                "i1", # Cid
                f"i2*{self.stride}-{self.padding}+i5", # Hid
                f"i3*{self.stride}-{self.padding}+i6", # Wid
                f"i4*{self.stride}-{self.padding}+i7", # Did
            ])
            return xx.reduce(self.op, [5,6,7])

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
            raise TypeError(f"AdaptiveAvgPool2d only support int, tuple or list input. Not support {type(self.output_size)} yet.")
        if oh == 1 and ow == 1:
            return x.reduce("mean", [2,3], keepdims=True)
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