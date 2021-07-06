# ***************************************************************
# Copyright (c) 2021 Jittor. All Rights Reserved. 
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#
# 
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
        assert return_indices == None or op == "maximum"
        self.return_indices = return_indices
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.op = op
        stride = stride if stride else kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad and padding != 0

    def execute(self, x):
        N,C,H,W = x.shape
        if self.ceil_mode == False:
            h = (H+self.padding[0]*2-self.kernel_size[0])//self.stride[0]+1
            w = (W+self.padding[1]*2-self.kernel_size[1])//self.stride[1]+1
            use_code_op = self.op in ['maximum', 'minimum']
            # some second order avg_pool is require, so we don't use code op here  
        else:
            h = (H+self.padding[0]*2-self.kernel_size[0] + self.stride[0] - 1)//self.stride[0]+1
            w = (W+self.padding[1]*2-self.kernel_size[1] + self.stride[1] - 1)//self.stride[1]+1
            use_code_op = self.op in ['maximum', 'minimum', 'mean']

        if use_code_op:
            if self.op == 'mean':
                if self.count_include_pad:
                    count = f"int count = {self.kernel_size[0]*self.kernel_size[1]};"
                else:
                    count = "int count = (k2_ - k2) * (k3_ - k3);"
                count += "float32 rcount = 1.0f / count;"
            else:
                count = ""
            forward_body = f'''
                int k3 = i3*{self.stride[1]}-{self.padding[1]};
                int k2 = i2*{self.stride[0]}-{self.padding[0]};
                int k3_ = min(k3 + {self.kernel_size[1]}, in0_shape3);
                int k2_ = min(k2 + {self.kernel_size[0]}, in0_shape2);
                k3 = max(0, k3);
                k2 = max(0, k2);
                {count}
            '''
            if not self.return_indices:
                forward_body += f'''
                @out(i0, i1, i2, i3) = init_{self.op}(out_type);
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        @out(i0, i1, i2, i3) = {self.op}(out_type, @out(i0, i1, i2, i3), @in0(i0, i1, p, q));
                '''
            else:
                forward_body += f'''
                auto out_value = init_{self.op}(out_type);
                int out_index = -1;
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q) 
                        if (out_value < @in0(i0, i1, p, q)) {{
                            out_value = @in0(i0, i1, p, q);
                            out_index = (p - k2) * {self.kernel_size[1]} + (q - k3);
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
            '''
            if self.return_indices:
                return_shapes = [[N,C,h,w]] * 2
                return_dtypes = [x.dtype, 'uint8']
                ks = self.kernel_size[0] * self.kernel_size[1]
                if ks > 65536: return_dtypes[1] = 'uint32'
                elif ks > 256: return_dtypes[1] = 'uint16'
            else:
                return_shapes = [N,C,h,w]
                return_dtypes = x.dtype
            out = jt.code(return_shapes, return_dtypes, [x],
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
                            {{ {forward_body} }}
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
                                {{ {backward_body} }}
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
                        {{ {forward_body} }}
                ''',
                cpu_grad_src = [f'''
                    using namespace std;
                    std::memset(out_p, 0, out->size);
                    #define atomicAdd(a,b) (*a) += b

                    for (int i0=0; i0<pout_shape0; i0++)
                    for (int i1=0; i1<pout_shape1; i1++)
                    for (int i2=0; i2<pout_shape2; i2++) 
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

def _triple(x):
    if isinstance(x, tuple):
        assert len(x) == 3
        return x
    else:
        return (x,x,x)


class Pool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False, count_include_pad=True, op="maximum"):
        assert dilation == None
        assert return_indices == None or op == "maximum"
        self.return_indices = return_indices
        self.kernel_size = _triple(kernel_size)
        self.op = op
        stride = stride if stride else kernel_size
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad and padding != 0

    def execute(self, x):
        N,C,D,H,W = x.shape
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

        if use_code_op:
            if self.op == 'mean':
                if self.count_include_pad:
                    count = f"int count = {self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2]};"
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
                @out(i0, i1, i2, i3, i4) = init_{self.op}(out_type);
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q)
                        for (int r = k4; r < k4_; ++r)
                            @out(i0, i1, i2, i3, i4) = {self.op}(out_type, @out(i0, i1, i2, i3, i4), @in0(i0, i1, p, q, r));
                '''
            else:
                forward_body += f'''
                auto out_value = init_{self.op}(out_type);
                int out_index = -1;
                for (int p = k2; p < k2_; ++p)
                    for (int q = k3; q < k3_; ++q) 
                        for (int r = k4; q < k4_; ++r) 
                        if (out_value < @in0(i0, i1, p, q, r)) {{
                            out_value = @in0(i0, i1, p, q, r);
                            out_index = (p - k2) * {self.kernel_size[1]} * {self.kernel_size[2]} + (q - k3) * {self.kernel_size[2]} + (r - k4);
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
                    for (int q = k3; q < k3_ && bo; ++q) 
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
                return_dtypes = [x.dtype, 'uint8']
                ks = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]
                if ks > 65536: return_dtypes[1] = 'uint32'
                elif ks > 256: return_dtypes[1] = 'uint16'
            else:
                return_shapes = [N,C,d,h,w]
                return_dtypes = x.dtype
            out = jt.code(return_shapes, return_dtypes, [x],
                cuda_header="""
                    #include <ops/binary_op_defs.h>
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
                    int tx = min(1024, out_shape4);
                    int ty = min(1024 / tx, out_shape3);
                    int tz = min(1024 / tx / ty, out_shape2);
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
                        for (int i4 = p4; i4 < out_shape4; i4 += s4)
                        for (int i3 = p3; i3 < out_shape3; i3 += s3)
                        for (int i2 = p2; i2 < out_shape2; i2 += s2)
                                {{ {backward_body} }}
                    }}
                    cudaMemsetAsync(out_p, 0, out->size);
                    int tx = min(1024, pout_shape4);
                    int ty = min(1024 / tx, pout_shape3);
                    int tz = min(1024 / tx / ty, pout_shape2);
                    int bx = (pout_shape2 - 1) / tz + 1;
                    int by = pout_shape1;
                    int bz = pout_shape0;
                    dim3 s1(bx, by, bz);
                    dim3 s2(tx, ty, tz);
                    kernel3<<<s1, s2>>>(@ARGS);
                '''],
                cpu_header='#include <ops/binary_op_defs.h>',
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
                    for (int i2=0; i2<pout_shape2; i2++) 
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

class AdaptiveMaxPool2d(Module):
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
            raise TypeError(f"AdaptiveMaxPool2d only support int, tuple or list input. Not support {type(self.output_size)} yet.")
        if oh == 1 and ow == 1:
            return x.reduce("maximum", [2,3], keepdims=True)
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
        return xx.reduce("maximum", [4,5])


class AdaptiveAvgPool3d(Module):
    def __init__(self, output_size):
        self.output_size = _triple(output_size)

    def execute(self, x):
        od, oh, ow = self.output_size
        if od == 1 and oh == 1 and ow == 1:
            return x.reduce("mean", [2,3,4], keepdims=True)
        N,C,D,H,W = x.shape
        self.sd = math.floor(D / od)
        self.sh = math.floor(H / oh)
        self.sw = math.floor(W / ow)
        self.ksd = D - (od - 1) * self.sd
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        d = (D-self.ksd)//self.sd+1
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,d,h,w,self.ksd,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sd}+i5", # Did
            f"i3*{self.sh}+i6", # Hid
            f"i4*{self.sw}+i7", # Wid
        ])
        return xx.reduce("mean", [5,6,7])

class AdaptiveMaxPool3d(Module):
    def __init__(self, output_size):
        self.output_size = _triple(output_size)

    def execute(self, x):
        od, oh, ow = self.output_size
        if od == 1 and oh == 1 and ow == 1:
            return x.reduce("maximum", [2,3,4], keepdims=True)
        N,C,D,H,W = x.shape
        self.sd = math.floor(D / od)
        self.sh = math.floor(H / oh)
        self.sw = math.floor(W / ow)
        self.ksd = D - (od - 1) * self.sd
        self.ksh = H - (oh - 1) * self.sh
        self.ksw = W - (ow - 1) * self.sw
        d = (D-self.ksd)//self.sd+1
        h = (H-self.ksh)//self.sh+1
        w = (W-self.ksw)//self.sw+1
        xx = x.reindex([N,C,d,h,w,self.ksd,self.ksh,self.ksw], [
            "i0", # Nid
            "i1", # Cid
            f"i2*{self.sd}+i5", # Did
            f"i3*{self.sh}+i6", # Hid
            f"i4*{self.sw}+i7", # Wid
        ])
        return xx.reduce("maximun", [5,6,7])

def pool(x, kernel_size, op, padding=0, stride=None):
    return Pool(kernel_size, stride, padding, op=op)(x)

pool2d = pool

def pool3d(x, kernel_size, op, padding=0, stride=None):
    return Pool3d(kernel_size, stride, padding, op=op)(x)

class AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        self.layer = Pool(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, op="mean")
    
    def execute(self, x):
        return self.layer(x)

class AvgPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
        self.layer = Pool3d(kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode, count_include_pad=count_include_pad, op="mean")
    
    def execute(self, x):
        return self.layer(x)

def avg_pool2d(x, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
    return AvgPool2d(kernel_size, stride, padding, ceil_mode, count_include_pad)(x)

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
        self._layer = Pool(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, op="maximum")
    
    def execute(self, x):
        return self._layer(x)

class MaxPool3d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
        self._layer = Pool3d(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode, op="maximum")
    
    def execute(self, x):
        return self._layer(x)

def max_pool2d(x, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
    return MaxPool2d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)(x)


def max_pool3d(x, kernel_size, stride=None, padding=0, dilation=None, return_indices=None, ceil_mode=False):
    return MaxPool3d(kernel_size, stride, padding, dilation, return_indices, ceil_mode)(x)

class MaxUnpool2d(Module):
    ''' MaxUnpool2d is the invert version of MaxPool2d with indices.
    It takes the output index of MaxPool2d as input.
    The element will be zero if it is not the max pooled value.

    Example::

    >>> import jittor as jt
    >>> from jittor import nn

    >>> pool = nn.MaxPool2d(2, stride=2, return_indices=True)
    >>> unpool = nn.MaxUnpool2d(2, stride=2)
    >>> input = jt.array([[[[ 1.,  2,  3,  4,0],
                            [ 5,  6,  7,  8,0],
                            [ 9, 10, 11, 12,0],
                            [13, 14, 15, 16,0],
                            [0,  0,  0,  0, 0]]]])
    >>> output, indices = pool(input)
    >>> unpool(output, indices, output_size=input.shape)
    jt.array([[[[   0.,  0.,   0.,   0.,   0.],
                [   0.,  6.,   0.,   8.,   0.],
                [   0.,  0.,   0.,   0.,   0.],
                [   0., 14.,   0.,  16.,   0.],
                [   0.,  0.,   0.,   0.,   0.]]]])
    '''
    def __init__(self, kernel_size, stride=None):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if stride is None: stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride

    def execute(self, x, id, output_size=None):
        b, c, ph, pw = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        if output_size:
            h, w = output_size[-2:]
        else:
            h, w = ph * sh, pw * sw
        if self.stride == self.kernel_size:
            x = x.reindex(shape=[b, c, h, w], 
                indexes=['i0', 'i1', f'i2/{kh}', f'i3/{kw}'],
                extras=[id], 
                overflow_conditions=[
                    f'((i2%{kh})*{kw}+i3%{kw}) != @e0(i0,i1,i2/{kh},i3/{kw})'],
                overflow_value=0)
        else:
            x = x.reindex_reduce(
                op="add",
                shape=[b, c, h, w], 
                indexes=['i0', 'i1', 
                    f'i2*{sh}+@e0(i0,i1,i2,i3)/{kw}', 
                    f'i3*{sw}+@e0(i0,i1,i2,i3)%{kw}'],
                extras=[id], 
            )
        return x

class MaxUnpool3d(Module):
    ''' MaxUnpool3d is the invert version of MaxPool3d with indices.
    It takes the output index of MaxPool3d as input.
    The element will be zero if it is not the max pooled value.
    '''
    def __init__(self, kernel_size, stride=None):
        if stride is None: stride = kernel_size
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        self.kernel_size = kernel_size
        self.stride = stride

    def execute(self, x, id, output_size=None):
        b, c, pd, ph, pw = x.shape
        kd, kh, kw = self.kernel_size
        sd, sh, sw = self.stride
        if output_size:
            d, h, w = output_size[-3:]
        else:
            d, h, w = pd * sd, ph * sh, pw * sw
        if self.stride == self.kernel_size:
            x = x.reindex(shape=[b, c, d, h, w], 
                indexes=['i0', 'i1', f'i2/{kd}', f'i3/{kh}', f'i4/{kw}'],
                extras=[id], 
                overflow_conditions=[
                    f'((i2%{kd})*{kh*kw}+(i3%{kh})*{kw}+i4%{kw}) != @e0(i0,i1,i2/{kd},i3/{kh},i4/{kw})'],
                overflow_value=0)
        else:
            x = x.reindex_reduce(
                op="add",
                shape=[b, c, d, h, w], 
                indexes=['i0', 'i1', 
                    f'i2*{sd}+@e0(i0,i1,i2,i3,i4)/{kh*kw}', 
                    f'i3*{sh}+@e0(i0,i1,i2,i3,i4)/{kw}%{kh}', 
                    f'i4*{sw}+@e0(i0,i1,i2,i3,i4)%{kw}'],
                extras=[id], 
            )
        return x
