# ***************************************************************
# Copyright (c) 2022 Jittor. All Rights Reserved. 
# Maintainers: Dun Liang <randonlang@gmail.com>. 
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import unittest
import jittor as jt
import numpy as np
import os

def transpose0231(x):
    s0, s1, s2, s3 = x.shape
    asize = 16
    bsize = 16
    ILP = 2
    return jt.code([s0, s2, s3, s1], x.dtype, [x],
    cuda_header="#include <type/fp16_compute.h>\n#include <cassert>",
    cuda_src=f"""
    __global__ void kernel(in0_type* __restrict__ x, in0_type* __restrict__ y, int s0, int s1, int s2, int s3) {{
        __shared__ in0_type t[{asize*ILP}*{bsize*ILP+1}];
        int t3 = threadIdx.x % {bsize};
        int t1 = threadIdx.x / {bsize};
        int b3 = blockIdx.x;
        int b2 = blockIdx.y;
        int b0 = blockIdx.z;
        int x3 = 1;
        int x2 = s3;
        int x1 = s2*x2;
        int x0 = s1*x1;
        int y3 = 1;
        int y2 = s1;
        int y1 = s3*y2;
        int y0 = s2*y1;
        in0_type tmp[{ILP}];
        for (int i=0; i<(s1-1)/{asize*ILP}+1; i++)
        {{
            int _b3 = b3 * {bsize*ILP} + t3*{ILP};
            if (_b3 < s3) {{
                #pragma unroll
                for (int j=0; j<{ILP}; j++) {{
                    vload<sizeof(in0_type)*{ILP}>(
                        tmp,
                        &x[b0*x0+(t1*{ILP}+j+i*{asize*ILP})*x1+b2*x2+_b3*x3]
                    );
                    #pragma unroll
                    for (int k=0; k<{ILP}; k++)
                        t[(t1*{ILP}+j)*{bsize*ILP+1}+t3*{ILP}+k] = tmp[k];
                    
                }}
            }}
            __syncthreads();
            int t3_ = threadIdx.x % {asize};
            int t1_ = threadIdx.x / {asize};
            _b3 = b3 * {bsize*ILP} + t1_*{ILP};
            if (_b3 < s3) {{
                #pragma unroll
                for (int j=0; j<{ILP}; j++) {{
                    #pragma unroll
                    for (int k=0; k<{ILP}; k++) {{
                        tmp[k] =
                            t[(t3*{ILP}+k)*{bsize*ILP+1}+t1_*{ILP}+j];
                    }}
                    vload<sizeof(in0_type)*{ILP}>(
                        &y[b0*y0+b2*y1+(_b3+j)*y2+((t3*{ILP})+i*{asize*ILP})*y3],
                        tmp
                    );
                }}
            }}
            __syncthreads();
        }}
    }}
    int s0, s1, s2, s3;
    in0->shape.unpack(s0, s1, s2, s3);
    kernel<<<{{(s3-1)/{bsize*ILP}+1, s2, s0 }}, {bsize*asize}>>>
        (in0_p, out0_p, s0, s1, s2, s3);
    """)

def transpose0231_2(x):
    s0, s1, s2, s3 = x.shape
    asize = 16
    bsize = 8
    ILP = 2
    return jt.code([s0, s2, s3, s1], x.dtype, [x],
    cuda_header="#include <type/fp16_compute.h>\n#include <cassert>",
    cuda_src=f"""
    __global__ __launch_bounds__({asize*bsize}) void kernel(in0_type* __restrict__ x, in0_type* __restrict__ y, int s0, int s1, int s2, int s3) {{
        __shared__ in0_type t[{asize*ILP}*{bsize*ILP+1}];
        int t3 = threadIdx.x % {bsize};
        int t1 = threadIdx.x / {bsize};
        int b3 = blockIdx.x;
        int b1 = blockIdx.y;
        int b2 = 0;
        int b0 = blockIdx.z;
        int x3 = 1;
        int x2 = s3;
        int x1 = s2*x2;
        int x0 = s1*x1;
        int y3 = 1;
        int y2 = s1;
        int y1 = s3*y2;
        int y0 = s2*y1;
        in0_type tmp[{ILP}];
        {{
            int _b3 = b3 * {bsize*ILP} + t3*{ILP};
            if (_b3 < s3) {{
                #pragma unroll
                for (int j=0; j<{ILP}; j++) {{
                    if (t1*{ILP}+j+b1*{asize*ILP} >= s1)
                        continue;
                    vload<sizeof(in0_type)*{ILP}>(
                        tmp,
                        &x[b0*x0+(t1*{ILP}+j+b1*{asize*ILP})*x1+b2*x2+_b3*x3]
                    );
                    #pragma unroll
                    for (int k=0; k<{ILP}; k++)
                        t[(t1*{ILP}+j)*{bsize*ILP+1}+t3*{ILP}+k] = tmp[k];
                    
                }}
            }}
            __syncthreads();
            int t3_ = threadIdx.x % {asize};
            int t1_ = threadIdx.x / {asize};
            _b3 = b3 * {bsize*ILP} + t1_*{ILP};
            int yy3 = (t3_*{ILP})+b1*{asize*ILP};
            if (_b3 < s3 && yy3 < s1) {{
                #pragma unroll
                for (int j=0; j<{ILP}; j++) {{
                    #pragma unroll
                    for (int k=0; k<{ILP}; k++) {{
                        tmp[k] =
                            t[(t3_*{ILP}+k)*{bsize*ILP+1}+t1_*{ILP}+j];
                    }}
                    vload<sizeof(in0_type)*{ILP}>(
                        &y[b0*y0+b2*y1+(_b3+j)*y2+yy3*y3],
                        tmp
                    );
                    // printf("%d %d %d %d %d\\n", b0*y0+b2*y1+(_b3+j)*y2+yy3*y3,
                    //    b0, b2, (_b3+j), yy3);
                }}
            }}
            __syncthreads();
        }}
    }}
    int s0, s1, s2, s3;
    in0->shape.unpack(s0, s1, s2, s3);
    kernel<<<{{(s3-1)/{bsize*ILP}+1, (s1-1)/{asize*ILP}+1, s0 }}, {bsize*asize}>>>
        (in0_p, out0_p, s0, s1, s2, s3);
    """)

def check_share():
    return
    a = jt.rand((30, 32, 4, 2000)).float32()
    jt.code(a.shape, a.dtype, [a],
    cuda_header="#include <type/fp16_compute.h>\n#include <cassert>",
    cuda_src="""
    __global__ void kernel(in0_type* __restrict__ a, in0_type* __restrict__ b) {
        __shared__ float x[32*33];
        for (int i=0; i<3; i++) {
        ((float2*)&x[i])[0] = ((float2*)&a[i])[0];
        ((float2*)&b[i])[0] = ((float2*)&x[i+1])[0];
        }
    }
    kernel<<<1024,16*16>>>(in0_p, out0_p);
    """).sync()
    jt.sync_all(True)
    # print(a[0]+1)
    print("pass test")

class TestFP16(unittest.TestCase):
    def test_array(self):
        a = np.array([1,2,3], dtype="float16")
        b = jt.array(a)
        np.testing.assert_allclose(a, b.data)

    def test_add(self):
        a = np.array([1,2,3], dtype="float16")
        b = jt.array(a)
        c = b+b
        np.testing.assert_allclose(c.data, a+a)
        d = c.sum()
        np.testing.assert_allclose(d.data, [12])
        c = c+1
        print(c)

    def test_matmul(self):
        a = jt.random((100,100)).float16()
        b = jt.random((100,100)).float16()
        c = jt.matmul(a, b)
        c.sync()

    def test_bmm(self):
        a = jt.random((10,3,4)).float16()
        b = jt.random((10,4,5)).float16()
        c = jt.matmul(a, b)
        c.sync()

    def test_matmul_grad(self):
        a = jt.random((100,100)).float16()
        b = jt.random((100,100)).float16()
        c = jt.matmul(a, b)
        c.sync()
        da, db = jt.grad(c, [a,b])
        jt.sync_all()
        assert da.dtype == "float16"
        assert db.dtype == "float16"

    def test_array_random_auto_cast(self):
        a = jt.array([1.0,2.0])
        assert a.dtype == "float32"
        with jt.flag_scope(amp_reg=2+16):
            a = jt.array([1.0,2.0])
            assert a.dtype == "float16", a.dtype
            
        a = jt.random([10])
        assert a.dtype == "float32"
        with jt.flag_scope(amp_reg=2+16):
            a = jt.random([10])
            assert a.dtype == "float16", a.dtype

    def test_conv(self):
        a = jt.random((3,4,5,5)).float16()
        b = jt.random((4,4,3,3)).float16()
        c = jt.nn.conv(a, b)
        c.sync()

    def test_max(self):
        a = jt.random((100,)).float16()
        b = jt.random((100,)).float16()
        c = a.maximum(b)
        c.sync()

    def test_reduce_dtype_infer(self):
        with jt.flag_scope(amp_reg=1):
            a = jt.random((3,4,5,5)).float16()
            b = a.sum()
            b.sync()
            assert b.dtype == "float32"
        with jt.flag_scope(amp_reg=2):
            a = jt.random((3,4,5,5)).float16()
            b = a.sum()
            b.sync()
            assert b.dtype == "float32"
        with jt.flag_scope(amp_reg=0):
            a = jt.random((3,4,5,5)).float16()
            b = a.sum()
            b.sync()
            assert b.dtype == "float32"
        with jt.flag_scope(amp_reg=2+4):
            a = jt.random((3,4,5,5)).float16()
            b = a.sum()
            b.sync()
            assert b.dtype == "float16", b.dtype

    def test_white_dtype_infer(self):
        with jt.flag_scope(amp_reg=1):
            a = jt.random((3,4,5,5)).float16()
            b = a**a
            b.sync()
            assert b.dtype == "float32"
        with jt.flag_scope(amp_reg=2):
            a = jt.random((3,4,5,5)).float16()
            b = a**a
            b.sync()
            assert b.dtype == "float32"
        with jt.flag_scope(amp_reg=0):
            a = jt.random((3,4,5,5)).float16()
            b = a**a
            b.sync()
            assert b.dtype == "float32"
        with jt.flag_scope(amp_reg=2+8):
            a = jt.random((3,4,5,5)).float16()
            b = a**a
            b.sync()
            assert b.dtype == "float16", b.dtype

    def test_module_half(self):
        a = jt.nn.Linear(10,10)
        assert a.weight.dtype == "float32"
        a.half()
        assert a.weight.dtype == "float16"



@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestFP16CUDA(TestFP16):
    def setUp(self):
        jt.flags.use_cuda = 1
    def tearDown(self):
        jt.flags.use_cuda = 0

    def test_softmax(self):
        a = jt.rand((120, 2000, 2000)).float16()
        # a = jt.rand((1, 2000, 2000)).float32()
        jt.sync_all()
        with jt.profile_scope(10, 100):
            a.log_softmax(-1).sync()

    def test_transpose(self):
        check_share()
        # return
        a = jt.rand((30, 32, 4, 2000)).float32()
        # a = jt.rand((1, 1024, 1, 2000)).float32()
        diff = transpose0231(a).data != a.transpose((0,2,3,1)).data
        print(np.where(diff))
        # return
        jt.sync_all()
        # with jt.profile_scope(100, 11000):
        with jt.profile_scope(100, 11000):
            # a.log_softmax(-1).sync()
            transpose0231(a).sync()

            a.transpose((0,2,3,1)).sync()
            # a.transpose((0,2,1,3)).sync()
            a.fuse_transpose((0,2,1,3)).sync()
            (a+1).sync()
        jt.sync_all(True)
        diff = transpose0231(a).data != a.transpose((0,2,3,1)).data
        print(np.where(diff))
        np.testing.assert_allclose(transpose0231(a).data, a.transpose((0,2,3,1)).data)

    def test_transpose2(self):
        # check_share()
        # return
        # a = jt.rand((30, 32, 4, 2000)).float32()
        # a = jt.rand((1, 10000, 1, 2000)).float32()
        a = jt.rand((1, 10000, 1, 2048)).float32()
        print("transpose")
        transpose0231_2(a).sync()
        print("add")
        (a+1).sync()
        return
        # a = jt.arange(32*16).reshape((1, 32, 1, 16))
        diff = transpose0231_2(a).data != a.transpose((0,2,3,1)).data
        print(np.where(diff))
        # return
        jt.sync_all()
        # with jt.profile_scope(100, 11000):
        with jt.profile_scope(100, 1100):
            # a.log_softmax(-1).sync()
            transpose0231_2(a).sync()

            a.transpose((0,2,3,1)).sync()
            # a.transpose((0,2,1,3)).sync()
            a.fuse_transpose((0,2,1,3)).sync()
            (a+1).sync()
        jt.sync_all(True)
        diff = transpose0231_2(a).data != a.transpose((0,2,3,1)).data
        print(np.where(diff))
        np.testing.assert_allclose(transpose0231_2(a).data, a.transpose((0,2,3,1)).data)
        
if __name__ == "__main__":
    unittest.main()