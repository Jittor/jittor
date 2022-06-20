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

n = 400000000
# n = 4000000
n = 7680000

def get_mem_band():
    a = jt.rand((n)).float32()
    for i in range(100):
        a.copy().sync()
    jt.sync_all(True)
    import time
    t = time.time()
    for i in range(1000):
        a.copy().sync()
    jt.sync_all(True)
    dt = time.time() - t
    band = a.numel() * 4 * 2000 / dt / 1024**3
    print("Mem band: ", band)
    return band

def check_simple_add_band():
    # copy: 816
    # S=1 128,1024, ILP=1 634
    # S=0 128,1024, ILP=1 734
    # S=0 128,512, ILP=1 716
    # S=0 64,1024, ILP=1 706
    # S=0 256,1024, ILP=1 706
    def test(S=0, B=128, T=1024, ILP=1):
        a = jt.rand((n)).float32()
        jt.sync_all(True)
        jt.flags.log_silent = 1
        with jt.profile_scope(100, 1000) as rep:
            b = jt.code(a.shape, a.dtype, [a], 
            cuda_header="#include \"type/fp16_compute.h\"",
            cuda_src=f"""
            __global__ void kernel(in0_type * __restrict__ a, in0_type* __restrict__ b, int num) {{
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                int tnum = blockDim.x * gridDim.x;
                #define ILP {ILP}
                for (int i=tid*ILP; i<num; i+=tnum*ILP) {{
                    // b[i] = a[i];
                    vload<ILP*sizeof(in0_type)>(b+i, a+i);
                    {"__syncthreads();" if S else ""}
                }}
            }}
            kernel<<<{B},{T}>>>(in0_p, out0_p, in0->num);
            """)
            b.sync()
        bw = float(rep[-1][9]) / 1024**3
        s = f"S={S}, B={B}, T={T}, ILP={ILP} BW={bw}"
        print(s)
        return s, bw

    def test2(S=0, B=128, T=1024, ILP=1):
        a = jt.rand((n)).float32()
        jt.sync_all(True)
        # jt.flags.log_silent = 0
        with jt.profile_scope(10, 1000) as rep:
            b = jt.code(a.shape, a.dtype, [a], 
            cuda_header="#include \"type/fp16_compute.h\"",
            cuda_src=f"""
            __global__ void kernel(float2 * __restrict__ a, float2* __restrict__ b, int num) {{
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                int tnum = blockDim.x * gridDim.x;
                #define ILP 1
                for (int i=tid*ILP; i<num; i+=tnum*ILP) {{
                    b[i] = a[i];
                    // b[i+1] = a[i+1];
                    // vload<ILP*sizeof(in0_type)>(b+i, a+i);
                    {"__syncthreads();" if S else ""}
                }}
            }}
            kernel<<<{B},{T}>>>((float2*)in0_p, (float2*)out0_p, in0->num/2);
            """)
            b.sync()
        bw = float(rep[-1][9]) / 1024**3
        s = f"T2: S={S}, B={B}, T={T}, ILP={ILP} BW={bw}"
        print(s)
        return s, bw

            
    def test3(S=0, B=128, T=1024, ILP=1, C=0):
        a = jt.rand((n)).float32()
        b = jt.rand(B)
        jt.sync_all(True)
        jt.flags.log_silent = 1
        with jt.profile_scope(100, 1000) as rep:
            b = jt.code(a.shape, a.dtype, [a, b], 
            cuda_header="#include \"type/fp16_compute.h\"",
            cuda_src=f"""
            __global__ void kernel(in0_type * __restrict__ a, in0_type* __restrict__ b, int num) {{
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                int tnum = blockDim.x * gridDim.x;
                #define ILP {ILP}
                for (int i=tid*ILP; i<num; i+=tnum*ILP) {{
                    // b[i] = a[i];
                    vload<ILP*sizeof(in0_type)>(b+i, a+i);
                    {"__syncthreads();" if S else ""}
                }}
                {"__syncthreads();" if C else ""}
            }}
            kernel<<<in1->shape[0],{T}>>>(in0_p, out0_p, in0->num);
            """)
            b.compile_options = {"FLAGS: -Xptxas -dlcm=ca ": C}
            # b.compile_options = {"FLAGS: –Xptxas –dlcm=ca ": 1}
            b.sync()

        bw = float(rep[-1][9]) / 1024**3
        s = f"T3: S={S}, B={B}, T={T}, ILP={ILP} C={C} BW={bw}"
        print(s)
        return s, bw

            
    def test4(S=0, B=128, T=1024, ILP=1, C=0, name="b.png"):
        a = jt.rand((n)).float32()
        b = jt.rand(B*4).uint32()
        jt.sync_all(True)
        # jt.flags.log_silent = 1
        with jt.profile_scope(100, 10000) as rep:
            _ = jt.code(a.shape, a.dtype, [a, b], 
            cuda_header="#include \"type/fp16_compute.h\"",
            cuda_src=f"""
            __device__ uint get_smid(void) {{
                uint ret;
                asm("mov.u32 %0, %smid;" : "=r"(ret) );
                return ret;
            }}
            __device__ uint get_time(void) {{
                uint ret;
                asm volatile("mov.u32 %0, %%globaltimer_lo;" : "=r"(ret));
                return ret;
            }}

            __global__ void kernel(in0_type * __restrict__ a, in0_type* __restrict__ b, int num, in1_type* __restrict__ c) {{
                uint t = get_time();
                int tid = threadIdx.x + blockIdx.x * blockDim.x;
                int tnum = blockDim.x * gridDim.x;
                #define ILP {ILP}
                for (int i=tid*ILP; i<num; i+=tnum*ILP) {{
                    // b[i] = a[i];
                    vload<ILP*sizeof(in0_type)>(b+i, a+i);
                    {"__syncthreads();" if S else ""}
                }}
                {"__syncthreads();" if C else ""}
                if (threadIdx.x == 0)
                    ((uint4* __restrict__)c)[blockIdx.x] = 
                    uint4{{get_smid(), t, get_time(), 0}};
            }}
            kernel<<<in1->shape[0]/4,{T}>>>(in0_p, out0_p, in0->num, in1_p);
            """)
            _.compile_options = {"FLAGS: -Xptxas -dlcm=ca ": C}
            # b.compile_options = {"FLAGS: –Xptxas –dlcm=ca ": 1}
            _.sync()

        bw = float(rep[-1][9]) / 1024**3
        b = b.data.reshape(-1, 4)[:,:3]
        mint = b[:,1].min()
        b[:,1:] -= mint
        smmax = int(b[:,0].max())
        smmin = int(b[:,0].min())
        maxt = b.max()

        # print(b)

        s = f"T4: S={S}, B={B}, T={T}, ILP={ILP} C={C} BW={bw:.3f} sm={smmin},{smmax} maxt={maxt}"
        print(s)
        import pylab as pl
        pl.figure(figsize=(16,16))
        texts = []
        pret = np.zeros(200, dtype="uint32")
        for i in range(B):
            smid, s, t = b[i]
            pl.plot([s,t], [smid, smid], 'ro-')
            texts.append((s, smid, i))
            texts.append((t, smid, i))

        texts = sorted(texts)
        for (s, smid, bid) in texts:
            cpos = max(pret[smid], s)
            pl.text(cpos, smid, str(bid))
            pret[smid] = cpos + maxt // 30


        # print("???")
        # adjust_text(texts, arrowprops=dict(arrowstyle='->', color='blue'))
        # print("???")
        pl.savefig(name)
        pl.close()
        return s, bw
    # test(S=0, B=128, T=1024, ILP=1)
    # test(S=1, B=128, T=1024, ILP=1)
    # test(S=0, B=64, T=1024, ILP=1)
    # test(S=0, B=256, T=1024, ILP=1)
    # test(S=1, B=128, T=512, ILP=1)
    # test(S=1, B=128, T=256, ILP=1)
    
    # test(S=0, B=128, T=1024, ILP=2)
    # test(S=0, B=128, T=1024, ILP=4)
    # test(S=0, B=128, T=512, ILP=2)
    # test(S=0, B=128, T=512, ILP=4)
    
    # test(S=1, B=128, T=1024, ILP=2)
    # test(S=1, B=128, T=1024, ILP=4)
    # test(S=1, B=128, T=1024, ILP=8)
    # test(S=1, B=128, T=1024, ILP=16)
    # test(S=1, B=128, T=512, ILP=2)
    # test(S=1, B=128, T=512, ILP=4)
    
    # test(S=1, B=256, T=1024, ILP=2)
    # test(S=1, B=512, T=1024, ILP=2)
    # test(S=1, B=256, T=1024, ILP=4)
    # test(S=1, B=256, T=1024, ILP=8)
    # test(S=1, B=256, T=1024, ILP=16)
    # test(S=1, B=256, T=512, ILP=2)
    # test(S=1, B=256, T=512, ILP=4)
    
    # test(S=1, B=128, T=256, ILP=2)
    # test(S=1, B=128, T=256, ILP=4)
    # test(S=0, B=128, T=256, ILP=2)
    # test(S=0, B=128, T=256, ILP=4)
    
    # for b in [1, 2, 4, 8, 16, 32, 64, 128,256]:
    #     test(S=1, B=b, T=512, ILP=2)

    import matplotlib as mpl
    mpl.use('Agg')
    import pylab as pl
    import numpy as np

    # test4(S=1, B=82, T=1024, ILP=2, C=0, name="b.png")
    # test4(S=1, B=83, T=1024, ILP=2, C=0, name="c.png")
    # test4(S=1, B=82*3, T=512, ILP=2, C=0, name="d1.png")
    # test4(S=1, B=82*3+1, T=512, ILP=2, C=0, name="d2.png")
    # test4(S=1, B=82*6+1, T=512, ILP=2, C=0, name="d3.png")
    # test4(S=0, B=82*6+1, T=512, ILP=2, C=0, name="d4.png")
    
    for b in range(70, 83):
        test4(S=1, B=b, T=1024, ILP=2, C=0, name=f"b-{b}.png")

    # data = []
    # for b in range(32, 2000, 8):
    #     _, bw = test3(S=0, B=b, T=32, ILP=2)
    #     data.append([b, bw])
    # data = np.array(data)
    # pl.plot(data[:,0], data[:,1])

    # for t in [32, 64, 128, 256, 512, 1024]:
    #     data = []
    #     for b in range(32, 2000, 8):
    #         _, bw = test3(S=1, B=b*(1024//t), T=t, ILP=2)
    #         data.append([b, bw])
    #     data = np.array(data)
    #     pl.plot(data[:,0], data[:,1])

    # for t in [1024]:
    #     for c in [0,1]:
    #         data = []
    #         # for b in range(32, 1000, 8):
    #         for b in range(32, 33, 8):
    #             _, bw = test3(S=c, B=b*(1024//t), T=t, ILP=2, C=0)
    #             data.append([b, bw])
    #         data = np.array(data)
    #         pl.plot(data[:,0], data[:,1])
    
    # for ilp in [2]:
    #     for s in [1]:
    #         for t in [1024,512,256,128]:
    #             data = []
    #             for b in range(32, 1100, 8):
    #                 _, bw = test3(S=s, B=b*(1024//t), T=t, ILP=ilp)
    #                 data.append([b, bw])
    #             data = np.array(data)
    #             pl.plot(data[:,0], data[:,1])

    # pl.savefig("a.png")
    # pl.close()
    # for b in range(80, 90, 1):
    #     _, bw = test3(S=1, B=b, T=1024, ILP=2)
    # # 82
    # for b in range(240, 260, 1):
    #     _, bw = test3(S=1, B=b, T=512, ILP=2)
    # # 82*3 = 246
    # for b in range(240, 500, 1):
    #     _, bw = test3(S=1, B=b, T=256, ILP=2)
    # # 492 = 82*6
    # for b in range(240, 1000, 1):
    #     _, bw = test3(S=1, B=b, T=128, ILP=2)
    # # 984 = 82*12


    # for b in [128,256]:
    #     test(S=1, B=b, T=1024, ILP=2)
    # for b in [128,256]:
    #     test(S=0, B=b, T=512, ILP=2)
    # for b in [128,256]:
    #     test(S=0, B=b, T=1024, ILP=2)
    # for b in [128,256]:
    #     test(S=1, B=b, T=512, ILP=1)
    # for b in [128,256]:
    #     test(S=1, B=b, T=1024, ILP=1)
    # for b in [128,256]:
    #     test(S=0, B=b, T=512, ILP=1)
    # for b in [128,256]:
    #     test(S=0, B=b, T=1024, ILP=1)
    # test(S=1, B=128, T=512, ILP=4)
    # test(S=1, B=64, T=512, ILP=2)
    # test(S=1, B=80, T=512, ILP=2)
    # test(S=1, B=100, T=512, ILP=2)
    # test(S=1, B=110, T=512, ILP=2)
    # test(S=1, B=115, T=512, ILP=2)
    # test(S=1, B=120, T=512, ILP=2)
    # test(S=1, B=130, T=512, ILP=2)
    # test(S=1, B=140, T=512, ILP=2)
    # test2(S=1, B=128, T=512, ILP=2)
    # test(S=1, B=128, T=256, ILP=4)
    # test(S=1, B=128, T=128, ILP=8)
    # test(S=1, B=128, T=64, ILP=16)
    


@unittest.skipIf(not jt.compiler.has_cuda, "No CUDA found")
class TestBenchmarkCUDA(unittest.TestCase):
    def setUp(self):
        jt.flags.use_cuda = 1
    def tearDown(self):
        jt.flags.use_cuda = 0

    def test_main(self):
        return
        get_mem_band()
        check_simple_add_band()

if __name__ == "__main__":
    unittest.main()