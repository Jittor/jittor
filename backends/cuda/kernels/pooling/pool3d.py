"""CUDA launch builders for 3D pooling; window math is shared with CPU."""


def pool3d_cuda_options(forward_body, backward_body):
    return {
        "cuda_header": """
                    #include <type/cuda_limits.h>
                """,
        "cuda_src": f'''
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
        "cuda_grad_src": [f'''
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
    }
