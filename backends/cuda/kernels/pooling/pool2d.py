"""CUDA launch builders for 2D pooling; window math is shared with CPU."""


def pool2d_cuda_options(forward_body, backward_body):
    return {
        "cuda_header": """
                    #include <type/cuda_limits.h>
                """,
        "cuda_src": f'''
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
        "cuda_grad_src": [f'''
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
    }
